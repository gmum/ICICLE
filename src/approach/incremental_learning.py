import time
import torch
import numpy as np
from argparse import ArgumentParser
import torch.nn.functional as F

from loggers.exp_logger import ExperimentLogger
from datasets.exemplars_dataset import ExemplarsDataset
from networks.protopartnet import PPNet
from networks.protopartnet import push as ppnet_push
from networks.tesnet import TesNet
from networks.tesnet import push as tesnet_push


class Inc_Learning_Appr:
    """Basic class for implementing incremental learning approaches"""

    def __init__(self, model, device, nepochs=100, lr=0.05, lr_min=1e-4, lr_factor=3, lr_patience=5, clipgrad=10000,
                 momentum=0, wd=0, multi_softmax=False, wu_nepochs=0, wu_lr_factor=1, fix_bn=False,
                 eval_on_train=False, logger: ExperimentLogger = None, exemplars_dataset: ExemplarsDataset = None):
        self.model = model
        self.device = device
        self.nepochs = nepochs
        self.lr = lr
        self.lr_min = lr_min
        self.lr_factor = lr_factor
        self.lr_patience = lr_patience
        self.clipgrad = clipgrad
        self.momentum = momentum
        self.wd = wd
        self.multi_softmax = multi_softmax
        self.logger = logger
        self.exemplars_dataset = exemplars_dataset
        self.warmup_epochs = wu_nepochs
        self.warmup_lr = lr * wu_lr_factor
        self.warmup_loss = torch.nn.CrossEntropyLoss()
        self.fix_bn = fix_bn
        self.eval_on_train = eval_on_train
        self.optimizer = None
        self.warm_optim = None
        self.joint_optim = None
        self.push_optim = None

    @staticmethod
    def extra_parser(args):
        """Returns a parser containing the approach specific parameters"""
        parser = ArgumentParser()
        return parser.parse_known_args(args)

    @staticmethod
    def exemplars_dataset_class():
        """Returns a exemplar dataset to use during the training if the approach needs it
        :return: ExemplarDataset class or None
        """
        return None

    def _get_optimizer(self):
        """Returns the optimizer"""
        if isinstance(self.model, PPNet):
            warm_params = [{'params': self.model.add_on_layers.parameters(), 'lr': 3 * self.lr, 'weight_decay': self.wd,
                            'momentum': self.momentum},
                           {'params': self.model.prototype_vectors, 'lr': 3 * self.lr},
                           ]
            joint_params = [{'params': self.model.features.parameters(), 'lr': self.lr / 10, 'weight_decay': self.wd,
                             'momentum': self.momentum},
                            {'params': self.model.add_on_layers.parameters(), 'lr': 3 * self.lr,
                             'weight_decay': self.wd, 'momentum': self.momentum},
                            {'params': self.model.prototype_vectors, 'lr': 3 * self.lr},
                            ]
            push_params = [{'params': self.model.last_layer.parameters(), 'lr': self.lr / 10,
                            'weight_decay': self.wd, 'momentum': self.momentum}, ]
            warm_optimizer = torch.optim.SGD(warm_params)
            joint_optimizer = torch.optim.SGD(joint_params)
            push_optimizer = torch.optim.SGD(push_params)
            return joint_optimizer, push_optimizer, warm_optimizer
        else:
            return torch.optim.SGD(self.model.parameters(), lr=self.lr, weight_decay=self.wd, momentum=self.momentum)

    def train(self, t, trn_loader, val_loader, push_loader=None):
        """Main train structure"""
        self.pre_train_process(t, trn_loader)
        self.train_loop(t, trn_loader, val_loader)
        self.post_train_process(t, trn_loader)

    def pre_train_process(self, t, trn_loader):
        """Runs before training all epochs of the task (before the train session)"""

        # Warm-up phase
        if self.warmup_epochs and t > 0:
            self.optimizer = torch.optim.SGD(self.model.heads[-1].parameters(), lr=self.warmup_lr)
            # Loop epochs -- train warm-up head
            for e in range(self.warmup_epochs):
                warmupclock0 = time.time()
                self.model.heads[-1].train()
                for images, targets in trn_loader:
                    outputs = self.model(images.to(self.device))
                    loss = self.warmup_loss(outputs[t], targets.to(self.device) - self.model.task_offset[t])
                    self.optimizer.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.heads[-1].parameters(), self.clipgrad)
                    self.optimizer.step()
                warmupclock1 = time.time()
                with torch.no_grad():
                    total_loss, total_acc_taw = 0, 0
                    self.model.eval()
                    for images, targets in trn_loader:
                        outputs = self.model(images.to(self.device))
                        loss = self.warmup_loss(outputs[t], targets.to(self.device) - self.model.task_offset[t])
                        pred = torch.zeros_like(targets.to(self.device))
                        for m in range(len(pred)):
                            this_task = (self.model.task_cls.cumsum(0) <= targets[m]).sum()
                            pred[m] = outputs[this_task][m].argmax() + self.model.task_offset[this_task]
                        hits_taw = (pred == targets.to(self.device)).float()
                        total_loss += loss.item() * len(targets)
                        total_acc_taw += hits_taw.sum().item()
                total_num = len(trn_loader.dataset.labels)
                trn_loss, trn_acc = total_loss / total_num, total_acc_taw / total_num
                warmupclock2 = time.time()
                print('| Warm-up Epoch {:3d}, time={:5.1f}s/{:5.1f}s | Train: loss={:.3f}, TAw acc={:5.1f}% |'.format(
                    e + 1, warmupclock1 - warmupclock0, warmupclock2 - warmupclock1, trn_loss, 100 * trn_acc))
                self.logger.log_scalar(task=t, iter=e + 1, name="loss", value=trn_loss, group="warmup")
                self.logger.log_scalar(task=t, iter=e + 1, name="acc", value=100 * trn_acc, group="warmup")

    def train_loop(self, t, trn_loader, val_loader):
        """Contains the epochs loop"""
        lr = self.lr
        best_loss = np.inf
        patience = self.lr_patience
        best_model = self.model.get_copy()

        if isinstance(self.model, PPNet):
            self.warm_optim, self.joint_optim, self.ll_optim = self._get_optimizer()
        else:
            self.optimizer = self._get_optimizer()

        # Loop epochs
        for e in range(self.nepochs):
            # Train
            clock0 = time.time()
            self.train_epoch(t, trn_loader)
            clock1 = time.time()
            if self.eval_on_train:
                train_loss, train_acc, _ = self.eval(t, trn_loader)
                clock2 = time.time()
                print('| Epoch {:3d}, time={:5.1f}s/{:5.1f}s | Train: loss={:.3f}, TAw acc={:5.1f}% |'.format(
                    e + 1, clock1 - clock0, clock2 - clock1, train_loss, 100 * train_acc), end='')
                self.logger.log_scalar(task=t, iter=e + 1, name="loss", value=train_loss, group="train")
                self.logger.log_scalar(task=t, iter=e + 1, name="acc", value=100 * train_acc, group="train")
            else:
                print('| Epoch {:3d}, time={:5.1f}s | Train: skip eval |'.format(e + 1, clock1 - clock0), end='')

            # Valid
            clock3 = time.time()
            valid_loss, valid_acc, _ = self.eval(t, val_loader)
            clock4 = time.time()
            print(' Valid: time={:5.1f}s loss={:.3f}, TAw acc={:5.1f}% |'.format(
                clock4 - clock3, valid_loss, 100 * valid_acc), end='')
            self.logger.log_scalar(task=t, iter=e + 1, name="loss", value=valid_loss, group="valid")
            self.logger.log_scalar(task=t, iter=e + 1, name="acc", value=100 * valid_acc, group="valid")

            # Adapt learning rate - patience scheme - early stopping regularization
            if valid_loss < best_loss:
                # if the loss goes down, keep it as the best model and end line with a star ( * )
                best_loss = valid_loss
                best_model = self.model.get_copy()
                patience = self.lr_patience
                print(' *', end='')
            else:
                # if the loss does not go down, decrease patience
                patience -= 1
                if patience <= 0:
                    # if it runs out of patience, reduce the learning rate
                    lr /= self.lr_factor
                    print(' lr={:.1e}'.format(lr), end='')
                    if lr < self.lr_min:
                        # if the lr decreases below minimum, stop the training session
                        print()
                        break
                    # reset patience and recover best model so far to continue training
                    patience = self.lr_patience
                    self.optimizer.param_groups[0]['lr'] = lr
                    self.model.set_state_dict(best_model)
            self.logger.log_scalar(task=t, iter=e + 1, name="patience", value=patience, group="train")
            self.logger.log_scalar(task=t, iter=e + 1, name="lr", value=lr, group="train")
            print()
        self.model.set_state_dict(best_model)

    def post_train_process(self, t, trn_loader):
        """Runs after training all the epochs of the task (after the train session)"""
        pass

    def train_epoch(self, t, trn_loader):
        """Runs a single epoch"""
        self.model.train()
        if self.fix_bn and t > 0:
            self.model.freeze_bn()
        for images, targets in trn_loader:
            if isinstance(self.model.model, PPNet) or isinstance(self.model.model, TesNet):
                outputs = self.model(images.to(self.device))
                logits = [outputs[i][1] for i in range(len(outputs))]
                min_distances = [outputs[i][2] for i in range(len(outputs))]
                entropy_loss = self.criterion(t, logits, targets.to(self.device))
                clst_loss_val, sep_loss_val, l1_loss, avg_separation_cost, orth_loss, sub_loss = self.protopnet_looses(
                    min_distances,
                    targets.to(self.device),
                    t,
                    all_out=self.exemplars_dataset is not None,
                )
                loss = entropy_loss + clst_loss_val * 0.8 + sep_loss_val * (-0.08) + 1e-4 * l1_loss + \
                       1 * orth_loss - 1e-7 * sub_loss
                # Backward
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clipgrad)
                self.optimizer.step()
            else:
                # Forward current model
                outputs = self.model(images.to(self.device))
                loss = self.criterion(t, outputs, targets.to(self.device))
                # Backward
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clipgrad)
                self.optimizer.step()

    def eval(self, t, val_loader):
        """Contains the evaluation code"""
        with torch.no_grad():
            total_loss, total_acc_taw, total_acc_tag, total_num = 0, 0, 0, 0
            self.model.eval()
            for images, targets in val_loader:
                # Forward current model
                outputs = self.model(images.to(self.device))
                loss = self.criterion(t, outputs, targets.to(self.device))
                hits_taw, hits_tag = self.calculate_metrics(outputs, targets)
                # Log
                total_loss += loss.item() * len(targets)
                total_acc_taw += hits_taw.sum().item()
                total_acc_tag += hits_tag.sum().item()
                total_num += len(targets)
        return total_loss / total_num, total_acc_taw / total_num, total_acc_tag / total_num

    def calculate_metrics(self, outputs, targets):
        """Contains the main Task-Aware and Task-Agnostic metrics"""
        pred = torch.zeros_like(targets.to(self.device))
        # Task-Aware Multi-Head
        for m in range(len(pred)):
            this_task = (self.model.task_cls.cumsum(0) <= targets[m]).sum()
            pred[m] = outputs[this_task][m].argmax() + self.model.task_offset[this_task]
        hits_taw = (pred == targets.to(self.device)).float()
        # Task-Agnostic Multi-Head
        if self.multi_softmax:
            outputs = [torch.nn.functional.log_softmax(output, dim=1) for output in outputs]
            pred = torch.cat(outputs, dim=1).argmax(1)
        else:
            pred = torch.cat(outputs, dim=1).argmax(1)
        hits_tag = (pred == targets.to(self.device)).float()
        return hits_taw, hits_tag

    def criterion(self, t, outputs, targets):
        """Returns the loss value"""
        return torch.nn.functional.cross_entropy(outputs[t], targets - self.model.task_offset[t])


class Inc_Learning_Appr_PPNet(Inc_Learning_Appr):
    """Basic class for implementing incremental learning approaches for prototypical parts"""

    def _get_optimizers(self, t):
        """Returns the optimizer"""
        warm_params = [{'params': self.model.model.add_on_layers.parameters(), 'lr': 3 * self.lr, 'weight_decay': self.wd},
                       {'params': self.model.heads[t].prototype_vectors, 'lr': 3 * self.lr},
                       ]
        joint_params = [{'params': self.model.model.features.parameters(), 'lr': self.lr / 10, 'weight_decay': self.wd},
                        {'params': self.model.model.add_on_layers.parameters(), 'lr': 3 * self.lr,
                         'weight_decay': self.wd},
                        {'params': self.model.heads[t].prototype_vectors, 'lr': 3 * self.lr},
                        ]
        push_params = [{'params': self.model.heads[i].last_layer.parameters(), 'lr': self.lr / 10,
                        'weight_decay': self.wd} for i in range(t + 1)]
        warm_optimizer = torch.optim.Adam(warm_params)
        joint_optimizer = torch.optim.Adam(joint_params)
        push_optimizer = torch.optim.Adam(push_params)
        return joint_optimizer, push_optimizer, warm_optimizer

    def train(self, t, trn_loader, val_loader, push_loader=None):
        """Main train structure"""
        self.train_loop(t, trn_loader, val_loader, push_loader)
        self.post_train_process(t, trn_loader)

    def train_loop(self, t, trn_loader, val_loader, push_loader=None):
        """Contains the epochs loop"""
        lr = self.lr
        best_loss = np.inf
        patience = self.lr_patience
        best_model = self.model.get_copy()

        self.joint_optim, self.push_optim, self.warm_optim = self._get_optimizers(t)

        # Loop epochs
        for e in range(self.nepochs):
            # Train
            clock0 = time.time()
            self.train_epoch(t, trn_loader, e)
            clock1 = time.time()
            if self.eval_on_train:
                train_loss, train_acc, _, train_ppnet_losses = self.eval(t, trn_loader)
                clock2 = time.time()
                print('| Epoch {:3d}, time={:5.1f}s/{:5.1f}s | Train: loss={:.3f}, TAw acc={:5.1f}% |'.format(
                    e + 1, clock1 - clock0, clock2 - clock1, train_loss, 100 * train_acc), end='')
                self.logger.log_scalar(task=t, iter=e + 1, name="loss", value=train_loss, group="train")
                self.logger.log_scalar(task=t, iter=e + 1, name="acc", value=100 * train_acc, group="train")
                for loss_key in train_ppnet_losses.keys():
                    self.logger.log_scalar(task=t, iter=e + 1, name=loss_key, value=train_ppnet_losses[loss_key], group="train")
            else:
                print('| Epoch {:3d}, time={:5.1f}s | Train: skip eval |'.format(e + 1, clock1 - clock0), end='')

            #PUSH
            if (e + 1) % self.model.model.push_at == 0:
                best_loss = np.inf
                clock_p0 = time.time()
                self.push_model(push_loader, t, False)
                clock_p1 = time.time()
                for e_push in range(self.model.model.num_push_tune):
                    self.train_epoch(t, trn_loader, np.inf, at_push=True)
                    if self.eval_on_train:
                        train_loss, train_acc, _, train_ppnet_losses = self.eval(t, trn_loader)
                        clock2 = time.time()
                        print('| Epoch_push {:3d}, time={:5.1f}s/{:5.1f}s | Train: loss={:.3f}, TAw acc={:5.1f}% |'.format(
                            e + 1, clock1 - clock0, clock2 - clock1, train_loss, 100 * train_acc), end='')
                        self.logger.log_scalar(task=t, iter=e * 10 + e_push, name="loss", value=train_loss, group="train_push")
                        self.logger.log_scalar(task=t, iter=e * 10 + e_push, name="acc", value=100 * train_acc, group="train_push")
                        for loss_key in train_ppnet_losses.keys():
                            self.logger.log_scalar(task=t, iter=e * 10 + e_push, name=loss_key,
                                                   value=train_ppnet_losses[loss_key], group="train_push")
                    else:
                        print('| Epoch_push {:3d}, time={:5.1f}s | Train: skip eval |'.format(e + 1, clock1 - clock0), end='')
                    clock3 = time.time()
                    valid_loss, valid_acc, _, eval_ppnet_losses = self.eval(t, val_loader)
                    if e_push == 0:
                        best_loss = np.inf
                    clock4 = time.time()
                    print(' Valid: time={:5.1f}s loss={:.3f}, TAw acc={:5.1f}% |'.format(
                        clock4 - clock3, valid_loss, 100 * valid_acc), end='')
                    self.logger.log_scalar(task=t, iter=e * 10 + e_push, name="loss", value=valid_loss, group="valid_push")
                    self.logger.log_scalar(task=t, iter=e * 10 + e_push, name="acc", value=100 * valid_acc, group="valid_push")
                    for loss_key in eval_ppnet_losses.keys():
                        self.logger.log_scalar(task=t, iter=e * 10 + e_push, name=loss_key,
                                               value=eval_ppnet_losses[loss_key], group="valid_push")
                    if valid_loss < best_loss:
                        # if the loss goes down, keep it as the best model and end line with a star ( * )
                        best_loss = valid_loss
                        best_model = self.model.get_copy()
                        patience = self.lr_patience
                        print(' *', end='')
                    else:
                        # if the loss does not go down, decrease patience
                        patience -= 1
                        if patience <= 0:
                            # if it runs out of patience, reduce the learning rate
                            lr /= self.lr_factor
                            print(' lr={:.1e}'.format(lr), end='')
                            if lr < self.lr_min:
                                # if the lr decreases below minimum, stop the training session
                                print()
                                break
                            # reset patience and recover best model so far to continue training
                            patience = self.lr_patience
                            for i in range(len(self.optimizer.param_groups)):
                                self.optimizer.param_groups[i]['lr'] /= self.lr_factor
                            # self.model.set_state_dict(best_model)

            # Valid
            clock3 = time.time()
            valid_loss, valid_acc, _, valid_ppnet_losses = self.eval(t, val_loader)
            clock4 = time.time()
            print(' Valid: time={:5.1f}s loss={:.3f}, TAw acc={:5.1f}% |'.format(
                clock4 - clock3, valid_loss, 100 * valid_acc), end='')
            self.logger.log_scalar(task=t, iter=e + 1, name="loss", value=valid_loss, group="valid")
            self.logger.log_scalar(task=t, iter=e + 1, name="acc", value=100 * valid_acc, group="valid")
            for loss_key in valid_ppnet_losses.keys():
                self.logger.log_scalar(task=t, iter=e + 1, name=loss_key,
                                       value=valid_ppnet_losses[loss_key], group="valid")

            # Adapt learning rate - patience scheme - early stopping regularization
            if valid_loss < best_loss:
                # if the loss goes down, keep it as the best model and end line with a star ( * )
                best_loss = valid_loss
                best_model = self.model.get_copy()
                patience = self.lr_patience
                print(' *', end='')
            else:
                # if the loss does not go down, decrease patience
                patience -= 1
                if patience <= 0:
                    # if it runs out of patience, reduce the learning rate
                    lr /= self.lr_factor
                    print(' lr={:.1e}'.format(lr), end='')
                    if lr < self.lr_min:
                        # if the lr decreases below minimum, stop the training session
                        print()
                        break
                    # reset patience and recover best model so far to continue training
                    patience = self.lr_patience
                    for i in range(len(self.optimizer.param_groups)):
                        self.optimizer.param_groups[i]['lr'] /= self.lr_factor
                    # self.model.set_state_dict(best_model)
            self.logger.log_scalar(task=t, iter=e + 1, name="patience", value=patience, group="train")
            self.logger.log_scalar(task=t, iter=e + 1, name="lr", value=lr, group="train")
            print()
        self.model.set_state_dict(best_model)

    def post_train_process(self, t, trn_loader):
        """Runs after training all the epochs of the task (after the train session)"""
        pass

    def train_epoch(self, t, trn_loader, e, at_push=False):
        """Runs a single epoch"""
        self.model.train()
        if self.fix_bn and t > 0:
            self.model.freeze_bn()

        if t == 0:
            if e < self.model.model.warm_num:
                for p in self.model.model.features.parameters():
                    p.requires_grad = False
                if self.model.model.share_add_ons:
                    for p in self.model.model.add_on_layers.parameters():
                        p.requires_grad = True
                else:
                    for p in self.model.heads[t].add_on_layers.parameters():
                        p.requires_grad = True
                self.model.heads[t].prototype_vectors.requires_grad = True
                for p in self.model.heads[t].last_layer.parameters():
                    p.requires_grad = False
                if len(self.model.neg_heads) > 0:
                    for p in self.model.neg_heads[t - 1].parameters():
                        p.requires_grad = False
            elif at_push:
                for p in self.model.model.features.parameters():
                    p.requires_grad = False
                if self.model.model.share_add_ons:
                    for p in self.model.model.add_on_layers.parameters():
                        p.requires_grad = False
                else:
                    for p in self.model.heads[t].add_on_layers.parameters():
                        p.requires_grad = False
                self.model.heads[t].prototype_vectors.requires_grad = False
                for p in self.model.heads[t].last_layer.parameters():
                    p.requires_grad = True
                if len(self.model.neg_heads) > 0:
                    for p in self.model.neg_heads[t - 1].parameters():
                        p.requires_grad = True
            elif e >= self.model.model.warm_num:
                for p in self.model.model.features.parameters():
                    p.requires_grad = True
                if self.model.model.share_add_ons:
                    for p in self.model.model.add_on_layers.parameters():
                        p.requires_grad = True
                else:
                    for p in self.model.heads[t].add_on_layers.parameters():
                        p.requires_grad = True
                self.model.heads[t].prototype_vectors.requires_grad = True
                for p in self.model.heads[t].last_layer.parameters():
                    p.requires_grad = False
                if len(self.model.neg_heads) > 0:
                    for p in self.model.neg_heads[t - 1].parameters():
                        p.requires_grad = False

        else:
            for p in self.model.model.features.parameters():
                p.requires_grad = False
            if self.model.model.share_add_ons:
                for p in self.model.model.add_on_layers.parameters():
                    p.requires_grad = True
            else:
                for p in self.model.heads[t].add_on_layers.parameters():
                    p.requires_grad = True
            self.model.heads[t].prototype_vectors.requires_grad = True
            for p in self.model.heads[t].last_layer.parameters():
                p.requires_grad = False
            if len(self.model.neg_heads) > 0:
                for p in self.model.neg_heads[t - 1].parameters():
                    p.requires_grad = False

        for images, targets in trn_loader:
            outputs = self.model(images.to(self.device))
            logits = [outputs[i][1] for i in range(len(outputs))]
            min_distances = [outputs[i][2] for i in range(len(outputs))]
            entropy_loss = self.criterion(t, logits, targets.to(self.device))
            clst_loss_val, sep_loss_val, l1_loss, avg_separation_cost, orth_loss, sub_loss = self.protopnet_looses(
                min_distances,
                targets.to(self.device),
                t,
                all_out=self.exemplars_dataset is not None,
            )
            loss = entropy_loss + clst_loss_val * 0.8 + sep_loss_val * self.model.model.sep_weight + 1e-4 * l1_loss
            # Backward
            if e < self.model.model.warm_num or t > 0:
                self.warm_optim.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clipgrad)
                self.warm_optim.step()
                self.optimizer = self.warm_optim
            elif at_push:
                self.push_optim.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clipgrad)
                self.push_optim.step()
                self.optimizer = self.push_optim
            elif e >= self.model.model.warm_num:
                self.joint_optim.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clipgrad)
                self.joint_optim.step()
                self.optimizer = self.joint_optim

    def eval(self, t, val_loader):
        """Contains the evaluation code"""
        with torch.no_grad():
            total_loss, total_acc_taw, total_acc_tag, total_num, total_clst, total_sep, total_l1, total_avg_sep, total_entropy = \
                0, 0, 0, 0, 0, 0, 0, 0, 0
            self.model.eval()
            for images, targets in val_loader:
                # Forward current model
                outputs = self.model(images.to(self.device))
                logits = [outputs[i][1] for i in range(len(outputs))]
                min_distances = [outputs[i][2] for i in range(len(outputs))]
                entropy_loss = self.criterion(t, logits, targets.to(self.device))
                clst_loss_val, sep_loss_val, l1_loss, avg_sep_cost, orth_loss, sub_loss = self.protopnet_looses(
                    min_distances,
                    targets.to(self.device),
                    t,
                    all_out=self.exemplars_dataset is not None,
                )
                loss = entropy_loss + clst_loss_val * 0.8 + sep_loss_val * self.model.model.sep_weight + 1e-4 * l1_loss
                hits_taw, hits_tag = self.calculate_metrics(logits, targets)
                # Log
                total_loss += loss.item() * len(targets)
                total_entropy += entropy_loss.item() * len(targets)
                total_clst += clst_loss_val.item()
                total_sep += sep_loss_val.item()
                total_avg_sep += avg_sep_cost.item()
                total_l1 += l1_loss.item()
                total_acc_taw += hits_taw.sum().item()
                total_acc_tag += hits_tag.sum().item()
                total_num += len(targets)
        ppnet_losses = {
            'clst': total_clst,
            'sep': total_sep,
            'avg_sep': total_avg_sep,
            'l1': total_l1,
            'entropy': total_entropy / total_num,
        }
        return total_entropy / total_num, total_acc_taw / total_num, total_acc_tag / total_num, ppnet_losses

    def ppnet_eval(self, t, val_loader):
        """Contains the evaluation code"""
        with torch.no_grad():
            proto_acts = []
            proto_acts_per_class = {}
            proto_acts_out_class = {}

            self.model.eval()
            for images, targets in val_loader:
                # Forward current model
                outputs = self.model(images.to(self.device))
                min_distances = [outputs[i][2] for i in range(t + 1)]
                all_min_distances = torch.cat(min_distances, dim=1)
                min_similarities = self.model.heads[t].distance_2_similarity(all_min_distances)
                proto_act = min_similarities
                for target in targets:
                    class_act = min_similarities[:, target: target + 10]
                    out_class_act = torch.cat([min_similarities[:, :target], min_similarities[:, target + 10:]], dim=1).mean(0)
                    if target in proto_acts_out_class.keys():
                        proto_acts_out_class[target] += out_class_act.detach().cpu().mean()
                        proto_acts_per_class[target] += class_act.detach().cpu().mean()
                    else:
                        proto_acts_out_class[target] = out_class_act.detach().cpu().mean()
                        proto_acts_per_class[target] = class_act.detach().cpu().mean()

                proto_acts.append(proto_act)
        proto_acts = torch.cat(proto_acts, dim=0)

        from matplotlib import pyplot as plt
        f_proto = plt.figure(dpi=300)
        ax = f_proto.subplots(nrows=1, ncols=1)
        vals = proto_acts.mean(0).squeeze().cpu().numpy().flatten()
        ax.bar(np.arange(len(vals)), vals, label="Task {}".format(t))
        ax.set_xlabel("Proto idx", fontsize=6, fontfamily='serif')
        ax.set_ylabel("Mean proto act on test set", fontsize=6, fontfamily='serif')
        ax.set_xlim(0, self.model.model.prototype_shape[0])

        f_proto_per_class = plt.figure(dpi=300)
        ax = f_proto_per_class.subplots(nrows=1, ncols=1)
        ax.bar(np.asarray(list(proto_acts_per_class.keys())), np.asarray(list(proto_acts_per_class.values())), label="Task {}".format(t))
        ax.set_xlabel("class idx", fontsize=6, fontfamily='serif')
        ax.set_ylabel("Mean of in-class proto activations", fontsize=6, fontfamily='serif')

        f_proto_out_class = plt.figure(dpi=300)
        ax = f_proto_out_class.subplots(nrows=1, ncols=1)
        ax.bar(np.asarray(list(proto_acts_per_class.keys())), np.asarray(list(proto_acts_per_class.values())),
               label="Task {}".format(t))
        ax.set_xlabel("class idx", fontsize=6, fontfamily='serif')
        ax.set_ylabel("Mean of out-class proto activations", fontsize=6, fontfamily='serif')
        return f_proto, f_proto_per_class, f_proto_out_class

    def protopnet_looses(self, min_distances, label, t, use_l1_mask=True, all_out=False):
        if all_out and t > 0:
            min_distances = torch.cat(min_distances[:t + 1], dim=1)
        else:
            min_distances = min_distances[t]
            label = label - self.model.task_offset[t]
        max_dist = (self.model.model.prototype_shape[1]
                    * self.model.model.prototype_shape[2]
                    * self.model.model.prototype_shape[3])
        if all_out and t > 0:
            prototypes_of_correct_class_m = torch.zeros(((t + 1) * self.model.model.num_prototypes,
                                                         (t + 1) * self.model.model.num_classes,))
            for i in range(t + 1):
                prototypes_of_correct_class_m[
                i * self.model.model.num_prototypes:(i + 1) * self.model.model.num_prototypes,
                i * self.model.model.num_classes:(i + 1) * self.model.model.num_classes] = \
                    self.model.heads[t].prototype_class_identity
            prototypes_of_correct_class_m = prototypes_of_correct_class_m.cuda()
            prototypes_of_correct_class = torch.t(prototypes_of_correct_class_m[:, label].cuda())
        else:
            prototypes_of_correct_class = torch.t(self.model.heads[t].prototype_class_identity[:, label]).cuda()
        inverted_distances, _ = torch.max((max_dist - min_distances) * prototypes_of_correct_class, dim=1)

        if isinstance(self.model.model, PPNet):
            # prototypes_of_correct_class is a tensor of shape batch_size * num_prototypes
            # calculate cluster cost
            cluster_cost = torch.mean(max_dist - inverted_distances)

            # calculate separation cost
            prototypes_of_wrong_class = 1 - prototypes_of_correct_class
            inverted_distances_to_nontarget_prototypes, _ = \
                torch.max((max_dist - min_distances) * prototypes_of_wrong_class, dim=1)
            separation_cost = torch.mean(max_dist - inverted_distances_to_nontarget_prototypes)

            # calculate avg cluster cost
            avg_separation_cost = \
                torch.sum(min_distances * prototypes_of_wrong_class, dim=1) / torch.sum(prototypes_of_wrong_class, dim=1)
            avg_separation_cost = torch.mean(avg_separation_cost)

            if use_l1_mask:
                if all_out and t > 0:
                    l1_mask = 1 - prototypes_of_correct_class_m.cuda()
                    ll_m = torch.zeros(((t + 1) * self.model.model.num_prototypes,
                                        (t + 1) * self.model.model.num_classes,)).cuda()
                    for i in range(t):
                        ll_m[i * self.model.model.num_prototypes:(i+1) * self.model.model.num_prototypes,
                             i * self.model.model.num_classes:(i+1) * self.model.model.num_classes] = \
                            torch.t(self.model.heads[i].last_layer.weight)
                    l1 = (ll_m * l1_mask).norm(p=1)
                else:
                    l1_mask = 1 - torch.t(self.model.heads[t].prototype_class_identity).cuda()
                    l1 = (self.model.heads[t].last_layer.weight * l1_mask).norm(p=1)
            else:
                l1 = self.model.heads[t].last_layer.weight.norm(p=1)
            if self.model.model.incorrect_weight_btw_tasks and len(self.model.heads) > 1:
                l1 += self.model.neg_heads[t-1].weight.norm(p=1)
            return cluster_cost, separation_cost, l1, avg_separation_cost, torch.zeros(1).cuda(), torch.zeros(1).cuda()
        else:
            subspace_max_dist = (self.model.model.prototype_shape[0] * self.model.model.prototype_shape[2] *
                                 self.model.model.prototype_shape[3])  # 2000
            # prototypes_of_correct_class is a tensor of shape batch_size * num_prototypes
            # calculate cluster cost
            cluster_cost = torch.mean(max_dist - inverted_distances)

            # calculate separation cost
            prototypes_of_wrong_class = 1 - prototypes_of_correct_class
            inverted_distances_to_nontarget_prototypes, _ = \
                torch.max((max_dist - min_distances) * prototypes_of_wrong_class, dim=1)
            separation_cost = torch.mean(max_dist - inverted_distances_to_nontarget_prototypes)

            # calculate avg cluster cost
            avg_separation_cost = \
                torch.sum(min_distances * prototypes_of_wrong_class, dim=1) / torch.sum(prototypes_of_wrong_class,
                                                                                        dim=1)
            avg_separation_cost = torch.mean(avg_separation_cost)

            # optimize orthogonality of prototype_vector
            basis = []
            for z in range(t+1):
                basis.append(self.model.heads[z].prototype_vectors)

            cur_basis_matrix = torch.cat(basis, dim=0)  # [2000,128]
            subspace_basis_matrix = cur_basis_matrix.reshape(self.model.model.num_classes * (t+1),
                                                             self.model.heads[-1].num_prototypes_per_class,
                                                             self.model.model.prototype_shape[1])  # [200,10,128]
            subspace_basis_matrix_T = torch.transpose(subspace_basis_matrix, 1, 2)  # [200,10,128]->[200,128,10]
            orth_operator = torch.matmul(subspace_basis_matrix,
                                         subspace_basis_matrix_T)  # [200,10,128] [200,128,10] -> [200,10,10]
            I_operator = torch.eye(subspace_basis_matrix.size(1), subspace_basis_matrix.size(1)).cuda()  # [10,10]
            difference_value = orth_operator - I_operator  # [200,10,10]-[10,10]->[200,10,10]
            orth_cost = torch.sum(torch.relu(torch.norm(difference_value, p=1, dim=[1, 2]) - 0))  # [200]->[1]

            del cur_basis_matrix
            del orth_operator
            del I_operator
            del difference_value

            # subspace sep
            projection_operator = torch.matmul(subspace_basis_matrix_T,
                                               subspace_basis_matrix)  # [200,128,10] [200,10,128] -> [200,128,128]
            del subspace_basis_matrix
            del subspace_basis_matrix_T

            projection_operator_1 = torch.unsqueeze(projection_operator, dim=1)  # [200,1,128,128]
            projection_operator_2 = torch.unsqueeze(projection_operator, dim=0)  # [1,200,128,128]
            pairwise_distance = torch.norm(projection_operator_1 - projection_operator_2 + 1e-10, p='fro',
                                           dim=[2, 3])  # [200,200,128,128]->[200,200]
            subspace_sep = 0.5 * torch.norm(pairwise_distance, p=1, dim=[0, 1], dtype=torch.double) / torch.sqrt(
                torch.tensor(2, dtype=torch.double)).cuda()
            del projection_operator_1
            del projection_operator_2
            del pairwise_distance

            if use_l1_mask:
                l1_mask = 1 - torch.t(self.model.heads[t].prototype_class_identity).cuda()
                l1 = (self.model.heads[-1].last_layer.weight * l1_mask).norm(p=1)
                # weight 200,2000   prototype_class_identity [2000,200]
            else:
                l1 = self.model.heads[-1].last_layer.weight.norm(p=1)
            self.model.heads[-1].prototype_vectors.data = F.normalize(self.model.heads[-1].prototype_vectors, p=2, dim=1).data
            return cluster_cost, separation_cost, l1, avg_separation_cost, orth_cost, subspace_sep

    def calculate_metrics(self, outputs, targets):
        """Contains the main Task-Aware and Task-Agnostic metrics"""
        pred = torch.zeros_like(targets.to(self.device))
        # Task-Aware Multi-Head
        for m in range(len(pred)):
            this_task = (self.model.task_cls.cumsum(0) <= targets[m]).sum()
            pred[m] = outputs[this_task][m].argmax() + self.model.task_offset[this_task]
        hits_taw = (pred == targets.to(self.device)).float()
        # Task-Agnostic Multi-Head
        if self.multi_softmax:
            outputs = [torch.nn.functional.log_softmax(output, dim=1) for output in outputs]
            pred = torch.cat(outputs, dim=1).argmax(1)
        else:
            pred = torch.cat(outputs, dim=1).argmax(1)
        hits_tag = (pred == targets.to(self.device)).float()
        return hits_taw, hits_tag

    def criterion(self, t, outputs, targets):
        """Returns the loss value"""
        return torch.nn.functional.cross_entropy(outputs[t], targets - self.model.task_offset[t])

    def push_model(self, train_push_loader, t, all_out=False):
        if isinstance(self.model.model, PPNet):
            push = ppnet_push
        else:
            push = tesnet_push
        if all_out:
            identities = [self.model.heads[i].prototype_class_identity for i in range(len(self.model.heads))]
            prototype_class_identity = torch.cat(identities, dim=1)
        else:
            prototype_class_identity = self.model.heads[t].prototype_class_identity
        global_min_proto_dist = np.full(self.model.model.num_prototypes, np.inf)
        global_min_fmap_patches = np.zeros(
            [self.model.model.num_prototypes,
             self.model.model.prototype_shape[1],
             self.model.model.prototype_shape[2],
             self.model.model.prototype_shape[3]])

        proto_rf_boxes = np.full(shape=[self.model.model.num_prototypes, 6],
                                 fill_value=-1)
        proto_bound_boxes = np.full(shape=[self.model.model.num_prototypes, 6],
                                    fill_value=-1)

        search_batch_size = train_push_loader.batch_size

        t_in = t
        if t > 0 and self.model.model.repeat_task_0:
            t_in += -1

        for push_iter, (search_batch_input, search_y) in enumerate(train_push_loader):
            '''
            start_index_of_search keeps track of the index of the image
            assigned to serve as prototype
            '''
            if not all_out:
                search_y = search_y - self.model.task_offset[t_in]
            start_index_of_search_batch = push_iter * search_batch_size

            push(global_min_fmap_patches, global_min_proto_dist, self.model, proto_bound_boxes, proto_rf_boxes,
                 search_batch_input, search_y, start_index_of_search_batch, prototype_class_identity, task=t,
                 log_path=self.logger.log_path + '/' + self.logger.exp_name)

        prototype_update = np.reshape(global_min_fmap_patches,
                                      tuple(self.model.model.prototype_shape))
        self.model.heads[t].prototype_vectors.data.copy_(torch.tensor(prototype_update, dtype=torch.float32).cuda())

