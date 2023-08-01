import torch
from copy import deepcopy
from argparse import ArgumentParser

from .incremental_learning import Inc_Learning_Appr_PPNet
from datasets.exemplars_dataset import ExemplarsDataset


class Appr(Inc_Learning_Appr_PPNet):
    """Class implementing the Learning Without Forgetting (LwF) approach
    described in https://arxiv.org/abs/1606.09282
    """

    # Weight decay of 0.0005 is used in the original article (page 4).
    # Page 4: "The warm-up step greatly enhances fine-tuning’s old-task performance, but is not so crucial to either our
    #  method or the compared Less Forgetting Learning (see Table 2(b))."
    def __init__(self, model, device, nepochs=100, lr=0.05, lr_min=1e-4, lr_factor=3, lr_patience=5, clipgrad=10000,
                 momentum=0, wd=0, multi_softmax=False, wu_nepochs=0, wu_lr_factor=1, fix_bn=False, eval_on_train=False,
                 logger=None, exemplars_dataset=None, lamb=1, T=2):
        super(Appr, self).__init__(model, device, nepochs, lr, lr_min, lr_factor, lr_patience, clipgrad, momentum, wd,
                                   multi_softmax, wu_nepochs, wu_lr_factor, fix_bn, eval_on_train, logger,
                                   exemplars_dataset)
        self.model_old = None
        self.lamb = lamb
        self.T = T

    @staticmethod
    def exemplars_dataset_class():
        return ExemplarsDataset

    @staticmethod
    def extra_parser(args):
        """Returns a parser containing the approach specific parameters"""
        parser = ArgumentParser()
        # Page 5: "lambda is a loss balance weight, set to 1 for most our experiments. Making lambda larger will favor
        # the old task performance over the new task’s, so we can obtain a old-task-new-task performance line by
        # changing lambda."
        parser.add_argument('--lamb', default=1, type=float, required=False,
                            help='Forgetting-intransigence trade-off (default=%(default)s)')
        # Page 5: "We use T=2 according to a grid search on a held out set, which aligns with the authors’
        #  recommendations." -- Using a higher value for T produces a softer probability distribution over classes.
        parser.add_argument('--T', default=2, type=int, required=False,
                            help='Temperature scaling (default=%(default)s)')
        return parser.parse_known_args(args)

    def _get_optimizer(self):
        """Returns the optimizer"""
        if len(self.exemplars_dataset) == 0 and len(self.model.heads) > 1:
            # if there are no exemplars, previous heads are not modified
            params = list(self.model.model.parameters()) + list(self.model.heads[-1].parameters())
        else:
            params = self.model.parameters()
        return torch.optim.SGD(params, lr=self.lr, weight_decay=self.wd, momentum=self.momentum)

    def _get_optimizers(self, t):
        """Returns the optimizer"""
        warm_params = [
            {'params': self.model.model.add_on_layers.parameters(), 'lr': 3 * self.lr, 'weight_decay': self.wd},
            {'params': self.model.heads[t].prototype_vectors, 'lr': 3 * self.lr},
        ]
        joint_params = [
            {'params': self.model.model.features.parameters(), 'lr': self.lr / 10, 'weight_decay': self.wd},
            {'params': self.model.model.add_on_layers.parameters(), 'lr': 3 * self.lr,
             'weight_decay': self.wd},
            {'params': self.model.heads[t].prototype_vectors, 'lr': 3 * self.lr},
        ]
        push_params = [{'params': self.model.heads[t].last_layer.parameters(), 'lr': self.lr,
                        'weight_decay': self.wd},
                       ]
        if len(self.model.neg_heads) > 0:
            push_params.append({'params': self.model.neg_heads[t - 1].parameters(), 'lr': self.lr / 10,
                                'weight_decay': self.wd})
        warm_optimizer = torch.optim.Adam(warm_params)
        joint_optimizer = torch.optim.Adam(joint_params)
        push_optimizer = torch.optim.Adam(push_params)
        return joint_optimizer, push_optimizer, warm_optimizer

    def train_loop(self, t, trn_loader, val_loader, push_loader=None):
        """Contains the epochs loop"""

        # add exemplars to train_loader
        if len(self.exemplars_dataset) > 0 and t > 0:
            trn_loader = torch.utils.data.DataLoader(trn_loader.dataset + self.exemplars_dataset,
                                                     batch_size=trn_loader.batch_size,
                                                     shuffle=True,
                                                     num_workers=trn_loader.num_workers,
                                                     pin_memory=trn_loader.pin_memory)
            # FINETUNING TRAINING -- contains the epochs loop
            super().train_loop(t, trn_loader, val_loader, push_loader)

            # EXEMPLAR MANAGEMENT -- select training subset
            self.exemplars_dataset.collect_exemplars(self.model, trn_loader, val_loader.dataset.transform)

    def post_train_process(self, t, trn_loader):
        """Runs after training all the epochs of the task (after the train session)"""

        # Restore best and save model for future tasks
        self.model_old = deepcopy(self.model)
        self.model_old.eval()
        self.model_old.freeze_all()

    def train_epoch(self, t, trn_loader, e, at_push=False):
        """Runs a single epoch"""
        self.model.train()
        if self.fix_bn and t > 0:
            self.model.freeze_bn()

        if e < self.model.model.warm_num:
            for p in self.model.model.features.parameters():
                p.requires_grad = False
            for p in self.model.model.add_on_layers.parameters():
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
            for p in self.model.model.add_on_layers.parameters():
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
            for p in self.model.model.add_on_layers.parameters():
                p.requires_grad = True
            self.model.heads[t].prototype_vectors.requires_grad = True
            for p in self.model.heads[t].last_layer.parameters():
                p.requires_grad = False
            if len(self.model.neg_heads) > 0:
                for p in self.model.neg_heads[t - 1].parameters():
                    p.requires_grad = False

        for images, targets in trn_loader:
            # Forward old model
            targets_old = None
            if t > 0:
                with torch.no_grad():
                    outputs = self.model_old(images.to(self.device))
                    targets_old = [outputs[i][1] for i in range(len(outputs))]
            # Forward current model
            outputs = self.model(images.to(self.device))
            logits = [outputs[i][1] for i in range(len(outputs))]
            min_distances = [outputs[i][2] for i in range(len(outputs))]
            entropy_loss = self.criterion(t, logits, targets.to(self.device), targets_old)
            clst_loss_val, sep_loss_val, l1_loss, avg_separation_cost, orth_loss, sub_loss = self.protopnet_looses(
                min_distances,
                targets.to(self.device),
                t,
                all_out=self.exemplars_dataset is not None,
            )
            loss = entropy_loss + clst_loss_val * 0.8 + sep_loss_val * self.model.model.sep_weight + 1e-4 * l1_loss + \
                   1e-4 * orth_loss - 1e-7 * sub_loss
            # Backward
            if e == 0:
                self.optimizer = self.joint_optim
            if e < self.model.model.warm_num:
                self.warm_optim.zero_grad()
                loss.backward()
                # torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clipgrad)
                self.warm_optim.step()
            elif at_push:
                self.push_optim.zero_grad()
                loss.backward()
                # torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clipgrad)
                self.push_optim.step()
            elif e >= self.model.model.warm_num:
                self.joint_optim.zero_grad()
                loss.backward()
                # torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clipgrad)
                self.joint_optim.step()

    def eval(self, t, val_loader):
        """Contains the evaluation code"""
        with torch.no_grad():
            total_loss, total_acc_taw, total_acc_tag, total_num, total_clst, total_sep, total_l1, total_avg_sep, total_entropy = \
                0, 0, 0, 0, 0, 0, 0, 0, 0
            self.model.eval()
            for images, targets in val_loader:
                # Forward old model
                targets_old = None
                if t > 0:
                    outputs = self.model_old(images.to(self.device))
                    targets_old = [outputs[i][1] for i in range(len(outputs))]
                # Forward current model
                outputs = self.model(images.to(self.device))
                logits = [outputs[i][1] for i in range(len(outputs))]
                min_distances = [outputs[i][2] for i in range(len(outputs))]
                entropy_loss = self.criterion(t, logits, targets.to(self.device), targets_old)
                # Log
                clst_loss_val, sep_loss_val, l1_loss, avg_sep_cost, orth_loss, sub_loss = self.protopnet_looses(
                    min_distances,
                    targets.to(self.device),
                    t,
                    all_out=self.exemplars_dataset is not None,
                )
                loss = entropy_loss + clst_loss_val * 0.8 + sep_loss_val * self.model.model.sep_weight + 1e-4 * l1_loss + \
                       1e-4 * orth_loss - 1e-7 * sub_loss
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

    def cross_entropy(self, outputs, targets, exp=1.0, size_average=True, eps=1e-5):
        """Calculates cross-entropy with temperature scaling"""
        out = torch.nn.functional.softmax(outputs, dim=1)
        tar = torch.nn.functional.softmax(targets, dim=1)
        if exp != 1:
            out = out.pow(exp)
            out = out / out.sum(1).view(-1, 1).expand_as(out)
            tar = tar.pow(exp)
            tar = tar / tar.sum(1).view(-1, 1).expand_as(tar)
        out = out + eps / out.size(1)
        out = out / out.sum(1).view(-1, 1).expand_as(out)
        ce = -(tar * out.log()).sum(1)
        if size_average:
            ce = ce.mean()
        return ce

    def criterion(self, t, outputs, targets, outputs_old=None):
        """Returns the loss value"""
        loss = 0
        if t > 0:
            # Knowledge distillation loss for all previous tasks
            loss += self.lamb * self.cross_entropy(torch.cat(outputs[:t], dim=1),
                                                   torch.cat(outputs_old[:t], dim=1), exp=1.0 / self.T)
        # Current cross-entropy loss -- with exemplars use all heads
        if len(self.exemplars_dataset) > 0:
            return loss + torch.nn.functional.cross_entropy(torch.cat(outputs, dim=1), targets)
        return loss + torch.nn.functional.cross_entropy(outputs[t], targets - self.model.task_offset[t])
