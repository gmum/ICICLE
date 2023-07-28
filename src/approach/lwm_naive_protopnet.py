import os
import torch
from copy import deepcopy
import torch.nn.functional as F
from argparse import ArgumentParser
from torchvision.utils import save_image

from networks.network import LLL_Net
from .incremental_learning import Inc_Learning_Appr_PPNet
from datasets.exemplars_dataset import ExemplarsDataset


class Appr(Inc_Learning_Appr_PPNet):
    """Class implementing the Learning without Memorizing (LwM) approach
    described in http://arxiv.org/abs/1811.08051
    """

    def __init__(self, model, device, nepochs=100, lr=0.05, lr_min=1e-4, lr_factor=3, lr_patience=5, clipgrad=10000,
                 momentum=0, wd=0, multi_softmax=False, wu_nepochs=0, wu_lr_factor=1, fix_bn=False, eval_on_train=False,
                 logger=None, exemplars_dataset=None, beta=1.0, gamma=1.0, gradcam_layer='layer3',
                 log_gradcam_samples=0):
        super(Appr, self).__init__(model, device, nepochs, lr, lr_min, lr_factor, lr_patience, clipgrad, momentum, wd,
                                   multi_softmax, wu_nepochs, wu_lr_factor, fix_bn, eval_on_train, logger,
                                   exemplars_dataset)
        self.beta = beta
        self.gamma = gamma
        self.gradcam_layer = gradcam_layer
        self.log_gradcam_samples = log_gradcam_samples
        self.model_old = None
        self._samples_to_log_X = []
        self._samples_to_log_y = []

    @staticmethod
    def exemplars_dataset_class():
        return ExemplarsDataset

    @staticmethod
    def extra_parser(args):
        parser = ArgumentParser()
        # Paper, sec 4: beta and gamma introduce in Eq.3, but no values are given in the paper
        parser.add_argument('--beta', default=1, type=float, required=False,
                            help='Trade-off for distillation loss (default=%(default)s)')
        parser.add_argument('--gamma', default=1, type=float, required=False,
                            help='Trade-off for attention loss(default=%(default)s)')
        return parser.parse_known_args(args)

    def _take_samples_for_current_task(self, trn_loader):
        current_task_X, current_task_y = [], []
        for X, y in trn_loader:
            n = self.log_gradcam_samples - sum(_x.size(0) for _x in current_task_X)
            if n > 0:
                current_task_X.append(X[:n])
                current_task_y.append(y[:n])
        self._samples_to_log_X.append(torch.cat(current_task_X))
        self._samples_to_log_y.append(torch.cat(current_task_y))

    def _save_gradcam_examples(self, t, trn_loader):
        self._take_samples_for_current_task(trn_loader)
        output_orig_filename = os.path.join(self.logger.exp_path, '{t}_orig_post_train.png'.format(t=t))
        output_map_filename = os.path.join(self.logger.exp_path, '{t}_gcam_post_train.png'.format(t=t))
        print('Task {} - Saving {} samples to: {}'.format(t, self.log_gradcam_samples, output_orig_filename))
        save_image(torch.cat(self._samples_to_log_X), output_orig_filename, normalize=True, nrow=(t + 1))
        print('Task {} - Saving {} samples with heatmaps to: {}'.format(t, self.log_gradcam_samples, output_map_filename))
        X_with_gradcam = []
        with GradCAM(self.model_old, self.gradcam_layer) as gradcam:
            for X in self._samples_to_log_X:
                img_with_heatmaps = []
                for x in X:
                    heatmap = gradcam(x.to(self.device))
                    img = gradcam.visualize_cam(heatmap, x)
                    img = img.view([1] + list(img.size()))
                    img_with_heatmaps.append(img)
                X_with_gradcam.append(torch.cat(img_with_heatmaps))
        save_image(torch.cat(X_with_gradcam), output_map_filename, nrow=(t + 1))

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
        push_params = [{'params': self.model.heads[t].last_layer.parameters(), 'lr': self.lr / 10,
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
        self.model_old = deepcopy(self.model)
        if self.log_gradcam_samples > 0:
            self._save_gradcam_examples(t, trn_loader)

    def train_epoch(self, t, trn_loader, e, at_push=False):
        """Runs a single epoch"""
        if self.model_old is None:  # First task only
            return super().train_epoch(t, trn_loader, e)
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

        # Do training with distillation losses
        for images, targets in trn_loader:
            images = images.to(self.device)
            # Forward old model
            outputs_old = self.model_old(images.to(self.device))
            self.model.zero_grad()
            self.model.train()
            outputs = self.model(images.to(self.device))
            logits = [outputs[i][1] for i in range(len(outputs))]
            logits_old = [outputs_old[i][1] for i in range(len(outputs_old))]
            min_distances = [outputs[i][2] for i in range(len(outputs))]
            outputs_old = outputs_old[:t]
            outputs = outputs[:t]
            distances_old = torch.cat([outputs_old[i][0] for i in range(len(outputs_old))], dim=1)
            distances = torch.cat([outputs[i][0] for i in range(len(outputs))], dim=1)
            clst_loss_val, sep_loss_val, l1_loss, avg_separation_cost = self.protopnet_looses(
                min_distances,
                targets.to(self.device),
                t,
                all_out=self.exemplars_dataset is not None,
            )
            entropy_loss = self.criterion(t, logits, targets.to(self.device), logits_old, distances, distances_old)
            loss = entropy_loss + clst_loss_val * 0.8 + sep_loss_val * self.model.model.sep_weight + 1e-4 * l1_loss

            # Backward
            if e < self.model.model.warm_num:
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
        if t == 0:
            return super().eval(t, val_loader)
        total_loss, total_acc_taw, total_acc_tag, total_num, total_clst, total_sep, total_l1, total_avg_sep, total_entropy = \
            0, 0, 0, 0, 0, 0, 0, 0, 0
        self.model.eval()
        for images, targets in val_loader:
            images = images.to(self.device)
            # Forward old model
            outputs_old = self.model_old(images)
            # Forward current model
            outputs = self.model(images)
            logits = [outputs[i][1] for i in range(len(outputs))]
            logits_old = [outputs_old[i][1] for i in range(len(outputs_old))]
            min_distances = [outputs[i][2] for i in range(len(outputs))]
            outputs_old = outputs_old[:t]
            outputs = outputs[:t]
            distances_old = torch.cat([outputs_old[i][0] for i in range(len(outputs_old))], dim=1)
            distances = torch.cat([outputs[i][0] for i in range(len(outputs))], dim=1)
            entropy_loss = self.criterion(t, logits, targets.to(self.device), logits_old, distances, distances_old)
            # Log
            clst_loss_val, sep_loss_val, l1_loss, avg_sep_cost = self.protopnet_looses(
                min_distances,
                targets.to(self.device),
                t,
                all_out=self.exemplars_dataset is not None,
            )
            loss = entropy_loss + clst_loss_val * 0.8 + sep_loss_val * self.model.model.sep_weight + 1e-4 * l1_loss
            hits_taw, hits_tag = self.calculate_metrics(logits, targets)
            # Log
            total_loss += loss.item() * len(targets)
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
        return total_loss / total_num, total_acc_taw / total_num, total_acc_tag / total_num, ppnet_losses

    def attention_distillation_loss(self, attention_map1, attention_map2):
        """Calculates the attention distillation loss"""
        return torch.norm(attention_map2 - attention_map1, p=1, dim=1).sum(dim=1).mean()

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

    def criterion(self, t, outputs, targets, outputs_old=None, attmap=None, attmap_old=None):
        """Returns the loss value"""
        loss = 0
        if t > 0 and outputs_old is not None:
            # Knowledge distillation loss for all previous tasks
            loss += self.beta * self.cross_entropy(torch.cat(outputs[:t], dim=1),
                                                   torch.cat(outputs_old[:t], dim=1), exp=1.0 / 2.0)
        # Attention Distillation loss
        if attmap is not None and attmap_old is not None:
            loss += self.gamma * self.attention_distillation_loss(attmap, attmap_old)
        # Current cross-entropy loss -- with exemplars use all heads
        if len(self.exemplars_dataset) > 0:
            return loss + torch.nn.functional.cross_entropy(torch.cat(outputs, dim=1), targets)
        return loss + torch.nn.functional.cross_entropy(outputs[t], targets - self.model.task_offset[t])

