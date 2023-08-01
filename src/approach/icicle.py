import torch
from copy import deepcopy
from argparse import ArgumentParser
from sklearn.cluster import KMeans

from .incremental_learning import Inc_Learning_Appr_PPNet
from networks.tesnet import TesNet
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
                 logger=None, exemplars_dataset=None, lamb=1, T=2, perc=5, similarity_reg=False, normalize_sim=False,
                 lr_old=None, permute_settlement=False):
        super(Appr, self).__init__(model, device, nepochs, lr, lr_min, lr_factor, lr_patience, clipgrad, momentum, wd,
                                   multi_softmax, wu_nepochs, wu_lr_factor, fix_bn, eval_on_train, logger,
                                   exemplars_dataset)
        self.model_old = None
        self.lamb = lamb
        self.T = T
        self.perc = perc
        self.similarity_reg = similarity_reg
        self.normalize_sim = normalize_sim
        self.settlers = None
        self.permute_settlement = permute_settlement
        if lr_old:
            self.lr_old = lr_old
        else:
            self.lr_old = 3 * self.lr

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
        parser.add_argument('--lamb', default=0.25, type=float, required=False,
                            help='Forgetting-intransigence trade-off (default=%(default)s)')
        # Page 5: "We use T=2 according to a grid search on a held out set, which aligns with the authors’
        #  recommendations." -- Using a higher value for T produces a softer probability distribution over classes.
        parser.add_argument('--T', default=2, type=int, required=False,
                            help='Temperature scaling (default=%(default)s)')
        parser.add_argument('--perc', default=97, type=int, required=False,
                            help='Percentile to mask the acts')
        parser.add_argument('--similarity_reg', action='store_true',
                            help='Whether to use similarity or distances to regularize')
        parser.add_argument('--normalize_sim', action='store_true',
                            help='Whether to normalize the similarities')
        parser.add_argument('--lr_old', default=0.25, type=float, required=False,
                            help='Learning rate')
        parser.add_argument('--permute_settlement', action='store_true',
                            help='Whether to permute protots to settlement')
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
        if not self.model.model.share_add_ons:
            warm_params = [
                {'params': self.model.heads[t].add_on_layers.parameters(), 'lr': 3 * self.lr, 'weight_decay': self.wd},
                {'params': self.model.heads[t].prototype_vectors, 'lr': 3 * self.lr},
            ]
            joint_params = [
                {'params': self.model.model.features.parameters(), 'lr': self.lr / 10, 'weight_decay': self.wd},
                {'params': self.model.heads[t].add_on_layers.parameters(), 'lr': 3 * self.lr,
                 'weight_decay': self.wd},
                {'params': self.model.heads[t].prototype_vectors, 'lr': 3 * self.lr},
            ]
            push_params = [{'params': self.model.heads[t].last_layer.parameters(), 'lr': self.lr,
                            'weight_decay': self.wd},
                           ]
        else:
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
        if t > 0:
            joint_params.extend([
                {'params': self.model.heads[i].prototype_vectors, 'lr': self.lr_old} for i in range(t)
            ])
            warm_params.extend([
                {'params': self.model.heads[i].prototype_vectors, 'lr': self.lr_old} for i in range(t)
            ])
            if not self.model.model.share_add_ons:
                joint_params.extend([
                    {'params': self.model.heads[i].add_on_layers.parameters(), 'lr': self.lr_old} for i in range(t)
                ])
                warm_params.extend([
                    {'params': self.model.heads[i].add_on_layers.parameters(), 'lr': self.lr_old} for i in range(t)
                ])


        warm_optimizer = torch.optim.Adam(warm_params)
        joint_optimizer = torch.optim.Adam(joint_params)
        proto_optimizer = torch.optim.Adam(push_params)
        return joint_optimizer, proto_optimizer, warm_optimizer

    def train_loop(self, t, trn_loader, val_loader, push_loader=None):
        """Contains the epochs loop"""

        # add exemplars to train_loader
        if len(self.exemplars_dataset) > 0 and t > 0:
            trn_loader = torch.utils.data.DataLoader(trn_loader.dataset + self.exemplars_dataset,
                                                     batch_size=trn_loader.batch_size,
                                                     shuffle=True,
                                                     num_workers=trn_loader.num_workers,
                                                     pin_memory=trn_loader.pin_memory)

        if t > 0:
            self.settlement(push_loader, task=t)
        # FINETUNING TRAINING -- contains the epochs loop
        super().train_loop(t, trn_loader, val_loader, push_loader)

        # EXEMPLAR MANAGEMENT -- select training subset
        self.exemplars_dataset.collect_exemplars(self.model, trn_loader, val_loader.dataset.transform)

    def settlement(self, loader, task=0):
        with torch.no_grad():
            if task > 0:
                vals = []
                labels = []
                for it, (input_data, label) in enumerate(loader):
                    outputs = self.model(input_data.cuda())
                    distances = torch.cat([outputs[i][0] for i in range(len(outputs))], dim=1)
                    vals.append(distances.reshape(distances.shape[0], distances.shape[1], -1).cpu())
                    labels.append(label)
                vals = torch.cat(vals, dim=0)

                if isinstance(self.model.model, TesNet):
                    class_vals_flt = torch.unique(vals.flatten())[:50000]
                else:
                    class_vals_flt = torch.unique(vals.flatten())[::2**(task-1)]
                q_h = torch.quantile(class_vals_flt, 0.90)
                class_vals_flt = class_vals_flt[class_vals_flt < q_h]
                q_l = class_vals_flt.min()
                qs = (q_l, q_h)

                reps = []
                dists_all = []
                for it, (input_data, label) in enumerate(loader):
                    dists = []
                    for t_inner in range(task):
                        c, d = self.model.push_forward(input_data.cuda(), t=t_inner)
                        conv_fts = c
                        dists.append(d.cpu())
                    distances = torch.cat(dists, dim=1)
                    if isinstance(self.model.model, TesNet):
                        distances = distances
                    reps.append(conv_fts)
                    dists_all.append(distances)
                reps = torch.cat(reps, dim=0).cpu()
                dists_all = torch.cat(dists_all, dim=0).cpu()

                kmean = KMeans(n_clusters=self.model.heads[0].prototype_vectors.shape[0], max_iter=10)
                inclass_reps = reps.permute(0, 2, 3, 1).reshape(-1, reps.shape[1]).numpy()
                cond = ((dists_all.mean(1).flatten() <= qs[1]))  # * (dists_all.mean(1).flatten() >= qs[0]))
                set_of_reps = inclass_reps[cond]
                kmean.fit(set_of_reps)
                settlers = torch.tensor(kmean.cluster_centers_, dtype=torch.float32).unsqueeze(2).unsqueeze(2).cuda()
                self.model.heads[-1].prototype_vectors.data.copy_(settlers)
                self.settlers = settlers

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

        for images, targets in trn_loader:
            # Forward old model
            distances_old = None
            distances = None
            outputs = self.model(images.to(self.device))
            if t > 0:
                if self.model.model.repeat_task_0:
                    init_t = 1 if t > 1 else 0
                else:
                    init_t = 0
                with torch.no_grad():
                    outputs_old = self.model_old(images.to(self.device))
                    distances_old = torch.cat([outputs_old[i][0] for i in range(init_t, len(outputs_old))], dim=1)

                distances = torch.cat([outputs[i][0] for i in range(init_t, len(outputs_old))], dim=1)
                if self.similarity_reg:
                    distances_old = self.model.heads[-1].distance_2_similarity(distances_old)
                    distances = self.model.heads[-1].distance_2_similarity(distances)
            # Forward current model
            logits = [outputs[i][1] for i in range(len(outputs))]
            min_distances = [outputs[i][2] for i in range(len(outputs))]
            entropy_loss = self.criterion(t, logits, targets.to(self.device), distances, distances_old)
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
                distances_old = None
                distances = None
                outputs = self.model(images.to(self.device))
                if t > 0:
                    with torch.no_grad():
                        outputs_old = self.model_old(images.to(self.device))
                        distances_old = torch.cat([outputs_old[i][0] for i in range(len(outputs_old))], dim=1)
                    distances = torch.cat([outputs[i][0] for i in range(len(outputs_old))], dim=1)
                    if self.similarity_reg:
                        distances_old = self.model.heads[-1].distance_2_similarity(distances_old)
                        distances = self.model.heads[-1].distance_2_similarity(distances)
                # Forward current model
                logits = [outputs[i][1] for i in range(len(outputs))]
                min_distances = [outputs[i][2] for i in range(len(outputs))]
                entropy_loss = self.criterion(t, logits, targets.to(self.device), distances, distances_old)
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
        return total_loss / total_num, total_acc_taw / total_num, total_acc_tag / total_num, ppnet_losses

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

    def criterion(self, t, outputs, targets, distances=None, old_distances=None):
        """Returns the loss value"""
        loss = 0
        t_in = t
        if t > 0:
            if self.model.model.repeat_task_0:
                t_in = t - 1
            if self.normalize_sim:
                old_distances = old_distances / ((torch.sum(old_distances, dim=2).sum(2))[:, :, None, None] + 0.0001)
                distances = distances / (torch.sum(distances, dim=2).sum(2)[:, :, None, None] + 0.0001)
            with torch.no_grad():
                q = torch.quantile(old_distances.reshape([distances.shape[0], distances.shape[1], -1]), self.perc / 100, dim=2)
                if self.similarity_reg:
                    mask = old_distances >= q[:, :, None, None]
                else:
                    mask = old_distances <= q[:, :, None, None]
            # Knowledge distillation loss for all previous tasks
            loss += (self.lamb) * ((old_distances - distances) * mask).view(distances.shape[0], - 1).norm(2, dim=1).sum()
        # Current cross-entropy loss -- with exemplars use all heads
        if len(self.exemplars_dataset) > 0:
            return loss + torch.nn.functional.cross_entropy(torch.cat(outputs, dim=1), targets)
        return loss + torch.nn.functional.cross_entropy(outputs[t], targets - self.model.task_offset[t_in])
