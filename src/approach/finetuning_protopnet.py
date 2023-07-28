import torch
from argparse import ArgumentParser
from sklearn.cluster import KMeans

from .incremental_learning import Inc_Learning_Appr_PPNet
from datasets.exemplars_dataset import ExemplarsDataset


class Appr(Inc_Learning_Appr_PPNet):
    """Class implementing the finetuning baseline"""

    def __init__(self, model, device, nepochs=100, lr=0.05, lr_min=1e-4, lr_factor=3, lr_patience=5, clipgrad=10000,
                 momentum=0, wd=0, multi_softmax=False, wu_nepochs=0, wu_lr_factor=1, fix_bn=False, eval_on_train=False,
                 logger=None, exemplars_dataset=None, all_outputs=True):
        super(Appr, self).__init__(model, device, nepochs, lr, lr_min, lr_factor, lr_patience, clipgrad, momentum, wd,
                                   multi_softmax, wu_nepochs, wu_lr_factor, fix_bn, eval_on_train, logger,
                                   exemplars_dataset)
        self.all_out = all_outputs

    @staticmethod
    def exemplars_dataset_class():
        return ExemplarsDataset

    @staticmethod
    def extra_parser(args):
        """Returns a parser containing the approach specific parameters"""
        parser = ArgumentParser()
        parser.add_argument('--all-outputs', action='store_true', required=False,
                            help='Allow all weights related to all outputs to be modified (default=%(default)s)')
        return parser.parse_known_args(args)

    def _get_optimizers(self, t):
        """Returns the optimizer"""
        warm_params = [
            {'params': self.model.model.add_on_layers.parameters(), 'lr': 3 * self.lr, 'weight_decay': self.wd},
            {'params': self.model.heads[-1].prototype_vectors, 'lr': 3 * self.lr},
            ]
        joint_params = [{'params': self.model.model.features.parameters(), 'lr': self.lr / 10, 'weight_decay': self.wd},
                        {'params': self.model.model.add_on_layers.parameters(), 'lr': 3 * self.lr,
                         'weight_decay': self.wd},
                        {'params': self.model.heads[-1].prototype_vectors, 'lr': 3 * self.lr},
                        ]
        push_params = [{'params': self.model.heads[-1].last_layer.parameters(), 'lr': self.lr / 10,
                        'weight_decay': self.wd},
                       ]
        if len(self.model.neg_heads) > 0:
            push_params.append({'params': self.model.neg_heads[t].parameters(), 'lr': self.lr / 10,
                                'weight_decay': self.wd})
        warm_optimizer = torch.optim.Adam(warm_params)
        joint_optimizer = torch.optim.Adam(joint_params)
        push_optimizer = torch.optim.Adam(push_params)
        return joint_optimizer, push_optimizer, warm_optimizer

    def settlement(self, loader, task=0):
            vals = []
            labels = []
            with torch.no_grad():
                for it, (input_data, label) in enumerate(loader):
                    outputs = self.model(input_data.cuda())
                    distances = torch.cat([outputs[i][0] for i in range(len(outputs))], dim=1)
                    vals.append(distances.reshape(distances.shape[0], distances.shape[1], -1).cpu())
                    labels.append(label)
                vals = torch.cat(vals, dim=0)

                class_vals_flt = torch.unique(vals.flatten()[::(2**(task-1))])
                q_h = torch.quantile(class_vals_flt, 0.60)
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
                    reps.append(conv_fts)
                    dists_all.append(distances)
            reps = torch.cat(reps, dim=0).cpu()
            dists_all = torch.cat(dists_all, dim=0).cpu()

            kmean = KMeans(n_clusters=self.model.heads[0].prototype_vectors.shape[0], max_iter=10)
            inclass_reps = reps.permute(0, 2, 3, 1).reshape(-1, reps.shape[1]).numpy()
            cond = ((dists_all.mean(1).flatten() <= qs[1]) * (dists_all.mean(1).flatten() >= qs[0]))
            set_of_reps = inclass_reps[cond]
            kmean.fit(set_of_reps)
            settlers = torch.tensor(kmean.cluster_centers_, dtype=torch.float32).unsqueeze(2).unsqueeze(2).cuda()
            self.model.heads[-1].prototype_vectors.data.copy_(settlers)
            self.settlers = settlers

    def train_loop(self, t, trn_loader, val_loader, push_load=None):
        """Contains the epochs loop"""

        # add exemplars to train_loader
        if len(self.exemplars_dataset) > 0 and t > 0:
            trn_loader = torch.utils.data.DataLoader(trn_loader.dataset + self.exemplars_dataset,
                                                     batch_size=trn_loader.batch_size,
                                                     shuffle=True,
                                                     num_workers=trn_loader.num_workers,
                                                     pin_memory=trn_loader.pin_memory)

        # FINETUNING TRAINING -- contains the epochs loop
        super().train_loop(t, trn_loader, val_loader, push_load)

        # EXEMPLAR MANAGEMENT -- select training subset
        self.exemplars_dataset.collect_exemplars(self.model, trn_loader, val_loader.dataset.transform)

    def criterion(self, t, outputs, targets):
        """Returns the loss value"""
        if self.all_out or len(self.exemplars_dataset) > 0:
            return torch.nn.functional.cross_entropy(torch.cat(outputs, dim=1), targets)
        return torch.nn.functional.cross_entropy(outputs[t], targets - self.model.task_offset[t])
