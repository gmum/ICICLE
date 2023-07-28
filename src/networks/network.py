import torch
from torch import nn
from copy import deepcopy

from networks.protopartnet import PPNet, PPNet_head
from networks.tesnet import TesNet, TesNet_head


class LLL_Net(nn.Module):
    """Basic class for implementing networks"""

    def __init__(self, model, remove_existing_head=False):
        head_var = model.head_var
        assert type(head_var) == str
        assert not remove_existing_head or hasattr(model, head_var), \
            "Given model does not have a variable called {}".format(head_var)
        assert not remove_existing_head or type(getattr(model, head_var)) in [nn.Sequential, nn.Linear, PPNet_head,
                                                                              TesNet_head], \
            "Given model's head {} does is not an instance of nn.Sequential or nn.Linear".format(head_var)
        super(LLL_Net, self).__init__()

        self.model = model
        last_layer = getattr(self.model, head_var)

        if remove_existing_head:
            if type(last_layer) == nn.Sequential:
                self.out_size = last_layer[-1].in_features
                # strips off last linear layer of classifier
                del last_layer[-1]
            elif type(last_layer) == nn.Linear:
                self.out_size = last_layer.in_features
                # converts last layer into identity
                # setattr(self.model, head_var, nn.Identity())
                # WARNING: this is for when pytorch version is <1.2
                setattr(self.model, head_var, nn.Sequential())
            elif type(last_layer) == PPNet_head:
                self.out_size = (self.model.prototype_shape[1], 7, 7)
                # converts last layer into identity
                # setattr(self.model, head_var, nn.Identity())
                # WARNING: this is for when pytorch version is <1.2
                setattr(self.model, head_var, nn.Sequential())
        else:
            self.out_size = last_layer.out_features

        self.heads = nn.ModuleList()
        self.task_cls = []
        self.task_offset = []
        self._initialize_weights()

    def add_head(self, num_outputs):
        """Add a new head with the corresponding number of outputs. Also update the number of classes per task and the
        corresponding offsets
        """
        self.heads.append(nn.Linear(self.out_size, num_outputs))
        # we re-compute instead of append in case an approach makes changes to the heads
        self.task_cls = torch.tensor([head.out_features for head in self.heads])
        self.task_offset = torch.cat([torch.LongTensor(1).zero_(), self.task_cls.cumsum(0)[:-1]])

    def forward(self, x, return_features=False):
        """Applies the forward pass

        Simplification to work on multi-head only -- returns all head outputs in a list
        Args:
            x (tensor): input images
            return_features (bool): return the representations before the heads
        """
        x = self.model(x)
        assert (len(self.heads) > 0), "Cannot access any head"
        y = []
        for head in self.heads:
            y.append(head(x))
        if return_features:
            return y, x
        else:
            return y

    def get_copy(self):
        """Get weights from the model"""
        return deepcopy(self.state_dict())

    def set_state_dict(self, state_dict):
        """Load weights into the model"""
        self.load_state_dict(deepcopy(state_dict))
        return

    def freeze_all(self):
        """Freeze all parameters from the model, including the heads"""
        for param in self.parameters():
            param.requires_grad = False

    def freeze_backbone(self):
        """Freeze all parameters from the main model, but not the heads"""
        for param in self.model.parameters():
            param.requires_grad = False

    def freeze_bn(self):
        """Freeze all Batch Normalization layers from the model and use them in eval() mode"""
        for m in self.model.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()

    def _initialize_weights(self):
        """Initialize weights using different strategies"""
        # TODO: add different initialization strategies
        pass


class LLL_Net_PPNet(LLL_Net):
    def __init__(self, model, remove_existing_head=False):
        super(LLL_Net_PPNet, self).__init__(model, remove_existing_head)
        self.neg_heads = nn.ModuleList()

    def add_head(self, num_outputs):
        """Add a new head with the corresponding number of outputs. Also update the number of classes per task and the
        corresponding offsets
        """
        if isinstance(self.model, PPNet):
            self.heads.append(
                PPNet_head(
                    self.model.prototype_shape,
                    num_outputs * self.model.prototypes_per_class,
                    num_outputs,
                    self.model.prototype_activation_function,
                    self.model.focal,
                    incorrect_weight=self.model.incorrect_weight,
                    share_add_ons=self.model.share_add_ons,
                    first_add_on_layer_in_channels=self.model.first_add_on_layer_in_channels,
                )
            )
        else:
            self.heads.append(
                TesNet_head(
                    self.model.prototype_shape,
                    num_outputs * self.model.prototypes_per_class,
                    num_outputs,
                    self.model.prototype_activation_function,
                    self.model.focal,
                    incorrect_weight=self.model.incorrect_weight,
                    share_add_ons=self.model.share_add_ons,
                    first_add_on_layer_in_channels=self.model.first_add_on_layer_in_channels,
                )
            )
        if self.model.incorrect_weight_btw_tasks and (len(self.heads) > 1):
            neg_ll = nn.Linear(len(self.heads) * num_outputs * self.model.prototypes_per_class,
                               len(self.heads) * num_outputs, bias=False)
            t = len(self.heads)
            prototypes_of_correct_class_m = torch.zeros(((t) * self.model.num_prototypes,
                                                         (t) * self.model.num_classes,))
            for i in range(t):
                prototypes_of_correct_class_m[
                i * self.model.num_prototypes:(i + 1) * self.model.num_prototypes,
                i * self.model.num_classes:(i + 1) * self.model.num_classes] = \
                    self.heads[t - 1].prototype_class_identity
            positive_one_weights_locations = torch.t(prototypes_of_correct_class_m).cuda()
            negative_one_weights_locations = 1 - positive_one_weights_locations

            correct_class_connection = 0
            incorrect_class_connection = self.model.incorrect_weight
            neg_ll.weight.data.copy_(
                correct_class_connection * positive_one_weights_locations
                + incorrect_class_connection * negative_one_weights_locations)
            self.neg_heads.append(neg_ll)
        # we re-compute instead of append in case an approach makes changes to the heads
        self.task_cls = torch.tensor([head.out_features for head in self.heads])
        self.task_offset = torch.cat([torch.LongTensor(1).zero_(), self.task_cls.cumsum(0)[:-1]])

    def forward(self, x, return_features=False):
        """Applies the forward pass

        Simplification to work on multi-head only -- returns all head outputs in a list
        Args:
            x (tensor): input images
            return_features (bool): return the representations before the heads
        """
        x = self.model.conv_features(x)
        assert (len(self.heads) > 0), "Cannot access any head"
        y = []
        for i in range(len(self.heads)):
            y.append(self.heads[i](x))
        if self.model.incorrect_weight_btw_tasks and len(self.heads) > 1:
            min_dist = torch.cat([y[k][2] for k in range(len(y))], dim=1)
            neg_out = self.neg_heads[-1](self.heads[-1].distance_2_similarity(min_dist))
            for j in range(len(y)):
                num_out = y[j][1].shape[1]
                y[j] = list(y[j])
                y[j][1] = y[j][1] + neg_out[:, j * num_out:(j+1)*num_out]
        if return_features:
            return y, x
        else:
            return y

    def push_forward(self, x, t):
        if isinstance(self.model, PPNet):
            return self.push_forward_p(x, t)
        else:
            return self.push_forward_t(x, t)

    def push_forward_p(self, x, t):
        '''this method is needed for the pushing operation'''
        conv_output = self.model.conv_features(x)
        if self.model.share_add_ons:
            distances = self.heads[t].l2_convolution(conv_output)
        else:
            conv_output = self.heads[t].add_on_layers(conv_output)
            distances = self.heads[t].l2_convolution(conv_output)
        return conv_output, distances

    def push_forward_t(self, x, t):
        '''this method is needed for the pushing operation'''
        conv_output = self.model.conv_features(x)

        if self.model.share_add_ons:
            distances = -self.heads[t]._project2basis(conv_output)
        else:
            conv_output = self.heads[t].add_on_layers(conv_output)
            distances = -self.heads[t]._project2basis(conv_output)
        return conv_output, distances
