import os
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
import torch.nn.functional as F
from torch.nn.functional import gumbel_softmax
from torchvision.transforms import ToTensor

from .resnet_features import resnet18_features, resnet34_features, resnet50_features, resnet101_features, \
    resnet152_features
from .utils_ppnet import compute_proto_layer_rf_info_v2
from .utils_ppnet import compute_rf_prototype
from .utils_ppnet import find_high_activation_crop

base_architecture_to_features = {'resnet18': resnet18_features,
                                 'resnet34': resnet34_features,
                                 'resnet50': resnet50_features,
                                 'resnet101': resnet101_features,
                                 'resnet152': resnet152_features,
                                 'resnet50Nat': resnet50_features,
                                 }


class ProtoPool_head(nn.Module):
    def __init__(self, prototype_shape, num_prototypes, num_descriptive, num_classes, prototype_activation_function,
                 focal, share_add_ons=True, first_add_on_layer_in_channels=None, incorrect_weight=-0.5,
                 incorrect_weight_btw_tasks=False, use_last_layer=True):
        super(ProtoPool_head, self).__init__()
        self.num_prototypes = num_prototypes
        self.num_descriptive = num_descriptive
        self.prototype_shape = prototype_shape
        self.num_classes = num_classes
        self.focal = focal
        self.prototype_activation_function = prototype_activation_function
        self.epsilon = 1e-4
        self.share_add_ons = share_add_ons
        self.incorrect_weight = incorrect_weight

        self.prototype_vectors = nn.Parameter(torch.rand(self.prototype_shape), requires_grad=True)
        # do not make this just a tensor,
        # since it will not be moved automatically to gpu
        self.ones = nn.Parameter(torch.ones(self.prototype_shape), requires_grad=False)

        self.proto_presence = torch.zeros(num_classes, num_prototypes, num_descriptive)  # [c, p, n]
        # for j in range(num_classes):
        #     for k in range(num_descriptive):
        #         self.proto_presence[j, j * num_descriptive + k, k] = 1
        self.proto_presence = Parameter(self.proto_presence, requires_grad=True)
        nn.init.xavier_normal_(self.proto_presence, gain=1.0)

        self.use_last_layer = use_last_layer
        if self.use_last_layer:
            self.prototype_class_identity = torch.zeros(self.num_descriptive * self.num_classes, self.num_classes).cuda()

            for j in range(self.num_descriptive * self.num_classes):
                self.prototype_class_identity[j, j // self.num_descriptive] = 1
            self.last_layer = nn.Linear(self.num_descriptive * self.num_classes, self.num_classes, bias=False)
            self.set_last_layer_incorrect_connection(incorrect_strength=self.incorrect_weight)
        else:
            self.last_layer = nn.Identity()
        self.out_features = self.last_layer.out_features

        if not self.share_add_ons:
            add_on_layers = [
                nn.Conv2d(in_channels=first_add_on_layer_in_channels, out_channels=self.prototype_shape[1],
                          kernel_size=1),
                # nn.ReLU(),
                # nn.Conv2d(in_channels=self.prototype_shape[1], out_channels=self.prototype_shape[1], kernel_size=1),
                nn.Sigmoid(),
            ]
            self.add_on_layers = nn.Sequential(*add_on_layers)

    def l2_convolution(self, x):
        '''
        apply self.prototype_vectors as l2-convolution filters on input x
        '''
        x2 = x ** 2
        x2_patch_sum = F.conv2d(input=x2, weight=self.ones)

        p2 = self.prototype_vectors ** 2
        p2 = torch.sum(p2, dim=(1, 2, 3))
        # p2 is a vector of shape (num_prototypes,)
        # then we reshape it to (num_prototypes, 1, 1)
        p2_reshape = p2.view(-1, 1, 1)

        xp = F.conv2d(input=x, weight=self.prototype_vectors)
        intermediate_result = - 2 * xp + p2_reshape  # use broadcast
        # x2_patch_sum and intermediate_result are of the same shape
        distances = F.relu(x2_patch_sum + intermediate_result)

        return distances

    def forward(self, conv_features, gumbel_scale=0):
        if gumbel_scale == 0:
            proto_presence = torch.softmax(self.proto_presence, dim=1)
        else:
            proto_presence = gumbel_softmax(self.proto_presence * gumbel_scale, dim=1, tau=0.5)
        if not self.share_add_ons:
            conv_features = self.add_on_layers(conv_features)
        distances = self.l2_convolution(conv_features)

        '''
        we cannot refactor the lines below for similarity scores
        because we need to return min_distances
        '''
        # global min pooling
        min_distances = -F.max_pool2d(-distances,
                                      kernel_size=(distances.size()[2],
                                                   distances.size()[3])).squeeze()  # [b, p]
        avg_dist = F.avg_pool2d(distances, kernel_size=(distances.size()[2],
                                                        distances.size()[3])).squeeze()  # [b, p]
        min_mixed_distances = self._mix_l2_convolution(min_distances, proto_presence)  # [b, c, n]
        avg_mixed_distances = self._mix_l2_convolution(avg_dist, proto_presence)  # [b, c, n]
        x = self.distance_2_similarity(min_mixed_distances)  # [b, c, n]
        x_avg = self.distance_2_similarity(avg_mixed_distances)  # [b, c, n]
        x = x - x_avg
        # x = self.distance_2_similarity(min_distances)
        if self.use_last_layer:
            x = self.last_layer(x.flatten(start_dim=1))
        else:
            x = x.sum(dim=-1)
        return distances, x, min_distances, None, proto_presence  # [b,c,n] [b, p] [c, p, n]

    def distance_2_similarity(self, distances):  # [b,c,n]
        if self.prototype_activation_function == 'log':
            return torch.log((distances + 1) / (distances + self.epsilon))
        elif self.prototype_activation_function == 'linear':
            if self.use_thresh:
                distances = distances  # * torch.exp(self.alfa)  # [b, c, n]
            return 1 / (distances + 1)
        else:
            raise NotImplementedError

    def _mix_l2_convolution(self, distances, proto_presence):
        '''
        apply self.prototype_vectors as l2-convolution filters on input x
        '''
        # distances [b, p]
        # proto_presence [c, p, n]
        mixed_distances = torch.einsum('bp,cpn->bcn', distances, proto_presence)

        return mixed_distances  # [b, c, n]

    def get_map_class_to_prototypes(self):
        pp = gumbel_softmax(self.proto_presence * 10e3, dim=1, tau=0.5).detach()
        return np.argmax(pp.cpu().numpy(), axis=1)

    def focal_similarity(self, distances, min_similarities):
        avg_dist = F.avg_pool2d(distances, kernel_size=(distances.size()[2],
                                                        distances.size()[3])).squeeze()
        x = min_similarities
        x_avg = self.distance_2_similarity(avg_dist)  # [b, c, n]
        prototype_activations = x - x_avg
        return prototype_activations

    def set_last_layer_incorrect_connection(self, incorrect_strength):
        '''
        the incorrect strength will be actual strength if -0.5 then input -0.5
        '''
        positive_one_weights_locations = torch.t(self.prototype_class_identity)
        negative_one_weights_locations = 1 - positive_one_weights_locations

        correct_class_connection = 1
        incorrect_class_connection = 0  # -0.5
        self.last_layer.weight.data.copy_(
            correct_class_connection * positive_one_weights_locations
            + incorrect_class_connection * negative_one_weights_locations)

    def fine_tune_last_only(self):
        if not self.share_add_ons:
            for p in self.add_on_layers.parameters():
                p.requires_grad = False
        self.prototype_vectors.requires_grad = False
        self.proto_presence.requires_grad = False
        for p in self.last_layer.parameters():
            p.requires_grad = True


class ProtoPool(nn.Module):

    def __init__(self, features, img_size, prototype_shape,
                 proto_layer_rf_info, num_classes, init_weights=True,
                 prototype_activation_function='log',
                 add_on_layers_type='bottleneck',
                 focal=False, warm_num=10, push_at=10, num_push_tune=10, sep_weight=-0.08, share_add_ons=True,
                 incorrect_weight=-0.5, incorrect_weight_btw_tasks=False, repeat_task_0=False,

                 num_prototypes=202, num_descriptive=10, use_thresh=False, arch='resnet34', pretrained=True,
                 proto_depth=128, use_last_layer=False, inat=False, pp_ortho=True, pp_gumbel=True, gumbel_time=30
                 ):

        super(ProtoPool, self).__init__()
        self.focal = focal
        self.img_size = img_size
        self.prototype_shape = prototype_shape
        self.num_prototypes = prototype_shape[0]
        self.num_classes = num_classes
        self.epsilon = 1e-4
        self.prototypes_per_class = int(self.num_prototypes // self.num_classes)
        self.warm_num = warm_num
        self.push_at = push_at
        self.num_push_tune = num_push_tune
        self.sep_weight = sep_weight
        self.share_add_ons = share_add_ons
        self.incorrect_weight = incorrect_weight
        self.incorrect_weight_btw_tasks = incorrect_weight_btw_tasks
        self.repeat_task_0 = repeat_task_0

        self.num_descriptive = num_descriptive
        self.use_thresh = use_thresh
        self.arch = arch
        self.pretrained = pretrained,
        self.proto_depth = proto_depth
        self.use_last_layer = use_last_layer
        self.inat = inat
        self.pp_ortho = pp_ortho
        self.pp_gumbel = pp_gumbel
        self.gumbel_time = gumbel_time
        self.prototype_activation_function = prototype_activation_function
        self.prototype_shape = (self.num_prototypes, self.proto_depth, 1, 1)
        if self.use_thresh:
            self.alfa = Parameter(torch.Tensor(1, num_classes, num_descriptive))
            nn.init.xavier_normal_(self.alfa, gain=1.0)
        else:
            self.alfa = 1
            self.beta = 0

        self.proto_layer_rf_info = proto_layer_rf_info

        # this has to be named features to allow the precise loading
        self.features = features
        features_name = str(self.features).upper()
        if features_name.startswith('VGG') or features_name.startswith('RES'):
            first_add_on_layer_in_channels = \
                [i for i in features.modules() if isinstance(i, nn.Conv2d)][-1].out_channels
        elif features_name.startswith('DENSE'):
            first_add_on_layer_in_channels = \
                [i for i in features.modules() if isinstance(i, nn.BatchNorm2d)][-1].num_features
        else:
            raise Exception('other base base_architecture NOT implemented')

        self.first_add_on_layer_in_channels = first_add_on_layer_in_channels

        if self.share_add_ons:
            if add_on_layers_type == 'bottleneck':
                add_on_layers = []
                current_in_channels = first_add_on_layer_in_channels
                while (current_in_channels > self.prototype_shape[1]) or (len(add_on_layers) == 0):
                    current_out_channels = max(self.prototype_shape[1], (current_in_channels // 2))
                    add_on_layers.append(nn.Conv2d(in_channels=current_in_channels,
                                                   out_channels=current_out_channels,
                                                   kernel_size=1))
                    add_on_layers.append(nn.ReLU())
                    add_on_layers.append(nn.Conv2d(in_channels=current_out_channels,
                                                   out_channels=current_out_channels,
                                                   kernel_size=1))
                    if current_out_channels > self.prototype_shape[1]:
                        add_on_layers.append(nn.ReLU())
                    else:
                        assert (current_out_channels == self.prototype_shape[1])
                        add_on_layers.append(nn.Sigmoid())
                    current_in_channels = current_in_channels // 2
                self.add_on_layers = nn.Sequential(*add_on_layers)
            else:
                add_on_layers = [
                    nn.Conv2d(in_channels=first_add_on_layer_in_channels, out_channels=self.prototype_shape[1],
                              kernel_size=1),
                    # nn.ReLU(),
                    # nn.Conv2d(in_channels=self.prototype_shape[1], out_channels=self.prototype_shape[1], kernel_size=1),
                    nn.Sigmoid(),
                ]

                self.add_on_layers = nn.Sequential(*add_on_layers)

        self.protopool_head = ProtoPool_head(self.prototype_shape, self.num_prototypes, self.num_descriptive,
                                             self.num_classes, self.prototype_activation_function,
                                             self.focal, self.share_add_ons,
                                             first_add_on_layer_in_channels, incorrect_weight=self.incorrect_weight,
                                             use_last_layer=use_last_layer)

        self.head_var = 'protopool_head'

        if init_weights:
            self._initialize_weights()

        self.start_val = 1.3
        self.end_val = 10 ** 3
        self.epoch_interval = self.gumbel_time
        self.alpha = (self.end_val / self.start_val) ** 2 / self.epoch_interval

    def lambda1(self, epoch):
        return self.start_val * np.sqrt(self.alpha *
                                        (epoch)) if epoch < self.epoch_interval else self.end_val

    def conv_features(self, x):
        '''
        the feature input to prototype layer
        '''
        x = self.features(x)
        if self.share_add_ons:
            x = self.add_on_layers(x)
        return x

    def forward(self, x, gumbel_scale=0):
        conv_features = self.conv_features(x)
        distances, logits, min_distances, _, proto_presence = (
            self.protopool_head(conv_features, gumbel_scale))
        return logits, min_distances, self.prototype_class_identity, distances, proto_presence

    def prototype_distances(self, x):
        '''
        x is the raw input
        '''
        conv_features = self.conv_features(x)
        distances = self.protopool_head._l2_convolution(conv_features)  # [b, p, h, w]
        return distances  # [b, n, h, w], [b, p, h, w]

    def fine_tune_last_only(self):
        for p in self.features.parameters():
            p.requires_grad = False
        if self.share_add_ons:
            for p in self.add_on_layers.parameters():
                p.requires_grad = False
        self.protopool_head.fine_tune_last_only()

    def __repr__(self):
        rep = (
            'PPNet(\n'
            '\tfeatures: {},\n'
            '\timg_size: {},\n'
            '\tprototype_shape: {},\n'
            '\tproto_layer_rf_info: {},\n'
            '\tnum_classes: {},\n'
            '\tepsilon: {}\n'
            ')'
        )

        return rep.format(self.features,
                          self.img_size,
                          self.prototype_shape,
                          self.proto_layer_rf_info,
                          self.num_classes,
                          self.epsilon)

    def get_map_class_to_prototypes(self):
        return self.protopool_head.get_map_class_to_prototypes()

    def _initialize_weights(self):
        if self.share_add_ons:
            for m in self.add_on_layers.modules():
                if isinstance(m, (nn.Linear, nn.Conv2d)):
                    nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)

                elif isinstance(m, nn.BatchNorm2d):
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)


def dist_loss(model, min_distances, proto_presence, top_k, label, sep=False, all_out=False, t=0):
    #         model, [b, p],        [b, p, n],      [scalar]
    max_dist = (model.prototype_shape[1]
                * model.prototype_shape[2]
                * model.prototype_shape[3])
    basic_proto = proto_presence.sum(dim=-1).detach()  # [b, p]
    _, idx = torch.topk(basic_proto, top_k, dim=1)  # [b, n]
    binarized_top_k = torch.zeros_like(basic_proto)
    binarized_top_k.scatter_(1, src=torch.ones_like(
        basic_proto), index=idx)  # [b, p]
    if all_out and t > 0:
        prototypes_of_correct_class_m = torch.zeros(((t + 1) * model.num_classes,
                                                     (t + 1) * model.num_prototypes,))
        for i in range(t + 1):
            prototypes_of_correct_class_m[
            i * model.num_classes:(i + 1) * model.num_classes,
            i * model.num_prototypes:(i + 1) * model.num_prototypes] = \
                binarized_top_k
        prototypes_of_correct_class = prototypes_of_correct_class_m[label, :].cuda()
    else:
        prototypes_of_correct_class = binarized_top_k[label, :].cuda()
    inverted_distances, _ = torch.max((max_dist - min_distances) * prototypes_of_correct_class, dim=1)
    cost = torch.mean(max_dist - inverted_distances)
    return cost


def update_prototypes_on_batch_protopool(search_batch_input,
                                         start_index_of_search_batch,
                                         model_lll,
                                         global_min_proto_dist,  # this will be updated
                                         global_min_fmap_patches,  # this will be updated
                                         proto_rf_boxes,  # this will be updated
                                         proto_bound_boxes,  # this will be updated
                                         class_specific=True,
                                         search_y=None,  # required if class_specific == True
                                         num_classes=None,  # required if class_specific == True
                                         preprocess_input_function=None,
                                         prototype_layer_stride=1,
                                         dir_for_saving_prototypes=None,
                                         prototype_img_filename_prefix=None,
                                         prototype_self_act_filename_prefix=None,
                                         prototype_activation_function_in_numpy=None,
                                         prototype_list=None,
                                         task=None):
    model_lll.eval()
    search_batch = search_batch_input

    with torch.no_grad():
        search_batch = search_batch.cuda()
        # this computation currently is not parallelized
        protoL_input_torch, proto_dist_torch = model_lll.push_forward(search_batch)

    protoL_input_ = np.copy(protoL_input_torch.detach().cpu().numpy())
    proto_dist_ = np.copy(proto_dist_torch.detach().cpu().numpy())

    del protoL_input_torch, proto_dist_torch

    prototype_shape = model_lll.model.prototype_shape
    n_prototypes = prototype_shape[0]
    proto_h = prototype_shape[2]
    proto_w = prototype_shape[3]
    max_dist = prototype_shape[1] * prototype_shape[2] * prototype_shape[3]

    if class_specific:
        map_class_to_prototypes = model_lll.model.get_map_class_to_prototypes()
        protype_to_img_index_dict = {key: [] for key in range(n_prototypes)}
        # img_y is the image's integer label

        for img_index, img_y in enumerate(search_y):
            img_label = img_y.item()
            [protype_to_img_index_dict[prototype].append(
                img_index) for prototype in map_class_to_prototypes[img_label]]

    for j in range(n_prototypes):
        if class_specific:
            # target_class is the class of the class_specific prototype

            # if there is not images of the target_class from this batch
            # we go on to the next prototype
            if len(protype_to_img_index_dict[j]) == 0:
                continue
            proto_dist_j = proto_dist_[protype_to_img_index_dict[j]][:, j]
        else:
            # if it is not class specific, then we will search through
            # every example
            proto_dist_j = proto_dist_[:, j]

        batch_min_proto_dist_j = np.amin(proto_dist_j)

        if batch_min_proto_dist_j < global_min_proto_dist[j]:
            batch_argmin_proto_dist_j = \
                list(np.unravel_index(np.argmin(proto_dist_j, axis=None),
                                      proto_dist_j.shape))
            if class_specific:
                '''
                change the argmin index from the index among
                images of the target class to the index in the entire search
                batch
                '''

                batch_argmin_proto_dist_j[0] = protype_to_img_index_dict[j][batch_argmin_proto_dist_j[0]]

            # retrieve the corresponding feature map patch
            img_index_in_batch = batch_argmin_proto_dist_j[0]
            fmap_height_start_index = batch_argmin_proto_dist_j[1] * \
                                      prototype_layer_stride
            fmap_height_end_index = fmap_height_start_index + proto_h
            fmap_width_start_index = batch_argmin_proto_dist_j[2] * \
                                     prototype_layer_stride
            fmap_width_end_index = fmap_width_start_index + proto_w

            batch_min_fmap_patch_j = protoL_input_[img_index_in_batch,
                                     :,
                                     fmap_height_start_index:fmap_height_end_index,
                                     fmap_width_start_index:fmap_width_end_index]

            global_min_proto_dist[j] = batch_min_proto_dist_j
            global_min_fmap_patches[j] = batch_min_fmap_patch_j

            # get the receptive field boundary of the image patch
            # that generates the representation
            # protoL_rf_info = model.proto_layer_rf_info
            layer_filter_sizes, layer_strides, layer_paddings = model_lll.model.features.conv_info()
            protoL_rf_info = compute_proto_layer_rf_info_v2(224, layer_filter_sizes, layer_strides, layer_paddings,
                                                            prototype_kernel_size=1)
            rf_prototype_j = compute_rf_prototype(search_batch.size(2), batch_argmin_proto_dist_j, protoL_rf_info)

            # get the whole image
            original_img_j = search_batch_input[rf_prototype_j[0]]
            original_img_j = original_img_j.numpy()
            original_img_j = np.transpose(original_img_j, (1, 2, 0))
            original_img_size = original_img_j.shape[0]
            original_img_j = (original_img_j - np.min(original_img_j)) / np.max(original_img_j - np.min(original_img_j))

            # crop out the receptive field
            rf_img_j = original_img_j[rf_prototype_j[1]:rf_prototype_j[2],
                       rf_prototype_j[3]:rf_prototype_j[4], :]

            # save the prototype receptive field information
            proto_rf_boxes[j, 0] = rf_prototype_j[0] + start_index_of_search_batch
            proto_rf_boxes[j, 1] = rf_prototype_j[1]
            proto_rf_boxes[j, 2] = rf_prototype_j[2]
            proto_rf_boxes[j, 3] = rf_prototype_j[3]
            proto_rf_boxes[j, 4] = rf_prototype_j[4]
            if proto_rf_boxes.shape[1] == 6 and search_y is not None:
                proto_rf_boxes[j, 5] = search_y[rf_prototype_j[0]].item()

            # find the highly activated region of the original image
            proto_dist_img_j = proto_dist_[img_index_in_batch, j, :, :]
            if model_lll.model.prototype_activation_function == 'log':
                proto_act_img_j = np.log((proto_dist_img_j + 1) / (proto_dist_img_j + model_lll.model.epsilon))
            elif model_lll.model.prototype_activation_function == 'linear':
                proto_act_img_j = max_dist - proto_dist_img_j
            else:
                proto_act_img_j = prototype_activation_function_in_numpy(proto_dist_img_j)
            upsampled_act_img_j = cv2.resize(proto_act_img_j, dsize=(original_img_size, original_img_size),
                                             interpolation=cv2.INTER_CUBIC)
            proto_bound_j = find_high_activation_crop(upsampled_act_img_j)
            # crop out the image patch with high activation as prototype image
            proto_img_j = original_img_j[proto_bound_j[0]:proto_bound_j[1],
                          proto_bound_j[2]:proto_bound_j[3], :]

            # save the prototype boundary (rectangular boundary of highly activated region)
            proto_bound_boxes[j, 0] = proto_rf_boxes[j, 0]
            proto_bound_boxes[j, 1] = proto_bound_j[0]
            proto_bound_boxes[j, 2] = proto_bound_j[1]
            proto_bound_boxes[j, 3] = proto_bound_j[2]
            proto_bound_boxes[j, 4] = proto_bound_j[3]
            if proto_bound_boxes.shape[1] == 6 and search_y is not None:
                proto_bound_boxes[j, 5] = search_y[rf_prototype_j[0]].item()

            if dir_for_saving_prototypes is not None:
                if prototype_self_act_filename_prefix is not None:
                    # save the numpy array of the prototype self activation
                    np.save(os.path.join(dir_for_saving_prototypes,
                                         prototype_self_act_filename_prefix + str(j) + '.npy'),
                            proto_act_img_j)
                if prototype_img_filename_prefix is not None:
                    # save the whole image containing the prototype as png
                    plt.imsave(os.path.join(dir_for_saving_prototypes,
                                            prototype_img_filename_prefix + '-original' + str(j) + '.png'),
                               original_img_j,
                               vmin=0.0,
                               vmax=1.0)
                    # overlay (upsampled) self activation on original image and save the result
                    rescaled_act_img_j = upsampled_act_img_j - np.amin(upsampled_act_img_j)
                    rescaled_act_img_j = rescaled_act_img_j / np.amax(rescaled_act_img_j)
                    heatmap = cv2.applyColorMap(np.uint8(255 * rescaled_act_img_j), cv2.COLORMAP_JET)
                    heatmap = np.float32(heatmap) / 255
                    heatmap = heatmap[..., ::-1]
                    overlayed_original_img_j = 0.5 * original_img_j + 0.3 * heatmap
                    plt.imsave(os.path.join(dir_for_saving_prototypes,
                                            prototype_img_filename_prefix + '-original_with_self_act' + str(
                                                j) + '.png'),
                               overlayed_original_img_j,
                               vmin=0.0,
                               vmax=1.0)

                    # if different from the original (whole) image, save the prototype receptive field as png
                    if rf_img_j.shape[0] != original_img_size or rf_img_j.shape[1] != original_img_size:
                        plt.imsave(os.path.join(dir_for_saving_prototypes,
                                                prototype_img_filename_prefix + '-receptive_field' + str(j) + '.png'),
                                   rf_img_j,
                                   vmin=0.0,
                                   vmax=1.0)
                        overlayed_rf_img_j = overlayed_original_img_j[rf_prototype_j[1]:rf_prototype_j[2],
                                             rf_prototype_j[3]:rf_prototype_j[4]]
                        plt.imsave(os.path.join(dir_for_saving_prototypes,
                                                prototype_img_filename_prefix + '-receptive_field_with_self_act' + str(
                                                    j) + '.png'),
                                   overlayed_rf_img_j,
                                   vmin=0.0,
                                   vmax=1.0)

                    # save the prototype image (highly activated region of the whole image)
                    plt.imsave(os.path.join(dir_for_saving_prototypes,
                                            prototype_img_filename_prefix + str(j) + '.png'),
                               proto_img_j,
                               vmin=0.0,
                               vmax=1.0)


def construct_ProtoPool(base_architecture, pretrained=True, img_size=224,
                    prototype_shape=(202, 512, 1, 1), num_classes=200,
                    prototype_activation_function='log',
                    add_on_layers_type='bottleneck', focal=False, warm_num=10, push_at=10, num_push_tune=10,
                    sep_weight=-0.08, share_add_ons=True, incorrect_weight=-0.5,
                    incorrect_weight_btw_tasks=False,
                    repeat_task_0=False,

                    pp_ortho=True,
                    pp_gumbel=True,
                    gumbel_time=30,
                    num_prototypes=202,
                    num_descriptive=10,
                    use_thresh=True,
                    proto_depth=512,
                    use_last_layer=True,
                    inat=True):
    if 'Nat' in base_architecture:
        features = base_architecture_to_features[base_architecture](pretrained=pretrained, inat=True)
    else:
        features = base_architecture_to_features[base_architecture](pretrained=pretrained)
    layer_filter_sizes, layer_strides, layer_paddings = features.conv_info()
    proto_layer_rf_info = compute_proto_layer_rf_info_v2(img_size=img_size,
                                                         layer_filter_sizes=layer_filter_sizes,
                                                         layer_strides=layer_strides,
                                                         layer_paddings=layer_paddings,
                                                         prototype_kernel_size=prototype_shape[2])
    return ProtoPool(features=features,
                     img_size=img_size,
                     prototype_shape=prototype_shape,
                     proto_layer_rf_info=proto_layer_rf_info,
                     num_classes=num_classes,
                     init_weights=True,
                     prototype_activation_function=prototype_activation_function,
                     add_on_layers_type=add_on_layers_type,
                     focal=focal,
                     warm_num=warm_num,
                     push_at=push_at,
                     num_push_tune=num_push_tune,
                     sep_weight=sep_weight,
                     share_add_ons=share_add_ons,
                     incorrect_weight=incorrect_weight,
                     incorrect_weight_btw_tasks=incorrect_weight_btw_tasks,
                     repeat_task_0=repeat_task_0,

                     pp_ortho=pp_ortho,
                     pp_gumbel=pp_gumbel,
                     gumbel_time=gumbel_time,
                     num_prototypes=num_prototypes,
                     num_descriptive=num_descriptive,
                     use_thresh=use_thresh,
                     proto_depth=proto_depth,
                     use_last_layer=use_last_layer,
                     inat=inat
                     )


def push(global_min_fmap_patches, global_min_proto_dist, model_lll, proto_bound_boxes, proto_rf_boxes,
         search_batch_input, search_y, start_index_of_search_batch, prototype_class_identity=None, task=0, log_path='./'):
    proto_img_dir = f'{log_path}/img_proto/task_{task}'
    Path(proto_img_dir).mkdir(parents=True, exist_ok=True)
    prototype_list = torch.zeros((model_lll.model.num_prototypes, 3, 224, 224))
    update_prototypes_on_batch_protopool(search_batch_input=search_batch_input,
                                         start_index_of_search_batch=start_index_of_search_batch,
                                         model_lll=model_lll,
                                         global_min_proto_dist=global_min_proto_dist,
                                         global_min_fmap_patches=global_min_fmap_patches,
                                         proto_rf_boxes=proto_rf_boxes,
                                         proto_bound_boxes=proto_bound_boxes,
                                         class_specific=True,
                                         search_y=search_y,
                                         num_classes=model_lll.model.num_classes,
                                         prototype_layer_stride=1,
                                         dir_for_saving_prototypes=proto_img_dir,
                                         prototype_img_filename_prefix='prototype-img',
                                         prototype_self_act_filename_prefix='prototype-self-act',
                                         prototype_activation_function_in_numpy=None,
                                         prototype_list=prototype_list,
                                         task=task)

