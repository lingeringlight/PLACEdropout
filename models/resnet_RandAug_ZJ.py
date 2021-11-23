import functools
from pathlib import Path

import torch
import torch.nn as nn
from torchvision.models import *
import torch.nn.functional as F
# from model.ibnnet import resnet18_ibn_b
# from model.in_model import InstanceNorm2d
# from utils.visualize import show_graphs


# def ibnnet_b(num_classes, pretrained, args):
#     model = resnet18_ibn_b(pretrained=pretrained)
#     num_ftrs = model.fc.in_features
#     model.fc = nn.Linear(num_ftrs, num_classes)
#     nn.init.xavier_uniform_(model.fc.weight, .1)
#     nn.init.constant_(model.fc.bias, 0.)
#     return model


def resnet_18(num_classes, pretrained, args):
    model = resnet18(pretrained=pretrained)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes)
    nn.init.xavier_uniform_(model.fc.weight, .1)
    nn.init.constant_(model.fc.bias, 0.)
    return model


def resnet_50(num_classes, pretrained, args):
    model = resnet50(pretrained=pretrained)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes)
    nn.init.xavier_uniform_(model.fc.weight, .1)
    nn.init.constant_(model.fc.bias, 0.)
    return model


def resnet_101(num_classes, pretrained, args):
    model = resnet101(pretrained=pretrained)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes)
    nn.init.xavier_uniform_(model.fc.weight, .1)
    nn.init.constant_(model.fc.bias, 0.)
    return model


# def convert_to_target(net, norm, start=0, end=5, verbose=True, res50=False):
#     def convert_norm(old_norm, new_norm, num_features, append=True):
#         norm_layer = new_norm(num_features)
#         if hasattr(norm_layer, 'load_old_dict'):
#             if verbose:
#                 print('Converted to : {}'.format(norm))
#             norm_layer.load_old_dict(old_norm)
#         elif hasattr(norm_layer, 'load_state_dict'):
#             state_dict = old_norm.state_dict()
#             # state_dict.pop('running_mean')
#             # state_dict.pop('running_var')
#             # state_dict.pop('num_batches_tracked')
#             ret = norm_layer.load_state_dict(state_dict, strict=False)
#             print(ret)
#         else:
#             print('No load_old_dict() found!!!')
#         return norm_layer
#
#     layers = [0, net.layer1, net.layer2, net.layer3, net.layer4]
#
#     converted_bns = {}
#     for i, layer in enumerate(layers):
#         if not (start <= i < end):
#             continue
#         if i == 0:
#             net.bn1 = convert_norm(net.bn1, norm, net.bn1.num_features)
#         else:
#             append = True
#             for j, bottleneck in enumerate(layer):
#                 bottleneck.bn1 = convert_norm(bottleneck.bn1, norm, bottleneck.bn1.num_features, append)
#                 converted_bns['{}-{}-{}'.format(i, j, 0)] = bottleneck.bn1
#                 bottleneck.bn2 = convert_norm(bottleneck.bn2, norm, bottleneck.bn2.num_features, append)
#                 converted_bns['{}-{}-{}'.format(i, j, 1)] = bottleneck.bn2
#                 if res50:
#                     bottleneck.bn3 = convert_norm(bottleneck.bn3, norm, bottleneck.bn3.num_features)
#                     converted_bns['{}-{}-{}'.format(i, j, 2)] = bottleneck.bn2
#                 if bottleneck.downsample is not None:
#                     bottleneck.downsample[1] = convert_norm(bottleneck.downsample[1], norm, bottleneck.downsample[1].num_features, append)
#     return net, converted_bns


# def get_recon_loss(target, weights, label, num_classes):
#     N, C = target.size()
#     weights = weights.view(1, num_classes, C).repeat(N, 1, 1)
#     l = label.view(N, 1, 1).repeat(1, 1, C)
#     gen_w = weights.gather(1, l).view(N, C)
#     rec = gen_w
#     rec_loss = ((rec - target) ** 2).mean()
#     return rec_loss


# class L2Norm(nn.BatchNorm2d):
#     def __init__(self, in_ch):
#         super(L2Norm, self).__init__(in_ch)
#
#     def forward(self, x):
#         out = super(L2Norm, self).forward(x)
#         out = F.normalize(out, dim=1)
#         return out
#
#     def load_old_dict(self, old_norm):
#         old_dict = old_norm.state_dict()
#         self.load_state_dict(old_dict)


class Resnet(nn.Module):
    def __init__(self, num_classes, pretrained, args):
        super(Resnet, self).__init__()
        self.args = args
        self.num_classes = num_classes
        try:
            model = eval(args.model)
        except Exception as e:
            model = resnet_18
            # print('No model : {} found, using default {} instead'.format(args.model, model))
        self.resnet = model(num_classes, pretrained=pretrained, args=args)

    def freeze(self):
        print('?')
        self.train()
        # self.eval()
        # self.resnet.bn1.eval()
        self.resnet.layer1.eval() #
        self.resnet.layer2.eval() #
        self.resnet.layer3.eval() #
        self.resnet.layer4.eval() #

        # bs=128;  69.94 -> 70.02 -> 70.20 -> 71.26
        # origin, 4eval,  3   ,   2  ,  1   ,  0;
        # 66.28; 51.41, 65.41, 69.94, 62.61, 22.65 : sketch
        # 76.07; 75.47, 73.29, 75.17, 68.64, 28.20 : cartoon -> 77.60(1024)
        # 77.20; 77.05, 73.34, 75.83, 66.41, 26.56 : art     -> 77.69
        # 95.69; 95.93, 95.75, 94.97, 93.83, 33.53 : photo

    def load(self):
        domains = ['photo', 'art_painting', 'cartoon', 'sketch']
        i = self.args.exp_num[0]
        path = Path('../script/adaptation/resnet18/{}0'.format(domains[i]))
        deepall_state_dict = torch.load(str(path / 'models' / 'model_best.pt'), map_location='cpu')
        dic = {}
        for k in deepall_state_dict.keys():
            # if 'weight' in k or 'bias' in k or 'run' in k:
            if 'conv' in k or 'fc' in k:
                dic[k] = deepall_state_dict[k]
        ret = self.load_state_dict(dic, strict=False)
        print(ret)

    def net_forward(self, x):
        net = self.resnet
        x = net.conv1(x)
        x = net.bn1(x)
        x = net.relu(x)
        x = net.maxpool(x)

        l1 = net.layer1(x)
        l2 = net.layer2(l1)
        l3 = net.layer3(l2)
        l4 = net.layer4(l3)
        return x, l1, l2, l3, l4

    def cos(self, x, weight):
        x = F.normalize(x, dim=1)
        weight = F.normalize(weight, dim=1)
        return (x @ weight.t() * self.alpha)

    def forward(self, x, label, aug_x=None):
        if self.training:
            x = torch.cat([x, aug_x], 0)
            label = torch.cat([label, label], 0)
        l0, l1, l2, l3, l4 = self.net_forward(x)
        l4_ = l4.mean((2, 3))
        final_logits = self.resnet.fc(l4_)
        ret = {
            # 'top2': {'acc_type': 'top2', 'pred': final_logits, 'target': label},
            'out': [l0, l1, l2, l3, l4],
            'logits': final_logits
        }
        if label is not None:
            ret.update({'main': {'loss_type': 'ce', 'acc_type': 'acc', 'pred': final_logits, 'target': label},})
        return ret

    def get_lr(self, fc_weight):
        lrs = [
            (self.resnet.conv1, 1.0), (self.resnet.bn1, 1.0),
            (self.resnet.layer1, 1.0), (self.resnet.layer2, 1.0),
            (self.resnet.layer3, 1.0), (self.resnet.layer4, 1.0),
            (self.resnet.fc, fc_weight),
        ]
        return lrs


class BaseResnet(nn.Module):
    def __init__(self, num_classes, pretrained, args, model=None):
        super(BaseResnet, self).__init__()
        self.num_classes = num_classes
        self.in_ch = 512
        self.args = args
        try:
            model = args.model if model is None else model
            model = eval(model)
        except Exception as e:
            model = resnet_18
            # print('No model : {} found, using default {} instead'.format(args.model, model))
        self.resnet = model(num_classes, pretrained=pretrained, args=args)
        self.fc = self.resnet.fc

    def load_pretrained(self, path):
        from utils.maps import DomainMap
        cur_domain = DomainMap.get(self.args.dataset)[self.args.exp_num[0]]
        path = path + '/{}0/models/model_best.pt'
        state = torch.load(path.format(cur_domain), map_location='cpu')
        ret = self.load_state_dict(state, strict=False)
        print(ret)

    def net_forward(self, x, net=None):
        x_ = x
        if net is None:
            net = self.resnet
        x = net.conv1(x)
        x = net.bn1(x)
        x = net.relu(x)
        x = net.maxpool(x)

        l1 = net.layer1(x)
        l2 = net.layer2(l1)
        l3 = net.layer3(l2)
        l4 = net.layer4(l3)
        return x, l1, l2, l3, l4

    def forward(self, x, label, aug_x, *args):
        l0, l1, l2, l3, l4 = self.net_forward(x)
        l4 = l4.mean((2, 3))
        final_logits = self.resnet.fc(l4)

        ret = {'main': {'loss_type': 'ce', 'acc_type': 'acc', 'pred': final_logits, 'target': label,},
               'out' : [final_logits],
               'logits': final_logits
               }
        return ret

    def get_lr(self, fc_weight):
        lrs = [
            (self.resnet.conv1, 1.0), (self.resnet.bn1, 1.0),
            (self.resnet.layer1, 1.0), (self.resnet.layer2, 1.0),
            (self.resnet.layer3, 1.0), (self.resnet.layer4, 1.0),
            (self.resnet.fc, fc_weight),
        ]
        return lrs


class JigResnet(nn.Module):
    def __init__(self, num_classes, pretrained, args):
        super(JigResnet, self).__init__()
        self.args = args
        self.resnet = resnet_18(num_classes, pretrained, args)
        self.jig_classifier = nn.Linear(self.resnet.fc.in_features, 32, bias=False)

    def net_forward(self, net, x):
        x = net.conv1(x)
        x = net.bn1(x)
        x = net.relu(x)
        x = net.maxpool(x)

        l1 = net.layer1(x)
        l2 = net.layer2(l1)
        l3 = net.layer3(l2)
        l4 = net.layer4(l3)
        return x, l1, l2, l3, l4

    def forward(self, x, label, jig_x, jig_label, *args):
        if self.training:
            x = jig_x

        l4 = self.net_forward(self.resnet, x)[-1]
        l4 = l4.mean((2, 3))
        class_logits = self.resnet.fc(l4)
        jigsaw_logits = self.jig_classifier(l4)
        return [(class_logits, label, 1), (jigsaw_logits, jig_label, self.args.jig_weight)]

    def get_lr(self, fc_weight):
        lrs = [
            (self.resnet.conv1, 1.0), (self.resnet.bn1, 1.0),
            (self.resnet.layer1, 1.0), (self.resnet.layer2, 1.0),
            (self.resnet.layer3, 1.0), (self.resnet.layer4, 1.0),
            (self.resnet.fc, fc_weight), (self.jig_classifier, fc_weight)
        ]
        return lrs


class MixBatchX(nn.BatchNorm2d):
    def __init__(self, in_ch):
        super(MixBatchX, self).__init__(in_ch, affine=True)
        from collections import deque
        self.saved_batch_buffer = deque(maxlen=2)

    def forward(self, x):
        if self.training:
            N, C, H, W = x.size()
            mean = x.mean((0, 2, 3))
            var = x.var((0, 2, 3))
            invstd = 1 / (var + 1e-5).sqrt()

            self.saved_batch_buffer.append((mean.detach(), var.detach()))
            saved_mean, saved_var = self.saved_batch_buffer[0]
            saved_std = (saved_var + 1e-5).sqrt()
            recalibrated_x = ((x - mean) * invstd) * saved_std + saved_mean

        return super(MixBatchX, self).forward(recalibrated_x)

    def load_old_dict(self, old_norm):
        old_dict = old_norm.state_dict()
        ret = super(MixBatchX, self).load_state_dict(old_dict, strict=False)


class MixBatch(nn.Module):
    def __init__(self):
        super(MixBatch, self).__init__()
        from collections import deque
        self.queue = deque(maxlen=100)
        self.mu = 0
        self.var = 0
        self.using_history = False

    def forward(self, x):
        if self.training:
            N, C, H, W = x.size()
            mu = x.mean((2, 3), keepdims=True)
            var = x.var((2, 3), keepdims=True)
            normed_x = (x - mu) / (var + 1e-5).sqrt()

            if self.using_history:
                if len(self.queue) == 0 or len(x) == self.queue[0][0].size(0) == N:
                    self.queue.append((mu.detach(), var.detach()))

                # momentum = 0.9
                # if self.mu == 0 or len(self.mu) == N:
                #     self.mu = self.mu * momentum + (1-momentum)*mu
                #     self.var = self.var * momentum + (1-momentum)*var
                #     mu, var = self.mu, self.var

                i = torch.randint(0, len(self.queue), (1,)).item()
                mu, var = self.queue[i]

            # bn_mu = x.mean((0, 2, 3), keepdims=True)
            # res_mu = mu - bn_mu
            # bn_var = x.var((0, 2, 3), keepdims=True)
            rand_idx = torch.randperm(N)
            # rand_idx2 = torch.randperm(N)
            perm_mu = mu[rand_idx] #+ res_mu[rand_idx2]
            perm_var = var[rand_idx]

            stylized_x = normed_x * (perm_var + 1e-5).sqrt() + perm_mu
            return stylized_x
        else:
            return x


class MixBatchResnet(Resnet):
    def __init__(self, *args, **kwargs):
        super(MixBatchResnet, self).__init__(*args, **kwargs)
        self.mix_batch_layer = MixBatch()

    def forward(self, x, label, *args):
        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        x = self.resnet.maxpool(x)

        l1 = self.resnet.layer1(x)

        # l1 = self.mix_batch_layer(l1)

        l2 = self.resnet.layer2(l1)

        # l2 = self.mix_batch_layer(l2)

        l3 = self.resnet.layer3(l2)

        l3 = self.mix_batch_layer(l3)

        l4 = self.resnet.layer4(l3)
        feats = l4.mean((2, 3))
        logits = self.resnet.fc(feats)

        ret = {'main': {
            'loss_type': 'ce',
            'acc_type': 'acc',
            'pred': logits,
            'target': label,
        }}
        return ret


class ConvertedResnet(nn.Module):
    def __init__(self, bn='torch', *args, **kwargs):
        super(ConvertedResnet, self).__init__()
        self.resnet = Resnet(*args, **kwargs)
        from model.in_model import convert_to_target
        if bn == 'torch':
            bn = nn.BatchNorm2d
        elif bn == 'adabn':
            pass
        convert_to_target(self.resnet, bn, start=0, end=5, verbose=True, res50=False)

    def forward(self, *args):
        return self.resnet(*args)
