import torch.nn as nn
from torch.nn import functional as F
import numpy as np
import torch
import random
from .DropBlock import DropBlock2D


class Convolution(nn.Module):

    def __init__(self, c_in, c_out):
        super().__init__()
        self.conv = nn.Conv2d(c_in, c_out, 3, stride=1, padding=1)
        self.relu = nn.ReLU(True)

    def forward(self, x):
        return self.relu(self.conv(x))


class ConvNet(nn.Module):
    def __init__(self, device, c_hidden=64,
                 classes=10,
                 domains=3,
                 stage2_progressive_flag=0,
                 stage2_layers=None,
                 stage2_epochs_layer=10,
                 stage2_reverse_flag=0,
                 adjust_single_layer=0,
                 dropout_epoch_stop=0,
                 update_parameters_method=0,
                 random_dropout_layers_flag=0,

                 baseline_dropout_flag=0,
                 baseline_dropout_p=0.,
                 baseline_progressive_flag=0,
                 dropout_mode=1,
                 velocity=4,
                 ChannelorSpatial=0,
                 spatialBlock=[],

                 dropout_recover_flag=0,
                 MixStyle_flag=0,
                 MixStyle_p=0.,
                 MixStyle_Layers=[],
                 MixStyle_detach_flag=0,
                 ):
        super().__init__()

        self.device = device

        self.stage2_epochs_layer = stage2_epochs_layer
        self.stage2_reverse_flag = stage2_reverse_flag
        self.stage2_layers = stage2_layers
        self.dropout_epoch_stop = dropout_epoch_stop
        self.random_dropout_layers_flag = random_dropout_layers_flag
        self.baseline_dropout_flag = baseline_dropout_flag
        self.baseline_dropout_p = baseline_dropout_p
        self.baseline_progressive_flag = baseline_progressive_flag
        self.dropout_mode = dropout_mode
        self.velocity = velocity
        self.ChannelorSpatial = ChannelorSpatial
        self.spatialBlock = spatialBlock
        self.drop_block = DropBlock2D()
        self.dropout_recover_flag = dropout_recover_flag
        self.MixStyle_flag = MixStyle_flag
        self.MixStyle_p = MixStyle_p
        self.MixStyle_Layers = MixStyle_Layers
        if MixStyle_flag != 0:
            self.MixStyle = MixStyle(p=MixStyle_p, mix='random', style_detach=MixStyle_detach_flag)

        self.conv1 = Convolution(3, c_hidden)
        self.conv2 = Convolution(c_hidden, c_hidden)
        self.conv3 = Convolution(c_hidden, c_hidden)
        self.conv4 = Convolution(c_hidden, c_hidden)
        self.fc = nn.Linear(2**2 * c_hidden, classes)

        self._out_features = 2**2 * c_hidden

    def is_patch_based(self):
        return False

    def _check_input(self, x):
        H, W = x.shape[2:]
        assert (
            H == 32 and W == 32
        ), "Input to network must be 32x32, " "but got {}x{}".format(H, W)

    def forward(self, x, gt=None, epoch=None, mode=None, stage=None, layer_select=None):

        # self._check_input(x)

        if mode == "train":
            dropout_layers = []
            if stage == 2:
                if self.random_dropout_layers_flag != 0:
                    if self.random_dropout_layers_flag == 1:
                        # randomly select one
                        dropout_layers = [self.stage2_layers[layer_select]]
                        dropout_epochs = [epoch, epoch, epoch]
                    elif self.random_dropout_layers_flag == 2:
                        dropout_layers = self.stage2_layers
                        dropout_epochs = [epoch, epoch, epoch]
                    else:
                        # Bernoulli
                        pass

            if self.MixStyle_flag == 1 and 0 in self.MixStyle_Layers:
                x = self.MixStyle(x)
            for i, conv in enumerate([self.conv1, self.conv2, self.conv3, self.conv4]):
                x = conv(x)
                x = F.max_pool2d(x, 2)
                if self.MixStyle_flag == 1 and i + 1 in self.MixStyle_Layers:
                    x = self.MixStyle(x)
                if i + 1 in dropout_layers:
                    if stage == 2:
                        layer_index_in_dropout_layers = dropout_layers.index(i + 1)
                        layer_dropout_epoch = dropout_epochs[layer_index_in_dropout_layers]
                        if self.baseline_dropout_flag:
                            if self.baseline_progressive_flag:
                                x, _ = self.progressive_dropout_metric(x, layer_dropout_epoch, pmax=self.baseline_dropout_p, velocity=self.velocity)
                            else:
                                if self.dropout_mode == 0:
                                    x = nn.functional.dropout(x, p=self.baseline_dropout_p, training=self.training)
                                else:
                                    x = self.random_dropout(x, channel_drop=self.baseline_dropout_p,
                                                            spatial_drop=self.baseline_dropout_p, current_epoch=layer_dropout_epoch)
                    else:
                        pass

            x = x.view(x.size(0), -1)
            y = self.fc(x)
            return y
        else:
            channel_diversity_layers = [0., 0., 0., 0.]
            KL_divergence_layers = [0., 0., 0., 0.]

            x = self.conv1(x)
            x = F.max_pool2d(x, 2)
            x = self.conv2(x)
            x = F.max_pool2d(x, 2)
            x = self.conv3(x)
            x = F.max_pool2d(x, 2)
            x = self.conv4(x)
            x = F.max_pool2d(x, 2)

            x = x.view(x.size(0), -1)
            y = self.fc(x)
            return y, channel_diversity_layers, KL_divergence_layers

    def random_dropout(self, x, sample_drop=0., channel_drop=0., spatial_drop=0., current_epoch=0):
        batch_size = x.shape[0]
        channel_num = x.shape[1]
        H = x.shape[2]
        W = x.shape[3]

        if self.ChannelorSpatial == 0:
            channel_or_spatial = 0
        elif self.ChannelorSpatial == 1:
            channel_or_spatial = 1
        else:
            channel_or_spatial = np.random.randint(2, size=1)[0]

        if channel_or_spatial == 0:
            dropout_channel_num = int(channel_num * channel_drop)
            # smooth: same or different mask for each samples
            mask = torch.ones_like(x).to(self.device)
            for i in range(batch_size):
                zero_channel_index = random.sample(range(0, channel_num), dropout_channel_num)
                mask[i][zero_channel_index] = torch.zeros([H, W]).to(self.device)
            x = x * mask
            if self.dropout_recover_flag:
                x = x * 1 / (1 - channel_drop)
            return x
        else:
            block_size = random.choice(self.spatialBlock)
            x = self.drop_block(input=x, keep_prob=1 - spatial_drop, block_size=block_size)
            return x

    def progressive_dropout_metric(self, x, current_epoch, pmax, velocity=4):
        channel_dropout = pmax * 2 / np.pi * np.arctan(current_epoch/velocity)

        if self.dropout_mode == 0:
            x = nn.functional.dropout(x, p=channel_dropout, training=self.training)
        else:
            x = self.random_dropout(x, channel_drop=channel_dropout, spatial_drop=channel_dropout,
                                current_epoch=current_epoch)
        return x, channel_dropout


def init_network_weights(model, init_type="normal", gain=0.02):
    def _init_func(m):
        classname = m.__class__.__name__

        if hasattr(m, "weight") and (
            classname.find("Conv") != -1 or classname.find("Linear") != -1
        ):
            if init_type == "normal":
                nn.init.normal_(m.weight.data, 0.0, gain)
            elif init_type == "xavier":
                nn.init.xavier_normal_(m.weight.data, gain=gain)
            elif init_type == "kaiming":
                nn.init.kaiming_normal_(m.weight.data, a=0, mode="fan_in")
            elif init_type == "orthogonal":
                nn.init.orthogonal_(m.weight.data, gain=gain)
            else:
                raise NotImplementedError(
                    "initialization method {} is not implemented".
                    format(init_type)
                )
            if hasattr(m, "bias") and m.bias is not None:
                nn.init.constant_(m.bias.data, 0.0)
        elif classname.find("BatchNorm") != -1:
            nn.init.constant_(m.weight.data, 1.0)
            nn.init.constant_(m.bias.data, 0.0)
        elif classname.find("InstanceNorm") != -1:
            if m.weight is not None and m.bias is not None:
                nn.init.constant_(m.weight.data, 1.0)
                nn.init.constant_(m.bias.data, 0.0)

    model.apply(_init_func)

def cnn_digitsdg(**kwargs):
    """
    This architecture was used for DigitsDG dataset in:
        - Zhou et al. Deep Domain-Adversarial Image Generation
        for Domain Generalisation. AAAI 2020.
    """
    model = ConvNet(c_hidden=64, **kwargs)
    init_network_weights(model, init_type="kaiming")
    return model