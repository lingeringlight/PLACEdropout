from torch.utils import model_zoo
from torchvision.models.resnet import BasicBlock, model_urls, Bottleneck
import torch
from torch import nn as nn
import numpy as np
import torch.nn.functional as F
import random
from.SwapStyle import SwapStyle
from .DropBlock import DropBlock2D



class ResNet(nn.Module):
    def __init__(self, block, layers, device, jigsaw_classes=1000, classes=100, domains=3,
                 stage2_layers=None,
                 random_dropout_layers_flag=0,

                 baseline_dropout_flag=0,
                 baseline_dropout_p=0.,
                 baseline_progressive_flag=0,
                 dropout_mode=1,
                 velocity=4,
                 ChannelorSpatial=0,
                 spatialBlock=[],

                 dropout_recover_flag=0,
                 SwapStyle_flag=0,
                 SwapStyle_Layers=[],
                 SwapStyle_detach_flag=0,
                 ):
        self.inplanes = 64
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.class_classifier = nn.Linear(512 * block.expansion, classes)

        self.device = device

        self.stage2_layers = stage2_layers
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
        self.SwapStyle_flag = SwapStyle_flag
        self.SwapStyle_Layers = SwapStyle_Layers
        if SwapStyle_flag != 0:
            self.SwapStyle = SwapStyle(detach_flag=SwapStyle_detach_flag)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))
        return nn.Sequential(*layers)

    def is_patch_based(self):
        return False

    def forward(self, x, gt=None, epoch=None, mode=None, stage=None, layer_select=None):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        if mode == "train":
            dropout_layers = []
            if stage == 2:
                if self.random_dropout_layers_flag == 0:
                    dropout_layers = self.stage2_layers
                else:
                    # randomly select one
                    dropout_layers = [self.stage2_layers[layer_select]]
                dropout_epochs = [epoch, epoch, epoch]

            if self.SwapStyle_flag == 1 and 0 in self.SwapStyle_Layers:
                x = self.SwapStyle(x)
            for i, layer in enumerate([self.layer1, self.layer2, self.layer3, self.layer4]):
                x = layer(x)
                if self.SwapStyle_flag == 1 and i+1 in self.SwapStyle_Layers:
                    x = self.SwapStyle(x)
                if stage == 2:
                    if i + 1 in dropout_layers:
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
                                                            spatial_drop=self.baseline_dropout_p)
            x = self.avgpool(x)
            x = x.view(x.size(0), -1)
            y = self.class_classifier(x)
            return y
        else:
            for i, layer in enumerate([self.layer1, self.layer2, self.layer3, self.layer4]):
                x = layer(x)
            x = self.avgpool(x)
            x = x.view(x.size(0), -1)
            y = self.class_classifier(x)
            return y

    def random_dropout(self, x, channel_drop=0., spatial_drop=0.):
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
            x = self.random_dropout(x, channel_drop=channel_dropout, spatial_drop=channel_dropout)
        return x, channel_dropout


def resnet18(pretrained=True, **kwargs):
    """Constructs a ResNet-18 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet18']), strict=False)
    return model

def resnet50(pretrained=True, **kwargs):
    """Constructs a ResNet-50 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet50']), strict=False)
    return model
