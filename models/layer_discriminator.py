import torch.nn as nn
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo
import torch


class GradReverse(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, lambd, reverse=True):
        ctx.lambd = lambd
        ctx.reverse = reverse
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        if ctx.reverse:
            return (grad_output * -ctx.lambd), None, None
        else:
            return (grad_output * ctx.lambd), None, None


def grad_reverse(x, lambd=1.0, reverse=True):
    return GradReverse.apply(x, lambd, reverse)


class Discriminator(nn.Module):
    def __init__(self, classes, channel_num, structure, dropP=0.5, grl=True, LayerBlock=None):
        super(Discriminator, self).__init__()
        self.grl = grl
        self.structure = structure
        # pooling之后
        # self.model = nn.Sequential(
        #     nn.Linear(dims[0], dims[1]),
        #     nn.ReLU(),
        #     nn.Dropout(0.5),
        #     nn.Linear(dims[1], dims[2]),
        #     nn.ReLU(),
        #     nn.Dropout(0.5),
        #     nn.Linear(dims[2], dims[3]),
        # )
        if structure == 1:
            # 1x1Conv + BN + ReLU + AvgPooling
            self.model = nn.Sequential(
                nn.Conv2d(channel_num, 512, kernel_size=1, stride=2, bias=False),
                nn.BatchNorm2d(512),
                nn.ReLU(inplace=True)
            )
            self.avgPool = nn.AdaptiveAvgPool2d(1)
            self.dropout = nn.Dropout(dropP)
            self.classifier = nn.Linear(512, classes)
        elif structure == 2:
            # 3x3Conv + BN + ReLU + AvgPooling
            self.model = nn.Sequential(
                nn.Conv2d(channel_num, 512, kernel_size=3, stride=2, bias=False),
                nn.BatchNorm2d(512),
                nn.ReLU(inplace=True),
            )
            self.avgPool = nn.AdaptiveAvgPool2d(1)
            self.dropout = nn.Dropout(dropP)
            self.classifier = nn.Linear(512, classes)
        else:
            # resnet18 layer4
            self.model = LayerBlock
            self.avgPool = nn.AvgPool2d(7, stride=1)
            self.dropout = nn.Dropout(dropP)
            self.classifier = nn.Linear(512, classes)

        self.lambd = 1.0

    def set_lambd(self, lambd):
        self.lambd = lambd

    def forward(self, x, reverse):
        if self.grl:
            x = grad_reverse(x, self.lambd, reverse)

        x = self.model(x)
        x = self.avgPool(x)
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        y = self.classifier(x)

        return y


