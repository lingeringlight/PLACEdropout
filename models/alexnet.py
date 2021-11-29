import torch
import torch.nn as nn
import torch.utils.model_zoo as model_zoo
from torchvision.models import AlexNet

__all__ = ['AlexNet', 'alexnet']

model_urls = {
    'alexnet': 'https://download.pytorch.org/models/alexnet-owt-4df8aa71.pth',
}


class DGalexnet(nn.Module):
    def __init__(self, num_classes=1000):
        super(DGalexnet, self).__init__()

        self.layer0 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.layer1 = nn.Sequential(
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.feature_layers = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
        )
        self.fc = nn.Linear(4096, num_classes)

        nn.init.xavier_uniform_(self.fc.weight, .1)
        nn.init.constant_(self.fc.bias, 0.)

    def forward(self, x, gt=None, epoch=None, mode=None, stage=None, layer_select=None):

        x = self.layer0(x)
        x = self.layer1(x)
        x = self.layer2(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)

        x = self.feature_layers(x)
        x = self.fc(x)
        return x

def alexnet(classes, pretrained=True):
    """AlexNet model architecture from the
    `"One weird trick..." <https://arxiv.org/abs/1404.5997>`_ paper.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = DGalexnet(classes)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['alexnet']))
        print('Load pre trained model')
    return model