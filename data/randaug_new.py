import random

from torchvision.transforms import transforms

from randaug import randaug
import torchvision.transforms.functional as F
import numpy as np


def Shear(img, v):
    if random.random() > 0.5:
        return randaug.ShearX(img, v)
    else:
        return randaug.ShearY(img, v)


def Translate(img, v):
    if random.random() > 0.5:
        return randaug.TranslateXabs(img, v)
    else:
        return randaug.TranslateYabs(img, v)


def Gray(img, _):
    from torchvision.transforms import RandomGrayscale
    op = RandomGrayscale(0.2)
    return op(img)


def augment_list():  # 16 oeprations and their ranges
    # https://github.com/tensorflow/tpu/blob/8462d083dd89489a79e3200bcc8d4063bf362186/models/official/efficientnet/autoaugment.py#L505
    l = [
        # Fixed
        (randaug.Identity, 0., 1.0, [0, 30]),

        (randaug.Equalize, 0, 1, [0, 30]),  # 0
        (randaug.Invert, 0, 1, [0, 30]),  # 1
        (randaug.AutoContrast, 0, 1, [0, 30]),  # 5
        (Gray, 0, 1, [0, 30]),  # 5

        # random
        (randaug.Posterize, 0, 2, [0, 30]),  # 2
        (randaug.Solarize, 0, 256, [0, 30]),  # 3
        (randaug.SolarizeAdd, 0, 110, [0, 30]),  # 4
        (randaug.Color, 0.1, 1.9, [0, 30]),  # 7
        (randaug.Sharpness, 0.1, 1.9, [0, 30]),  # 9  TODO change.
        #
        (randaug.Contrast, 0.7, 1.9, [0, 30]),  # 6
        (randaug.Brightness, 0.7, 1.5, [3, 29]),  # 8
        # (randaug.CutoutAbs, 0, 40., [22, 23]),  # 12
        (Shear, 0., 0.3, [4, 11]),  # 10
        # (randaug.Rotate, 0, 10, [0, 30]),  # 15
        # (Translate, 0., 100, [3, 12]),  # 13
    ]

    return l


class MyRandAugment:
    def __init__(self, args):
        # 只在训练的时候增广。 N 表示 sample 次数，次数越多，性能越高
        self.n = 1
        self.m = 4
        self.augment_list = augment_list()

        self.pre_transform = transforms.Compose(
                [transforms.RandomResizedCrop(args.img_size, scale=(args.min_scale, 1.0)),
                 transforms.RandomHorizontalFlip(), ]
            )
        self.post_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

    def __call__(self, img):
        img = self.pre_transform(img)
        ops = np.random.choice(len(self.augment_list), self.n, replace=True)
        for idx in ops:
            op, minval, maxval, randrange = self.augment_list[idx]
            val = (float(self.m) / 30) * float(maxval - minval) + minval
            img = op(img, val)
        img = self.post_transform(img)
        return img
