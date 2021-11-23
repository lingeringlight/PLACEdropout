import argparse

import torch
import sys
sys.path.append("..")
from data import data_helper
from models import model_factory
import numpy as np
from models.resnet_Content_Style import resnet18, resnet50
import os
import random
import torch.nn.functional as F
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

import time

# os.environ['CUDA_VISIBLE_DEVICES'] = "7"

def get_args():
    parser = argparse.ArgumentParser(description="Script to launch jigsaw training",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--target", default=1, type=int, help="Target")
    parser.add_argument("--device", type=int, default=4, help="GPU num")
    parser.add_argument("--time", default=0, type=int, help="train time")

    parser.add_argument("--batch_size", "-b", type=int, default=128, help="Batch size")
    parser.add_argument("--image_size", type=int, default=224, help="Image size")

    parser.add_argument("--data", default="PACS")
    parser.add_argument("--data_root", default="/data/DataSets/")

    # data aug stuff
    parser.add_argument("--min_scale", default=0.8, type=float, help="Minimum scale percent")
    parser.add_argument("--max_scale", default=1.0, type=float, help="Maximum scale percent")
    parser.add_argument("--gray_flag", default=1, type=int, help="whether use random gray")
    parser.add_argument("--random_horiz_flip", default=0.5, type=float, help="Chance of random horizontal flip")
    parser.add_argument("--jitter", default=0.4, type=float, help="Color jitter amount")
    parser.add_argument("--tile_random_grayscale", default=0.1, type=float,
                        help="Chance of randomly greyscaling a tile")

    parser.add_argument("--limit_source", default=None, type=int,
                        help="If set, it will limit the number of training samples")
    parser.add_argument("--limit_target", default=None, type=int,
                        help="If set, it will limit the number of testing samples")

    parser.add_argument("--network", choices=model_factory.nets_map.keys(), help="Which network to use",
                        default="resnet18")
    parser.add_argument("--tf_logger", type=bool, default=False, help="If true will save tensorboard compatible logs")
    # parser.add_argument("--tf_logger", type=bool, default=False, help="If true will save tensorboard compatible logs")
    parser.add_argument("--folder_name", default='test', help="Used by the logger to save logs")
    parser.add_argument("--bias_whole_image", default=0.9, type=float,
                        help="If set, will bias the training procedure to show more often the whole image")
    parser.add_argument("--TTA", type=bool, default=False, help="Activate test time data augmentation")
    parser.add_argument("--classify_only_sane", default=False, type=bool,
                        help="If true, the network will only try to classify the non scrambled images")
    parser.add_argument("--train_all", default=True, type=bool, help="If true, all network weights will be trained")
    parser.add_argument("--suffix", default="", help="Suffix for the logger")
    parser.add_argument("--nesterov", default=True, type=bool, help="Use nesterov")

    return parser.parse_args()


class Trainer:
    def __init__(self, args, device):
        self.args = args
        self.device = device
        if args.network == 'resnet18':
            model = resnet18(
                pretrained=True,
                classes=args.n_classes,
                device=device,
                jigsaw_classes=1000,
                domains=args.n_domains
            )

        self.model = model.to(device)

        # get val dataloader for each domain
        self.test_loaders = {}

        target_loader = data_helper.get_val_dataloader(args, patches=model.is_patch_based())
        self.test_loaders[args.target] = target_loader

        for domain in args.source:
            args.target = domain
            source_loader = data_helper.get_val_dataloader(args, patches=model.is_patch_based())
            self.test_loaders[args.target] = source_loader

        for name, loader in self.test_loaders.items():
            print(name + ": " + str(len(loader.dataset)))
        self.n_classes = args.n_classes


    def get_KL_CS(self):
        self.model.eval()
        with torch.no_grad():
            KL_all_3 = []
            KL_all_4 = []
            CS_all_3 = []
            CS_all_4 = []
            for phase, loader in self.test_loaders.items():
                KL_3, KL_4, CS_3, CS_4 = self.do_test(loader)
                KL_all_3.append(KL_3.cpu().numpy())
                KL_all_4.append(KL_4.cpu().numpy())
                CS_all_3.append(CS_3.cpu().numpy())
                CS_all_4.append(CS_4.cpu().numpy())
        return KL_all_3, KL_all_4, CS_all_3, CS_all_4

    def do_test(self, loader):
        KL_diversity_3 = []
        KL_diversity_4 = []
        channel_similarity_3 = []
        channel_similarity_4 = []
        for it, ((data, _, class_l, _), _) in enumerate(loader):
            data, class_l = data.to(self.device), class_l.to(self.device)
            # get feature
            feature_3, feature_4 = self.model.KLCS_forward(data)
            KL_3 = self.model.KL_divergence(feature_3)
            CS_3 = self.model.response_diversity(feature_3)
            KL_4 = self.model.KL_divergence(feature_4)
            CS_4 = self.model.response_diversity(feature_4)
            
            KL_diversity_3.append(KL_3)
            KL_diversity_4.append(KL_4)
            channel_similarity_3.append(CS_3)
            channel_similarity_4.append(CS_4)
            
        KL_diversity_3 = torch.stack(KL_diversity_3, dim=0)
        KL_diversity_4 = torch.stack(KL_diversity_4, dim=0)
        channel_similarity_3 = torch.stack(channel_similarity_3, dim=0)
        channel_similarity_4 = torch.stack(channel_similarity_4, dim=0)

        KL_final_3 = torch.mean(KL_diversity_3)
        KL_final_4 = torch.mean(KL_diversity_4)
        CS_final_3 = torch.mean(channel_similarity_3)
        CS_final_4 = torch.mean(channel_similarity_4)

        return KL_final_3, KL_final_4, CS_final_3, CS_final_4


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    # torch.backends.cudnn.deterministic = True

domain_map = {
    'PACS': ['photo', 'art_painting', 'cartoon', 'sketch'],
    'PACS_random_split': ['photo', 'art_painting', 'cartoon', 'sketch'],
    'OfficeHome': ['Art', 'Clipart', 'Product', 'RealWorld'],
    'VLCS': ["CALTECH", "LABELME", "PASCAL", "SUN"],
    'miniDomainNet': ['clipart', 'painting', 'real', 'sketch'],
    'digits_dg': ['mnist', 'mnist_m', 'svhn', 'syn'],
}

classes_map = {
    'PACS': 7,
    'PACS_random_split': 7,
    'OfficeHome': 65,
    'miniDomainNet': 126,
    'VLCS': 5,
    'digits_dg': 10,
}

domains_map = {
    'PACS': 3,
    'PACS_random_split': 3,
    'OfficeHome': 3,
    'miniDomainNet': 3,
    'VLCS': 3,
    'digits_dg': 3,
}

val_size_map = {
    'PACS': 0.1,
    'PACS_random_split': 0.1,
    'OfficeHome': 0.1,
    'miniDomainNet': 0.1,
    'VLCS': 0.3,
    'digits_dg': 0.2,
}

image_size_map = {
    'PACS': 224,
    'PACS_random_split': 224,
    'OfficeHome': 224,
    'miniDomainNet': 96,
    'VLCS': 224,
    'digits_dg': 224,
}

max_scale_map = {
    'PACS': 1.0,
    'PACS_random_split': 1.0,
    'OfficeHome': 1.0,
    'miniDomainNet': 1.25,
    'VLCS': 1.0,
    'digits_dg': 1.0,
}


def get_domain(name):
    if name not in domain_map:
        raise ValueError('Name of dataset unknown %s' %name)
    return domain_map[name]


def main():
    args = get_args()

    if "PACS" in args.data:
        args.data_root = os.path.join(args.data_root, "PACS")
    elif args.data == "digits_dg" or args.data == "miniDomainNet":
        args.data_root = "/data/gjt/DataSets/" + args.data
    else:
        args.data_root = os.path.join(args.data_root, args.data)

    args.n_classes = classes_map[args.data]
    args.n_domains = domains_map[args.data]
    args.val_size = val_size_map[args.data]
    args.image_size = image_size_map[args.data]
    args.max_scale = max_scale_map[args.data]

    setup_seed(args.time)
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.device)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    base_path = "/data/gjt/RSC-master/Content_Style/PACS/Deepall_RandAug_MixStyle/" + \
    "resnet18_stage1_lr0.001_batch128" + "/"

                # "resnet_layer_3_MixStyle3_random0.0_randaug_m4_n8_stage1_w10.0_w20.0_radio0.5_Conv1BNReLUDropout0.0_stage2_epochs_style30_learning_rate_style0.001_lr0.001_batch64" + "/"


    dropout_name = "dropout34_lastEpoch_random_one_layer_behind_first_layer_all0.33_progressive_velocity4_channel"
    # dropout_name = "dropout34_epoch15_lr0.001_lastEpoch_random_one_layer_behind_first_layer_all0.33_progressive"

    domain = ['photo', 'art_painting', 'cartoon', 'sketch']
    args.target = domain.pop(args.target)
    args.source = domain
    print("Target domain: {}".format(args.target))
    trainer = Trainer(args, device)
    for drop_flag in range(0, 3):
        seed = 1
        if drop_flag == 0:
            model_path = base_path + args.target + str(seed) + "/models/model_stage1.pt"
            print("Baseline")
        elif drop_flag == 1:
            model_path = "/data/gjt/RSC-master/RSC/" + args.target + str(0) + "/models/model_best.pt"
            print("RSC")
        else:
            model_path = base_path + args.target + str(seed) + "/" + dropout_name + "/models/model_best.pt"
            print("PLACE")

        # else:
        #     model_path = base_path + domain[domain_index] + str(seed) + "/models/model_best.pt"

        trainer.model.load_state_dict(torch.load(model_path, map_location='cuda'), strict=False)

        KL_all_3, KL_all_4, CS_all_3, CS_all_4 = trainer.get_KL_CS()
        print("KL divergence of Layer 3: train " + str(np.mean(KL_all_3[1:])) + "; target " + str(KL_all_3[0]))
        print("KL divergence of Layer 4: train " + str(np.mean(KL_all_4[1:])) + "; target " + str(KL_all_4[0]))
        print("Channel similarity of Layer 3: train " + str(np.mean(CS_all_3[1:])) + "; target " + str(CS_all_3[0]))
        print("Channel similarity of Layer 4: train " + str(np.mean(CS_all_4[1:])) + "; target " + str(CS_all_4[0]))


if __name__ == "__main__":
    torch.backends.cudnn.benchmark = True
    main()
