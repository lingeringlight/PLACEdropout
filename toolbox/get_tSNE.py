import argparse

import torch
import sys
from tSNE import tSNE
sys.path.append("..")
from data import data_helper
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
    parser.add_argument("--target", default=3, type=int, help="Target")
    parser.add_argument("--device", type=int, default=3, help="GPU num")
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

    parser.add_argument("--network", help="Which network to use",
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

        target_loader = data_helper.get_val_dataloader(args, patches=model.is_patch_based(), tSNE_flag=0)
        self.test_loaders[args.target] = target_loader

        for domain in args.source:
            args.target = domain
            source_loader = data_helper.get_val_dataloader(args, patches=model.is_patch_based(), tSNE_flag=1)
            self.test_loaders[args.target] = source_loader

        for name, loader in self.test_loaders.items():
            print(name + ": " + str(len(loader.dataset)))
        self.n_classes = args.n_classes

        # self.domain_index = {'photo': 0, 'art_painting': 1, 'cartoon': 2, 'sketch': 3}

    def get_features(self):
        self.model.eval()
        with torch.no_grad():
            features_all = []
            class_all = []
            domain_all = []
            domain_num = [3,0,1,2]
            count = 0
            for phase, loader in self.test_loaders.items():
                features, class_l = self.do_test(loader)
                # domain_l = [self.domain_index[phase] for i in range(features.shape[0])]
                domain_l = [domain_num[count] for i in range(features.shape[0])]
                features_all.append(features)
                class_all.append(class_l)
                domain_all.extend(domain_l)

            # features_all = torch.cat(features_all, dim=0)
            # class_all = torch.cat(class_all, dim=0)
            # domain_all = torch.cat(domain_all, dim=0)
        return features_all, class_all, domain_all

    def do_test(self, loader):
        features_list = []
        class_l_list = []

        for it, ((data, _, class_l, _), _) in enumerate(loader):
            data, class_l = data.to(self.device), class_l.to(self.device)
            # get feature
            features = self.model.feature_forward(data)
            features = F.normalize(features, dim=1)  # N x B -> N x B
            features_list.append(features)
            class_l_list.append(class_l)

        features = torch.cat(features_list, dim=0)
        class_l = torch.cat(class_l_list, dim=0)
        return features, class_l

    def plot_test_train(self, features_all, class_all, domain_all, target_domain, class_name, title_pic, mode=0):
        # tsne = TSNE(n_components=2, init='pca', random_state=0, perplexity=50)
        tsne = tSNE()
        features_all = torch.cat(features_all, dim=0)
        features_tsne = tsne.get_tsne_result(features_all.cpu().numpy())
        class_tsne = torch.cat(class_all).cpu().numpy()
        domain_tsne = domain_all

        tsne.plot_tsne(features_tsne, class_tsne, domain_labels=domain_tsne, save_name=title_pic + ".jpg", label_name=class_name)


        # x_feats, y_feats = features_tsne[:, 0]*2, features_tsne[:, 1]*2
        #
        # color = ['c', 'b', 'g', 'r', 'm', 'y', 'k']
        # shape = ['o', '^', 's', '*']
        #
        # plt.style.use('seaborn-whitegrid')
        # plt.cla()
        # count = 0
        # print("begin!")
        #
        # if mode == 0:
        #     # domain
        #     for i, domain in enumerate(domain_tsne):
        #         num = len(domain)
        #         x = x_feats[count: count+num]
        #         y = y_feats[count: count+num]
        #         domain_name = domain[0]
        #         if domain_name == target_domain:
        #             domain_name += "(target)"
        #         plt.scatter(x, y, marker="o", c=color[i], s=1, label=domain_name)
        #         count += num
        #         # print(domain[0] + " OK.")
        # elif mode == 1:
        #     for label in range(0, 7):
        #         index = np.where(class_tsne == label)
        #         x = x_feats[index]
        #         y = y_feats[index]
        #         # plt.scatter(x, y, marker="o", c=color[label], s=1, label=class_name[label])
        #         scatter = plt.scatter(x, y, marker="o", c=classes, cmap='plasma', s=1)
        #         # plt.scatter(x, y, marker="o", c=class_tsne, cmap='plasma', s=1, label=class_name[label])
        # else:
        #     scatters = []
        #     # print(domain_tsne)
        #     for i, domain in enumerate(domain_tsne):
        #         num = len(domain)
        #         x = x_feats[count: count+num]
        #         y = y_feats[count: count+num]
        #         classes = class_tsne[count: count+num]
        #
        #         scatter = plt.scatter(x, y, marker=shape[i], c=classes, cmap='plasma', s=3)
        #         scatters.append(scatter)
        #         a, b = scatter.legend_elements()
        #         legend = plt.legend(a, class_name, loc="upper right")
        #         plt.gca().add_artist(legend)
        #         count += num
        #     domain_names = ['photo', 'art_painting', 'cartoon', 'sketch']
        #     domain_show = ['Photo', 'Art', 'Cartoon', 'Sketch']
        #     for i, name in enumerate(domain_names):
        #         if name == target_domain:
        #             domain_show[i] += "(target)"
        #
        #     legend1 = plt.legend(scatters, domain_show, loc="upper left")
        #     plt.gca().add_artist(legend1)
        # plt.axis('off')
        # plt.xlim(min(x_feats)-10, max(x_feats)+15)
        # plt.ylim(min(y_feats)-10, max(y_feats)+30)
        # plt.savefig(title_pic + ".jpg", dpi=500)
        # # plt.show()


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
    dropout_name = "dropout34_lastEpoch_random_one_layer_behind_first_layer_all0.33_progressive_velocity4_channel"

    domain = ['photo', 'art_painting', 'cartoon', 'sketch']
    target_domain = domain[args.target]
    args.target = domain.pop(args.target)
    args.source = domain
    print("Target domain: {}".format(args.target))
    trainer = Trainer(args, device)
    for drop_flag in range(0, 2):
        for mode in range(0, 2):
            domain_index = 1
            seed = 0
            title_pic = target_domain
            if drop_flag == 1:
                model_path = base_path + domain[domain_index] + str(seed) + "/" + dropout_name + "/models/model_best.pt"
                title_pic += "_PLCD"
            else:
                model_path = base_path + domain[domain_index] + str(seed) + "/models/model_best.pt"
                title_pic += "_Deepall"

            if mode == 0:
                title_pic += "_domain"
            elif mode == 1:
                title_pic += "_class"
            else:
                title_pic += "_both"

            trainer.model.load_state_dict(torch.load(model_path, map_location='cuda'), strict=False)
            class_name = ["dog", "elephant", "giraffe", "guitar", "horse", "house", "person"]

            time1 = time.time()
            features_all, class_all, domain_all = trainer.get_features()
            time2 = time.time()
            print("Time for getting feature is " + str(format(time2 - time1, '.0f')) + "s")
            time1 = time.time()
            trainer.plot_test_train(features_all, class_all, domain_all, target_domain=target_domain,
                                    class_name=class_name, title_pic=title_pic, mode=mode)
            time2 = time.time()
            print("Time for drawing t-SNE is " + str(format(time2 - time1, '.0f')) + "s")


if __name__ == "__main__":
    torch.backends.cudnn.benchmark = True
    main()
