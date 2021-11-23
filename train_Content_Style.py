import argparse

import torch
from torch import nn
from data import data_helper
from models import model_factory
from optimizer.optimizer_helper import get_optim_and_scheduler, get_optim_and_scheduler_style, get_optim_and_scheduler_scatter
from utils.Logger import Logger
import numpy as np
from models.resnet_Content_Style import resnet18, resnet50
from models.alexnet import alexnet
from models.convnet import cnn_digitsdg
import os
import random
import time
from CrossEntropyLoss import CrossEntropyLoss

# os.environ['CUDA_VISIBLE_DEVICES'] = "7"

def get_args():
    parser = argparse.ArgumentParser(description="Script to launch jigsaw training",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--target", default=1, type=int, help="Target")
    parser.add_argument("--device", type=int, default=3, help="GPU num")
    parser.add_argument("--time", default=1, type=int, help="train time")

    parser.add_argument("--batch_size", "-b", type=int, default=128, help="Batch size")
    parser.add_argument("--image_size", type=int, default=224, help="Image size")

    parser.add_argument("--result_path", default="", help="")

    parser.add_argument("--data", default="PACS")
    parser.add_argument("--data_root", default="/data/DataSets/")

    parser.add_argument("--MixStyle_flag", default=0, type=int)
    parser.add_argument("--MixStyle_detach_flag", default=1, type=int)
    parser.add_argument("--MixStyle_p", default=0.5, type=float)
    parser.add_argument("--MixStyle_Layers", default=[3], nargs="+", type=int)
    parser.add_argument("--MixStyle_mix_flag", default=0, type=int)

    parser.add_argument("--train_diversity_flag", default=0, type=int)
    parser.add_argument("--test_diversity_flag", default=0, type=int)

    parser.add_argument("--RandAug_flag", default=0, type=int)
    parser.add_argument("--m", default=4, type=int)
    parser.add_argument("--n", default=8, type=int)

    parser.add_argument("--label_smooth", default=0, type=int, help="whether use label smooth loss")

    parser.add_argument("--stage1_flag", default=1, type=int, help="")

    parser.add_argument("--stage2_flag", default=0, type=int, help="")
    parser.add_argument("--stage2_one_stage_flag", default=0, type=int, help="whether use one-stage scheme.")
    parser.add_argument("--stage2_stage1_LastOrBest", default=0, type=int, help="0: use the model of last epoch; "
                                                                                "1: use the model of best epoch.")
    parser.add_argument("--stage2_progressive_flag", default=1, type=int, help="Add PCD into Layer progressively")
    parser.add_argument("--stage2_reverse_flag", default=0, type=int, help="whether reverse the order of each layer")
    parser.add_argument("--stage2_layers", default=[3, 4], nargs="+", type=int, help="Layers of PCD module")
    parser.add_argument("--epochs_layer", type=int, default=15, help="Number of epochs for layer")
    parser.add_argument("--adjust_single_layer", type=int, default=0, help="Adjust single layer or all of the following layers")
    parser.add_argument("--dropout_epoch_stop", type=int, default=0, help="For the last stage, dropout P stop or not")
    parser.add_argument("--update_parameters_method", type=int, default=0, help="0: the module behind the first dropout layer;"
                                                                                "1: the whole network;"
                                                                                "2: the module behind the selected layer;"
                                                                                "3: the layer behind the selected layer;")
    parser.add_argument("--random_dropout_layers_flag", default=1, type=int, help="whether to use random layer dropout;"
                                                                                  "1: randomly select one layer;"
                                                                                  "2: all layer;"
                                                                                  "3: Bernoulli Distribution.")
    parser.add_argument("--not_before_flag", type=int, default=0, help="whether include the layer dropout")

    # stage 1
    parser.add_argument("--learning_rate", "-l", type=float, default=.01, help="Learning rate")
    parser.add_argument("--epochs", "-e", type=int, default=30, help="Number of epochs")

    # stage 2
    parser.add_argument("--epochs_style", type=int, default=30, help="Number of epochs")
    parser.add_argument("--learning_rate_style", type=float, default=.001, help="Learning rate")

    parser.add_argument("--method_index", type=int, default=0, help="0: RandAug+MixStyle; 1: RandAug")

    parser.add_argument("--baseline_dropout_flag", type=int, default=1, help="Whether dropout for all channels")
    parser.add_argument("--baseline_dropout_p", default=0.33, type=float, help="Dropout p for All channels")
    parser.add_argument("--baseline_progressive_flag", default=1, type=int,
                        help="whether use progressive dropout for All channels")
    parser.add_argument("--dropout_mode", type=int, default=1, help="0: normal dropout; 1: channel dropout")
    parser.add_argument("--velocity", default=4, type=int, help="Progressive velocity")
    parser.add_argument("--ChannelorSpatial", default=0, type=int, help="0: channel; 1: spatial; 2: channel or spatial")
    parser.add_argument("--spatialBlock", default=[3, 5], nargs="+", type=int, help="block size for Spatial dropout")

    parser.add_argument("--dropout_recover_flag", default=1, type=int,
                        help="whether to recover the scale of feature with mask")
    parser.add_argument("--test_dropout_flag", default=0, type=int, help="whether to dropout in test time")

    # data aug stuff
    parser.add_argument("--min_scale", default=0.8, type=float, help="Minimum scale percent")
    parser.add_argument("--max_scale", default=1.0, type=float, help="Maximum scale percent")
    parser.add_argument("--gray_flag", default=0, type=int, help="whether use random gray")
    parser.add_argument("--random_horiz_flip", default=0.5, type=float, help="Chance of random horizontal flip")
    parser.add_argument("--jitter", default=0.4, type=float, help="Color jitter amount")
    parser.add_argument("--tile_random_grayscale", default=0.1, type=float,
                        help="Chance of randomly greyscaling a tile")
    #
    parser.add_argument("--limit_source", default=None, type=int,
                        help="If set, it will limit the number of training samples")
    parser.add_argument("--limit_target", default=None, type=int,
                        help="If set, it will limit the number of testing samples")
    parser.add_argument("--network", choices=model_factory.nets_map.keys(), help="Which network to use",
                        default="convnet")
    parser.add_argument("--tf_logger", type=bool, default=False, help="If true will save tensorboard compatible logs")
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
                domains=args.n_domains,

                stage2_progressive_flag=args.stage2_progressive_flag,
                stage2_reverse_flag=args.stage2_reverse_flag,
                stage2_layers=args.stage2_layers,
                stage2_epochs_layer=args.epochs_layer,
                adjust_single_layer=args.adjust_single_layer,
                dropout_epoch_stop=args.dropout_epoch_stop,
                update_parameters_method=args.update_parameters_method,
                random_dropout_layers_flag=args.random_dropout_layers_flag,

                baseline_dropout_flag=args.baseline_dropout_flag,
                baseline_dropout_p=args.baseline_dropout_p,
                baseline_progressive_flag=args.baseline_progressive_flag,
                dropout_mode=args.dropout_mode,
                velocity=args.velocity,
                ChannelorSpatial=args.ChannelorSpatial,
                spatialBlock=args.spatialBlock,

                dropout_recover_flag=args.dropout_recover_flag,
                MixStyle_flag=args.MixStyle_flag,
                MixStyle_p=args.MixStyle_p,
                MixStyle_Layers=args.MixStyle_Layers,
                MixStyle_detach_flag=args.MixStyle_detach_flag,
                MixStyle_mix_flag=args.MixStyle_mix_flag,

                test_dropout_flag=args.test_dropout_flag,

                train_diversity_flag=args.train_diversity_flag,
                test_diversity_flag=args.test_diversity_flag,
                regularizer_weight=1)
        elif args.network == 'resnet50':
            model = resnet50(
                pretrained=True,
                classes=args.n_classes,
                device=device,
                jigsaw_classes=1000,
                domains=args.n_domains,

                stage2_progressive_flag=args.stage2_progressive_flag,
                stage2_reverse_flag=args.stage2_reverse_flag,
                stage2_layers=args.stage2_layers,
                stage2_epochs_layer=args.epochs_layer,
                adjust_single_layer=args.adjust_single_layer,
                dropout_epoch_stop=args.dropout_epoch_stop,
                update_parameters_method=args.update_parameters_method,
                random_dropout_layers_flag=args.random_dropout_layers_flag,

                baseline_dropout_flag=args.baseline_dropout_flag,
                baseline_dropout_p=args.baseline_dropout_p,
                baseline_progressive_flag=args.baseline_progressive_flag,
                dropout_mode=args.dropout_mode,
                velocity=args.velocity,
                ChannelorSpatial=args.ChannelorSpatial,
                spatialBlock=args.spatialBlock,

                dropout_recover_flag=args.dropout_recover_flag,
                MixStyle_flag=args.MixStyle_flag,
                MixStyle_p=args.MixStyle_p,
                MixStyle_Layers=args.MixStyle_Layers,
                MixStyle_detach_flag=args.MixStyle_detach_flag,
                MixStyle_mix_flag=args.MixStyle_mix_flag,

                test_dropout_flag=args.test_dropout_flag,

                train_diversity_flag=args.train_diversity_flag,
                test_diversity_flag=args.test_diversity_flag,
                regularizer_weight=1)
        elif args.network == 'alexnet':
            model = alexnet(pretrained=True, classes=args.n_classes)
        elif args.network == 'convnet':
            model = cnn_digitsdg(
                classes=args.n_classes,
                device=device,
                domains=args.n_domains,

                stage2_progressive_flag=args.stage2_progressive_flag,
                stage2_reverse_flag=args.stage2_reverse_flag,
                stage2_layers=args.stage2_layers,
                stage2_epochs_layer=args.epochs_layer,
                adjust_single_layer=args.adjust_single_layer,
                dropout_epoch_stop=args.dropout_epoch_stop,
                update_parameters_method=args.update_parameters_method,
                random_dropout_layers_flag=args.random_dropout_layers_flag,

                baseline_dropout_flag=args.baseline_dropout_flag,
                baseline_dropout_p=args.baseline_dropout_p,
                baseline_progressive_flag=args.baseline_progressive_flag,
                dropout_mode=args.dropout_mode,
                velocity=args.velocity,
                ChannelorSpatial=args.ChannelorSpatial,
                spatialBlock=args.spatialBlock,

                dropout_recover_flag=args.dropout_recover_flag,
                MixStyle_flag=args.MixStyle_flag,
                MixStyle_p=args.MixStyle_p,
                MixStyle_Layers=args.MixStyle_Layers,
                MixStyle_detach_flag=args.MixStyle_detach_flag,
                MixStyle_mix_flag=args.MixStyle_mix_flag
            )
        else:
            model = resnet18(pretrained=True, classes=args.n_classes)

        self.model = model.to(device)

        self.source_loader, self.val_loader = data_helper.get_train_dataloader(args, patches=model.is_patch_based())
        self.target_loader = data_helper.get_val_dataloader(args, patches=model.is_patch_based())
        self.test_loaders = {"val": self.val_loader, "test": self.target_loader}
        self.len_dataloader = len(self.source_loader)
        print("Dataset size: train %d, val %d, test %d" % (
        len(self.source_loader.dataset), len(self.val_loader.dataset), len(self.target_loader.dataset)))

        self.optimizer, self.scheduler = get_optim_and_scheduler(model, network=args.network, epochs=args.epochs,
                                                                 lr=args.learning_rate, nesterov=args.nesterov)

        # this part is to create optimizers and schedules flexibly
        if args.stage2_one_stage_flag == 1:
            step_radio = 0.8
        else:
            step_radio = 1.0
        self.optimizers_scatter, self.schedulers_scatter = \
            get_optim_and_scheduler_scatter(model, network=args.network, epochs=args.epochs_style, lr=args.learning_rate_style,
                                            nesterov=args.nesterov, step_radio=step_radio)

        self.n_classes = args.n_classes
        if args.target in args.source:
            self.target_id = args.source.index(args.target)
            print("Target in source: %d" % self.target_id)
            print(args.source)
        else:
            self.target_id = None
        self.val_best = 0.0
        self.test_corresponding = 0.0

        self.label_smooth = args.label_smooth

        if self.args.stage1_flag == 1:
            # get the path of stage 1
            self.base_result_path = self.args.result_path + "/" + args.data + "/"
            self.base_result_path += "Deepall_RandAug_MixStyle/"
            self.base_result_path += args.network

            if self.args.MixStyle_flag == 1:
                line_layer = ""
                for i in self.args.MixStyle_Layers:
                    line_layer += str(i)
                self.base_result_path += "_MixStyle" + line_layer + "_random" + str(self.args.MixStyle_p)
                if self.args.MixStyle_mix_flag == 1:
                    self.base_result_path += "_mix"
                if self.args.MixStyle_detach_flag == 1:
                    self.base_result_path += "_detach"
            if self.args.gray_flag == 0:
                self.base_result_path += "_noGray"

            if self.args.RandAug_flag == 1:
                self.base_result_path += "_randaug" + "_m" + str(self.args.m) + "_n" + str(self.args.n)

            if self.args.label_smooth == 1:
                self.base_result_path += "_labelSmooth"

            self.base_result_path += "_stage1"

            self.base_result_path += "_lr" + str(self.args.learning_rate) + "_batch" + str(self.args.batch_size)
            self.base_result_path += "/"+self.args.target+str(self.args.time)+"/"

            if not os.path.exists(self.base_result_path):
                os.makedirs(self.base_result_path)
        print("The file of stage1 has been created.\n")

        if self.args.stage2_flag == 1:
            if self.args.stage1_flag == 1:
                # stage1 need to be trained
                self.stage2_base_path = self.base_result_path + "/"
            else:
                if args.data == "VLCS":
                    self.stage2_base_path = "/data/gjt/RSC-master/Content_Style/" + "VLCS_V100/" + args.data + "/Deepall_RandAug_MixStyle/"\
                                            + args.models_name[args.method_index] + "/" + args.target + str(args.time) + "/"
                else:
                    self.stage2_base_path = "/data/gjt/RSC-master/Content_Style/" + args.data + "/Deepall_RandAug_MixStyle/" \
                                            + args.models_name[args.method_index] + "/" + args.target + str(
                        args.time) + "/"
            if args.stage2_one_stage_flag == 1:
                self.base_stage2_path = self.stage2_base_path + "OneStage_"+str(args.epochs_style) + "dropout"
            else:
                self.base_stage2_path = self.stage2_base_path + "dropout"
            self.stage2_base_path += "_epochs" + str(self.args.epochs_style) + "_lr" + str(self.args.learning_rate_style)

            layer_line = ""
            for layer in args.stage2_layers:
                layer_line += str(layer)
            self.base_stage2_path = self.base_stage2_path + layer_line

            if args.stage2_progressive_flag == 1:
                self.base_stage2_path += "_layerProgressive"
            if args.stage2_reverse_flag == 1:
                self.base_stage2_path += "_reverse"
            if args.adjust_single_layer == 1:
                self.base_stage2_path += "_adjustOneLayer"
            if args.dropout_epoch_stop == 1:
                self.base_stage2_path += "_radioStop"

            if args.stage2_stage1_LastOrBest == 0:
                self.base_stage2_path += "_lastEpoch"
            else:
                self.base_stage2_path += "_bestEpoch"

            if args.random_dropout_layers_flag == 1 or args.random_dropout_layers_flag == 2:
                if args.random_dropout_layers_flag == 1:
                    self.base_stage2_path += "_random_one_layer"
                else:
                    self.base_stage2_path += "_all_layer"

                if args.update_parameters_method == 0:
                    if args.not_before_flag == 1:
                        self.base_stage2_path += "_not_before_first_layer"
                    else:
                        self.base_stage2_path += "_behind_first_layer"
                elif args.update_parameters_method == 1:
                    self.base_stage2_path += "_whole_network"
                elif args.update_parameters_method == 2:
                    if args.not_before_flag == 1:
                        self.base_stage2_path += "_not_before_layer"
                    else:
                        self.base_stage2_path += "_behind_layer"
                elif args.update_parameters_method == 3:
                    self.base_stage2_path += "_the_layer_behind"
                else:
                    raise NotImplementedError
            else:
                raise NotImplementedError

            if self.args.baseline_dropout_flag:
                self.base_stage2_path += "_all" + str(self.args.baseline_dropout_p)
                if self.args.baseline_progressive_flag:
                    self.base_stage2_path += "_progressive"
                    self.base_stage2_path += "_velocity" + str(self.args.velocity)
                if self.args.dropout_mode == 0:
                    self.base_stage2_path += "_normal"
                else:
                    if self.args.ChannelorSpatial == 0:
                        print("channel")
                        self.base_stage2_path += "_channel"
                    elif self.args.ChannelorSpatial == 1:
                        line = ""
                        for size in args.spatialBlock:
                            line += str(size)
                        self.base_stage2_path += "_spatial" + line
                    else:
                        line = ""
                        for size in args.spatialBlock:
                            line += str(size)
                        self.base_stage2_path += "_channel" + "_spatial" + line

            if self.args.label_smooth == 1:
                self.base_stage2_path += "_labelSmooth"

            self.base_stage2_path += "/"

            if not os.path.exists(self.base_stage2_path):
                os.makedirs(self.base_stage2_path)

    def get_optimizer(self, layer_selected=None):
        # this function is for getting optimizer according to the hype-parameters
        if self.args.update_parameters_method == 0:
            # behind the first layer or not before: 1 2 3 4 ？ 2 3 4 5
            if self.args.network == "convnet":
                first_layer = self.args.stage2_layers[0]
            else:
                first_layer = self.args.stage2_layers[0] + 2
            if self.args.not_before_flag == 1:
                first_layer -= 1
            optimizer = self.optimizers_scatter[first_layer:]

        elif self.args.update_parameters_method == 1:
            # the whole network
            optimizer = self.optimizers_scatter

        elif self.args.update_parameters_method == 2:
            # behind the selected layer or not before
            layer_index = layer_selected + 2
            if self.args.not_before_flag == 1:
                layer_index -= 1
            optimizer = self.optimizers_scatter[layer_index:]

        elif self.args.update_parameters_method == 3:
            layer_index = layer_selected + 2
            optimizer = [self.optimizers_scatter[layer_index]]

        else:
            raise NotImplementedError

        return optimizer

    def get_scheduler(self, layer_selected=None):
        # this function is for getting scheduler according to the hype-parameters
        if self.args.update_parameters_method == 0:
            # behind the first layer or not before: 1 2 3 4 ？ 2 3 4 5
            if self.args.network == "convnet":
                first_layer = self.args.stage2_layers[0]
            else:
                first_layer = self.args.stage2_layers[0] + 2
            if self.args.not_before_flag == 1:
                first_layer -= 1
            scheduler = self.schedulers_scatter[first_layer:]

        elif self.args.update_parameters_method == 1:
            # the whole network
            scheduler = self.schedulers_scatter

        elif self.args.update_parameters_method == 2:
            # behind the selected layer or not before
            layer_index = layer_selected + 2
            if self.args.not_before_flag == 1:
                layer_index -= 1
            scheduler = self.schedulers_scatter[layer_index:]

        elif self.args.update_parameters_method == 3:
            layer_index = layer_selected + 2
            scheduler = [self.schedulers_scatter[layer_index]]

        else:
            raise NotImplementedError

        return scheduler

    def _do_epoch(self, epoch=None):
        criterion = nn.CrossEntropyLoss()
        self.model.train()
        CE_loss = 0.0
        filter_loss = 0.0
        response_loss = 0.0
        batch_num = 0.0
        class_right = 0.0
        class_total = 0.0

        dropout_p = 0.0

        CD_layers = [0., 0., 0., 0.]
        KL_layers = [0., 0., 0., 0.]
        for it, ((data, data_randaug, class_l, domain_l), d_idx) in enumerate(self.source_loader):
            data, data_randaug, class_l, d_idx = data.to(self.device), data_randaug.to(self.device), class_l.to(self.device), d_idx.to(self.device)
            self.optimizer.zero_grad()

            if self.args.RandAug_flag == 1:
                data = torch.cat((data, data_randaug))
                class_l = torch.cat((class_l, class_l))

            class_logit, L_SR_response, L_SR_filter = self.model(data, class_l, mode="train", epoch=epoch)

            class_loss = criterion(class_logit, class_l)
            _, cls_pred = class_logit.max(dim=1)

            loss = class_loss

            loss.backward()
            self.optimizer.step()
            cls_acc = torch.sum(cls_pred == class_l.data)

            CE_loss += class_loss
            filter_loss += L_SR_filter
            response_loss += L_SR_response
            batch_num += 1
            class_right += cls_acc
            class_total += data.shape[0]

            self.logger.log(it, len(self.source_loader),
                            {
                                "class": class_loss.item(),
                                "filter": L_SR_filter.item(),
                                "response": L_SR_response.item()
                            },
                            {"class": cls_acc}, data.shape[0])
            del loss, class_loss, class_logit

        CE_loss = float(CE_loss/batch_num)
        filter_loss = float(filter_loss/batch_num)
        response_loss = float(response_loss/batch_num)
        class_acc = float(class_right / class_total)

        dropout_p = float(dropout_p/batch_num)

        if self.args.train_diversity_flag == 1:
            CD_layers /= batch_num
            KL_layers /= batch_num

            for i in range(len(CD_layers)):
                result = "train" + ": Epoch: " + str(epoch) + ", CELoss: " + str(format(CE_loss, '.4f')) + \
                ", ACC: " + str(format(class_acc, '.4f')) + ", channel diversity: " + str(format(CD_layers[i], '.4f'))\
                + ", KL divergence: " + str(format(KL_layers[i], '.4f')) + "\n"

                with open(self.base_diversity_result_path + "/" + "train" + "_layer" +str(i) + ".txt", "a") as f:
                    f.write(result)

        result = "train" + ": Epoch: " + str(epoch) + ", CELoss: " + str(format(CE_loss, '.4f')) \
                 + ", filter: " + str(format(filter_loss, '.4f')) + ", response: " + str(format(response_loss, '.4f')) + \
                 ", ACC: " + str(format(class_acc, '.4f'))
        if self.args.progressive_dropout_flag == 4:
            result += ", progressive dropout p: " + str(dropout_p)
        result += "\n"

        with open(self.base_result_path + "/" + "train" + ".txt", "a") as f:
            f.write(result)

        self.model.eval()
        with torch.no_grad():
            val_test_acc = []

            for phase, loader in self.test_loaders.items():
                total = len(loader.dataset)
                if self.args.test_diversity_flag == 1:
                    class_correct, class_loss, CD_layers_test, KL_layers_test = self.do_test(loader)
                else:
                    class_correct, class_loss = self.do_test(loader)
                class_acc = float(class_correct) / total
                val_test_acc.append(class_acc)

                if self.args.test_diversity_flag == 1:
                    for i in range(len(CD_layers_test)):
                        result = phase + ": Epoch: " + str(epoch) + ", CELoss: " + str(format(CE_loss, '.4f')) + \
                                 ", ACC: " + str(format(class_acc, '.4f')) + ", channel diversity: " + \
                                 str(format(CD_layers_test[i], '.4f')) + ", KL divergence: " + \
                                 str(format(KL_layers_test[i], '.4f')) + "\n"
                        with open(self.base_diversity_result_path + "/" + phase + "_layer" + str(i) + ".txt", "a") as f:
                            f.write(result)

                result = phase + ": Epoch: " + str(epoch) +", CELoss: "+str(format(class_loss.item(), '.4f')) + \
                         ", ACC: "+str(format(class_acc, '.4f')) + "\n"
                with open(self.base_result_path + "/" + phase + ".txt", "a") as f:
                    f.write(result)

                self.logger.log_test(phase, {"class": class_acc})
                self.results[phase][self.current_epoch] = class_acc
            if val_test_acc[0] >= self.val_best:
                self.val_best = val_test_acc[0]
                # self.save_best_model(acc_val=val_test_acc[0],  acc_test=val_test_acc[1], epoch=epoch)

    def _do_epoch_stage1(self, epoch=None):
        criterion = nn.CrossEntropyLoss() if self.label_smooth == 0 else CrossEntropyLoss(self.n_classes)

        self.model.train()
        CE_loss = 0.0

        batch_num = 0.0
        class_right = 0.0
        class_total = 0.0

        CD_layers = [0., 0., 0., 0.]
        KL_layers = [0., 0., 0., 0.]
        for it, ((data, data_randaug, class_l, domain_l), d_idx) in enumerate(self.source_loader):
            data, data_randaug, class_l, domain_l, d_idx = data.to(self.device), data_randaug.to(self.device), \
                                                           class_l.to(self.device), domain_l.to(self.device), \
                                                           d_idx.to(self.device)
            self.optimizer.zero_grad()
            if self.args.RandAug_flag == 1:
                data = torch.cat((data, data_randaug))
                class_l = torch.cat((class_l, class_l))
                domain_l = torch.cat((domain_l, domain_l))

            y = self.model(data, class_l, mode="train", epoch=epoch, stage=1)

            class_loss = criterion(y, class_l)
            _, cls_pred = y.max(dim=1)

            loss = class_loss
            loss.backward()
            self.optimizer.step()

            cls_acc = torch.sum(cls_pred == class_l.data)
            CE_loss += class_loss

            batch_num += 1
            class_right += cls_acc
            class_total += data.shape[0]


            self.logger.log(it, len(self.source_loader),
                            {
                                "class": class_loss.item(),
                            },
                            {
                                "class": cls_acc,
                            },
                            data.shape[0]
                            )
            del loss, class_loss, y

        CE_loss = float(CE_loss/batch_num)
        class_acc = float(class_right / class_total)

        if self.args.train_diversity_flag == 1:
            CD_layers /= batch_num
            KL_layers /= batch_num
            for i in range(len(CD_layers)):
                result = "train" + ": Epoch: " + str(epoch) \
                         + ", CELoss: " + str(format(CE_loss, '.4f')) + ", ACC: " + str(format(class_acc, '.4f')) \
                         + ", channel diversity: " + str(format(CD_layers[i], '.4f')) \
                         + ", KL divergence: " + str(format(KL_layers[i], '.4f')) + "\n"
                with open(self.base_diversity_result_path + "/" + "train" + "_layer" +str(i) + ".txt", "a") as f:
                    f.write(result)

        result = "train" + ": Epoch: " + str(epoch) + ", CELoss: " + str(format(CE_loss, '.4f')) + \
                 ", ACC: " + str(format(class_acc, '.4f'))
        result += "\n"
        with open(self.base_result_path + "/" + "train" + ".txt", "a") as f:
            f.write(result)

        self.model.eval()
        with torch.no_grad():
            val_test_acc = []
            for phase, loader in self.test_loaders.items():
                total = len(loader.dataset)
                class_correct, class_loss, CD_layers_test, KL_layers_test = self.do_test(loader, stage=1)

                class_acc = float(class_correct) / total
                val_test_acc.append(class_acc)

                if self.args.test_diversity_flag == 1:
                    for i in range(len(CD_layers_test)):
                        result = phase + ": Epoch: " + str(epoch) + ", CELoss: " + str(format(CE_loss, '.4f')) + \
                                 ", ACC: " + str(format(class_acc, '.4f')) + ", channel diversity: " + \
                                 str(format(CD_layers_test[i], '.4f')) + ", KL divergence: " + \
                                 str(format(KL_layers_test[i], '.4f')) + "\n"
                        with open(self.base_diversity_result_path + "/" + phase + "_layer" + str(i) + ".txt", "a") as f:
                            f.write(result)

                result = phase + ": Epoch: " + str(epoch) + ", CELoss: "+str(format(class_loss.item(), '.4f')) + \
                         ", ACC: " + str(format(class_acc, '.4f'))
                result += "\n"
                with open(self.base_result_path + "/" + phase + ".txt", "a") as f:
                    f.write(result)

                self.logger.log_test(phase, {"class": class_acc})
                self.results[phase][self.current_epoch] = class_acc
            if val_test_acc[0] >= self.val_best:
                self.val_best = val_test_acc[0]
                self.save_best_model(self.base_result_path)

    def _do_epoch_stage2(self, epoch=None):

        criterion = nn.CrossEntropyLoss() if self.label_smooth == 0 else CrossEntropyLoss(self.n_classes)

        self.model.train()
        CE_loss = 0.0

        batch_num = 0.0
        class_right = 0.0
        class_total = 0.0

        CD_layers = [0., 0., 0., 0.]
        KL_layers = [0., 0., 0., 0.]
        for it, ((data, data_randaug, class_l, domain_l), d_idx) in enumerate(self.source_loader):
            data, data_randaug, class_l, domain_l, d_idx = data.to(self.device), data_randaug.to(self.device), \
                                                           class_l.to(self.device), domain_l.to(self.device), \
                                                           d_idx.to(self.device)

            if self.args.RandAug_flag == 1:
                data = torch.cat((data, data_randaug))
                class_l = torch.cat((class_l, class_l))
                domain_l = torch.cat((domain_l, domain_l))

            layer_select = None
            if self.args.update_parameters_method == 0 or self.args.update_parameters_method == 1:
                # update the fixed parameters
                if self.args.random_dropout_layers_flag == 0:
                    layer_select = int(epoch / self.args.epochs_layer)  # 0 1 2 3
                    if self.args.stage2_reverse_flag == 1:
                        layer_select = len(self.args.stage2_layers) - layer_select - 1
                else:
                    layer_select = np.random.randint(len(self.args.stage2_layers), size=1)[0]
                optimizer = self.get_optimizer(layer_selected=None)
            else:
                if self.args.random_dropout_layers_flag == 0:
                    # select layer according to current epoch
                    stage2_layer = int(epoch / self.args.epochs_layer)
                    if self.args.stage2_reverse_flag == 1:
                        stage2_layer = len(self.args.stage2_layers) - stage2_layer - 1
                    optimizer = self.get_optimizer(self.args.stage2_layers[stage2_layer])
                else:
                    layer_select = np.random.randint(len(self.args.stage2_layers), size=1)[0]
                    optimizer = self.get_optimizer(self.args.stage2_layers[layer_select])

            for opt in optimizer:
                opt.zero_grad()

            y = self.model(data, class_l, mode="train", epoch=epoch, stage=2, layer_select=layer_select)
            class_loss = criterion(y, class_l)
            _, cls_pred = y.max(dim=1)
            loss = class_loss
            loss.backward()

            for opt in optimizer:
                opt.step()

            cls_acc = torch.sum(cls_pred == class_l.data)
            CE_loss += class_loss
            batch_num += 1
            class_right += cls_acc
            class_total += data.shape[0]
            self.logger.log(it, len(self.source_loader),
                            {
                                "class": class_loss.item(),
                            },
                            {
                                "class": cls_acc,
                            },
                            data.shape[0]
                            )
            del loss, class_loss, y

        CE_loss = float(CE_loss/batch_num)
        class_acc = float(class_right / class_total)

        if self.args.train_diversity_flag == 1:
            CD_layers /= batch_num
            KL_layers /= batch_num
            for i in range(len(CD_layers)):
                result = "train" + ": Epoch: " + str(epoch) \
                         + ", CELoss: " + str(format(CE_loss, '.4f')) + ", ACC: " + str(format(class_acc, '.4f')) \
                         + ", channel diversity: " + str(format(CD_layers[i], '.4f')) \
                         + ", KL divergence: " + str(format(KL_layers[i], '.4f')) + "\n"
                with open(self.base_diversity_result_path + "/" + "train" + "_layer" +str(i) + ".txt", "a") as f:
                    f.write(result)

        result = "train" + ": Epoch: " + str(epoch) \
                 + ", CELoss: " + str(format(CE_loss, '.4f')) + ", ACC: " + str(format(class_acc, '.4f'))
        result += "\n"
        with open(self.base_stage2_path + "/" + "train" + ".txt", "a") as f:
            f.write(result)

        self.model.eval()
        with torch.no_grad():
            val_test_acc = []
            for phase, loader in self.test_loaders.items():
                total = len(loader.dataset)
                class_correct, class_loss, CD_layers_test, KL_layers_test= self.do_test(loader, stage=2)
                class_acc = float(class_correct) / total
                val_test_acc.append(class_acc)

                if self.args.test_diversity_flag == 1:
                    for i in range(len(CD_layers_test)):
                        result = phase + ": Epoch: " + str(epoch) + ", CELoss: " + str(format(CE_loss, '.4f')) + \
                                 ", ACC: " + str(format(class_acc, '.4f')) + ", channel diversity: " + \
                                 str(format(CD_layers_test[i], '.4f')) + ", KL divergence: " + \
                                 str(format(KL_layers_test[i], '.4f')) + "\n"
                        with open(self.base_diversity_result_path + "/" + phase + "_layer" + str(i) + ".txt", "a") as f:
                            f.write(result)

                result = phase + ": Epoch: " + str(epoch) +\
                         ", CELoss: "+str(format(class_loss.item(), '.4f')) + ", ACC: "+str(format(class_acc, '.4f')) + "\n"
                with open(self.base_stage2_path + "/" + phase + ".txt", "a") as f:
                    f.write(result)
                self.logger.log_test(phase, {"class": class_acc})
                self.results[phase][self.current_epoch] = class_acc
            if val_test_acc[0] >= self.val_best:
                self.val_best = val_test_acc[0]
                self.save_best_model(self.base_stage2_path)

    def save_best_model(self, base_path):
        model_path = base_path + "models/"
        if not os.path.exists(model_path):
            os.makedirs(model_path)
        # line = "Best val %g, corresponding test %g, epoch: %g" % (acc_val, acc_test, epoch)
        torch.save(self.model.state_dict(), os.path.join(model_path, "model_best.pt"))

    def save_stage2_model(self, base_path):
        model_path = base_path + "models/"
        if not os.path.exists(model_path):
            os.makedirs(model_path)
        # line = "Best val %g, corresponding test %g, epoch: %g" % (acc_val, acc_test, epoch)
        torch.save(self.model.state_dict(), os.path.join(model_path, "model_last.pt"))

    def save_stage1_model(self):
        model_path = self.base_result_path + "models/"
        if not os.path.exists(model_path):
            os.makedirs(model_path)
        torch.save(self.model.state_dict(), os.path.join(model_path, "model_stage1.pt"))

    def _do_epoch_test(self, epoch, base_path=None):
        self.model.eval()
        with torch.no_grad():
            val_test_acc = []
            for phase, loader in self.test_loaders.items():
                total = len(loader.dataset)
                class_correct, class_loss, CD_layers_test, KL_layers_test = self.do_test(loader, stage=3)
                class_acc = float(class_correct) / total
                val_test_acc.append(class_acc)

                result = phase + ": Epoch: " + str(epoch) + ", CELoss: " + str(format(class_loss.item(), '.4f')) \
                         + ", ACC: " + str(format(class_acc, '.4f')) + "\n"

                print(result)
                # self.logger.log_test(phase, {"class": class_acc})
                # self.results[phase][self.current_epoch] = class_acc
            if val_test_acc[0] >= self.val_best:
                self.val_best = val_test_acc[0]
                # self.save_best_model(base_path)

    def do_test(self, loader, stage=None):
        class_correct = 0
        class_loss = 0
        criterion = nn.CrossEntropyLoss()
        CD_layers_test = [0., 0., 0., 0.]
        KL_layers_test = [0., 0., 0., 0.]
        batch_num = 0
        for it, ((data, _, class_l, domain_l), _) in enumerate(loader):
            data, class_l = data.to(self.device), class_l.to(self.device)
            class_logit, channel_diversity_layers, KL_divergence_layers = self.model(data, class_l, mode="test",
                                                                                     stage=stage)
            if self.args.test_diversity_flag == 1:
                CD_layers_test = [CD_layers_test[i] + channel_diversity_layers[i] for i in range(len(CD_layers_test))]
                KL_layers_test = [KL_layers_test[i] + KL_divergence_layers[i] for i in range(len(KL_layers_test))]
            _, cls_pred = class_logit.max(dim=1)
            class_loss += criterion(class_logit, class_l)
            class_correct += torch.sum(cls_pred == class_l.data)
            batch_num += 1

        if self.args.test_diversity_flag == 1:
            CD_layers_test = [CD / batch_num for CD in CD_layers_test]
            KL_layers_test = [KL / batch_num for KL in KL_layers_test]

        return class_correct, class_loss, CD_layers_test, KL_layers_test

    def _do_epoch_test_dropout(self, epoch, base_path=None, test_times=10):
        self.model.eval()
        with torch.no_grad():
            val_test_acc = []
            for phase, loader in self.test_loaders.items():
                total = len(loader.dataset)
                class_correct, _, class_loss, _, CD_layers_test, KL_layers_test = self.do_test_dropout(loader, stage=2, test_times=test_times)
                class_acc = float(class_correct) / total
                val_test_acc.append(class_acc)

                result = phase + ": Epoch: " + str(epoch) + ", CELoss: " + str(format(class_loss.item(), '.4f')) \
                         + ", ACC: " + str(format(class_acc, '.4f')) + "\n"

                print(result)

    def do_test_dropout(self, loader, stage=None, test_times=10):
        class_correct = 0
        layer_class_correct = 0
        class_loss = 0
        layer_class_loss = 0
        criterion = nn.CrossEntropyLoss()
        CD_layers_test = [0., 0., 0., 0.]
        KL_layers_test = [0., 0., 0., 0.]
        batch_num = 0
        for it, ((data, _, class_l, domain_l), _) in enumerate(loader):
            data, class_l = data.to(self.device), class_l.to(self.device)

            class_logits = []
            for i in range(test_times):
                class_logit, _, _, _ = self.model(data, class_l, mode="test", stage=stage)
                class_logits.append(class_logit)
            class_logits = torch.stack(class_logits, dim=0)
            class_logit = class_logits.mean(dim=0)

            _, cls_pred = class_logit.max(dim=1)
            class_loss += criterion(class_logit, class_l)
            class_correct += torch.sum(cls_pred == class_l.data)

        return class_correct, layer_class_correct, class_loss, layer_class_loss, CD_layers_test, KL_layers_test

    def do_training(self):
        self.logger = Logger(self.args, update_frequency=300)
        self.results = {"val": torch.zeros(self.args.epochs), "test": torch.zeros(self.args.epochs)}
        for self.current_epoch in range(self.args.epochs):
            self.scheduler.step()
            self.logger.new_epoch(self.scheduler.get_lr())
            start_time = time.time()
            self._do_epoch(self.current_epoch)
            end_time = time.time()
            print("Time for one epoch is " + str(format(end_time-start_time, '.0f')) + "s")
        val_res = self.results["val"]
        test_res = self.results["test"]
        idx_best = val_res.argmax()
        line = "Best val %g, corresponding test %g - best test: %g, best epoch: %g" % (
        val_res.max(), test_res[idx_best], test_res.max(), idx_best)
        print(line)
        with open(self.base_result_path+"test.txt", "a") as f:
            f.write(line+"\n")

        self.logger.save_best(test_res[idx_best], test_res.max())
        return self.logger, self.model

    def do_training_stage1(self):
        self.logger = Logger(self.args, update_frequency=300)
        self.results = {"val": torch.zeros(self.args.epochs), "test": torch.zeros(self.args.epochs)}
        for self.current_epoch in range(self.args.epochs):
            self.scheduler.step()
            self.logger.new_epoch(self.scheduler.get_lr())
            # print("begin to record times.\n")
            start_time = time.time()
            self._do_epoch_stage1(self.current_epoch)
            end_time = time.time()
            print("Time for one epoch is " + str(format(end_time - start_time, '.0f')) + "s")
        self.save_stage1_model()

        val_res = self.results["val"]
        test_res = self.results["test"]
        idx_best = val_res.argmax()
        line = "Best val %g, corresponding test %g - best test: %g, best epoch: %g" % (
            val_res.max(), test_res[idx_best], test_res.max(), idx_best)
        print(line)
        with open(self.base_result_path + "test.txt", "a") as f:
            f.write(line + "\n")

    def do_training_stage2(self):
        self.val_best = 0.0
        self.logger = Logger(self.args, update_frequency=300)
        self.results = {"val": torch.zeros(self.args.epochs_style), "test": torch.zeros(self.args.epochs_style)}

        if self.args.random_dropout_layers_flag == 1:
            epochs_num = self.args.epochs_style
        else:
            epochs_num = self.args.epochs_layer * len(self.args.stage2_layers)

        for self.current_epoch in range(epochs_num):

            if self.args.update_parameters_method == 0 or self.args.update_parameters_method == 1:
                scheduler_stage2 = self.get_scheduler(layer_selected=None)

            else:
                # scheduler is used to adjust the learning rate
                # we can adjust all the optimizers' lr, but there still exist not enough problems
                # we can also adjust according to the selected layer
                if self.args.random_dropout_layers_flag == 0:
                    layer_index = int(self.current_epoch / self.args.epochs_layer)
                    if self.args.stage2_reverse_flag == 1:
                        layer_index = len(self.args.stage2_layers) - layer_index - 1
                    scheduler_stage2 = self.get_scheduler(layer_selected=self.args.stage2_layers[layer_index])
                else:
                    # layer_select = np.random.randint(len(self.args.stage2_layers), size=1)[0]
                    # if not change lr, schedulers will be useless; if change lr, random changing seems to be unmeaning
                    # so here I adaptive the scheme that update lr of network behind the first layer
                    scheduler_stage2 = self.get_scheduler(layer_selected=self.args.stage2_layers[0])

            for scl in scheduler_stage2:
                scl.step()
            self.logger.new_epoch(scheduler_stage2[0].get_lr())

            start_time = time.time()
            self._do_epoch_stage2(self.current_epoch)
            end_time = time.time()
            print("Time for one epoch is " + str(format(end_time - start_time, '.0f')) + "s")
        self.save_stage2_model(self.base_stage2_path)
        val_res = self.results["val"]
        test_res = self.results["test"]
        idx_best = val_res.argmax()
        line = "Best val %g, corresponding test %g - best test: %g, best epoch: %g" % (
            val_res.max(), test_res[idx_best], test_res.max(), idx_best)
        print(line)
        with open(self.base_stage2_path + "test.txt", "a") as f:
            f.write(line + "\n")
        self.logger.save_best(test_res[idx_best], test_res.max())
        return self.logger, self.model


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
    'digits_dg': 32,
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

    domain = get_domain(args.data)
    args.target = domain.pop(args.target)
    args.source = domain
    print("Target domain: {}".format(args.target))

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

    models_name = [
        "resnet_layer_3_MixStyle3_random0.0_randaug_m4_n8_stage1_w10.0_w20.0_radio0.5_Conv1BNReLUDropout0.0_stage2_epochs_style30_learning_rate_style0.001_lr0.001_batch64",
        "resnet_layer_3_randaug_m4_n8_stage1_w10.0_w20.0_radio0.5_Conv1BNReLUDropout0.0_stage2_epochs_style30_learning_rate_style0.001_lr0.001_batch64",
        "resnet18_MixStyle3_random0.0_randaug_m4_n8_stage1_lr0.001_batch64", # 2
        # "resnet18_MixStyle3_random0.0_stage1_lr0.001_batch64",
        "resnet18_MixStyle3_random0.0_stage1_lr0.001_batch128",
        "resnet18_randaug_m4_n8_stage1_lr0.001_batch64",
        "resnet18_stage1_lr0.001_batch128",
        "convnet_stage1_lr0.01_batch128",

        # "resnet18_MixStyle3_random0.5_mix_detach_stage1_lr0.001_batch128",
        # "resnet18_MixStyle3_random0.5_mix_detach_randaug_m4_n8_stage1_lr0.001_batch64",
        # "resnet18_MixStyle123_random0.5_mix_detach_stage1_lr0.001_batch128",
        # "resnet18_MixStyle123_random0.5_mix_detach_randaug_m4_n8_stage1_lr0.001_batch64",
    ]
    args.models_name = models_name

    trainer = Trainer(args, device)
    if args.stage1_flag:
        print("Start Stage1 training.\n")
        trainer.do_training_stage1()
    # model load
    if args.stage2_flag:

        stage1_model_name = "model_stage1.pt" if args.stage2_stage1_LastOrBest == 0 else "model_best.pt"

        if args.data == "VLCS":
            model_path = "/data/gjt/RSC-master/Content_Style/" + "VLCS_V100/" + args.data + "/Deepall_RandAug_MixStyle/" + \
                         models_name[args.method_index] + "/" + args.target + str(args.time) + "/models/" + stage1_model_name
        else:
            model_path = "/data/gjt/RSC-master/Content_Style/"+ args.data + "/Deepall_RandAug_MixStyle/" + \
                         models_name[args.method_index] + "/" + args.target + str(
                args.time) + "/models/" + stage1_model_name

        if args.stage2_one_stage_flag == 0:
            trainer.model.load_state_dict(torch.load(model_path, map_location='cuda'), strict=False)
        trainer.do_training_stage2()


if __name__ == "__main__":
    torch.backends.cudnn.benchmark = True
    main()
