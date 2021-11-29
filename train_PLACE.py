import argparse

import torch
from torch import nn
from data import data_helper
from models import model_factory
from optimizer.optimizer_helper import get_optim_and_scheduler, get_optim_and_scheduler_scatter
from utils.Logger import Logger
import numpy as np
from models.resnet_PLACE import resnet18, resnet50
import os
import random
import time


def get_args():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--target", default=1, type=int, help="Target")
    parser.add_argument("--device", type=int, default=3, help="GPU num")
    parser.add_argument("--time", default=1, type=int, help="train time")

    parser.add_argument("--batch_size", "-b", type=int, default=64, help="Batch size")
    parser.add_argument("--image_size", type=int, default=224, help="Image size")

    parser.add_argument("--result_path", default="/data/gjt/PLACE_results/", help="")

    parser.add_argument("--data", default="PACS")
    parser.add_argument("--network", choices=model_factory.nets_map.keys(), help="Which network to use",
                        default="resnet18")
    parser.add_argument("--data_root", default="/data/DataSets/")

    parser.add_argument("--SwapStyle_flag", default=1, type=int)
    parser.add_argument("--SwapStyle_detach_flag", default=0, type=int)
    parser.add_argument("--SwapStyle_Layers", default=[3], nargs="+", type=int)

    parser.add_argument("--RandAug_flag", default=1, type=int)
    parser.add_argument("--m", default=4, type=int)
    parser.add_argument("--n", default=8, type=int)

    # stage 1
    parser.add_argument("--learning_rate", "-l", type=float, default=.001, help="Learning rate")
    parser.add_argument("--epochs", "-e", type=int, default=30, help="Number of epochs for stage 1")
    parser.add_argument("--stage1_flag", default=1, type=int, help="whether train the stage 1.")
    parser.add_argument("--stage1_result_path", default="", help="if stage1_flag is 0, stage2 can "
                                                                 "be trained on the specified model directly."
                                                                 "For example, stage1_result_path + /sketch3/.")
    # stage 2
    parser.add_argument("--epochs_PLACE", type=int, default=30, help="Number of epochs for stage 2")
    parser.add_argument("--learning_rate_style", type=float, default=.001, help="Learning rate")

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

    parser.add_argument("--stage2_flag", default=1, type=int, help="whether train the stage 2.")
    parser.add_argument("--stage2_one_stage_flag", default=0, type=int, help="whether use one-stage scheme.")
    parser.add_argument("--stage2_stage1_LastOrBest", default=0, type=int, help="0: use the model of last epoch; "
                                                                                "1: use the model of best epoch.")
    parser.add_argument("--stage2_layers", default=[3, 4], nargs="+", type=int, help="Layers of PCD module")
    parser.add_argument("--update_parameters_method", type=int, default=0,
                        help="0: the module behind the first dropout layer;"
                             "1: the whole network;"
                             "2: the module behind the selected layer;"
                             "3: the layer behind the selected layer;")
    parser.add_argument("--random_dropout_layers_flag", default=1, type=int, help="whether to use random layer dropout;"
                                                                                  "0: all layer;"
                                                                                  "1: randomly select one layer;")
    parser.add_argument("--not_before_flag", type=int, default=0, help="whether include the layer dropout")

    # data aug stuff
    parser.add_argument("--min_scale", default=0.8, type=float, help="Minimum scale percent")
    parser.add_argument("--max_scale", default=1.0, type=float, help="Maximum scale percent")
    parser.add_argument("--gray_flag", default=1, type=int, help="whether use random gray")
    parser.add_argument("--random_horiz_flip", default=0.5, type=float, help="Chance of random horizontal flip")
    parser.add_argument("--jitter", default=0.4, type=float, help="Color jitter amount")
    parser.add_argument("--tile_random_grayscale", default=0.1, type=float,
                        help="Chance of randomly greyscaling a tile")
    #
    parser.add_argument("--limit_source", default=None, type=int,
                        help="If set, it will limit the number of training samples")
    parser.add_argument("--limit_target", default=None, type=int,
                        help="If set, it will limit the number of testing samples")
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

                stage2_layers=args.stage2_layers,
                random_dropout_layers_flag=args.random_dropout_layers_flag,

                baseline_dropout_flag=args.baseline_dropout_flag,
                baseline_dropout_p=args.baseline_dropout_p,
                baseline_progressive_flag=args.baseline_progressive_flag,
                dropout_mode=args.dropout_mode,
                velocity=args.velocity,
                ChannelorSpatial=args.ChannelorSpatial,
                spatialBlock=args.spatialBlock,

                dropout_recover_flag=args.dropout_recover_flag,
                SwapStyle_flag=args.SwapStyle_flag,
                SwapStyle_Layers=args.SwapStyle_Layers,
                SwapStyle_detach_flag=args.SwapStyle_detach_flag,
                )
        elif args.network == 'resnet50':
            model = resnet50(
                pretrained=True,
                classes=args.n_classes,
                device=device,
                jigsaw_classes=1000,
                domains=args.n_domains,

                stage2_layers=args.stage2_layers,
                random_dropout_layers_flag=args.random_dropout_layers_flag,

                baseline_dropout_flag=args.baseline_dropout_flag,
                baseline_dropout_p=args.baseline_dropout_p,
                baseline_progressive_flag=args.baseline_progressive_flag,
                dropout_mode=args.dropout_mode,
                velocity=args.velocity,
                ChannelorSpatial=args.ChannelorSpatial,
                spatialBlock=args.spatialBlock,

                dropout_recover_flag=args.dropout_recover_flag,
                SwapStyle_flag=args.SwapStyle_flag,
                SwapStyle_Layers=args.SwapStyle_Layers,
                SwapStyle_detach_flag=args.SwapStyle_detach_flag,
               )
        else:
            model = resnet18(pretrained=True, classes=args.n_classes)

        self.model = model.to(device)

        self.source_loader, self.val_loader = data_helper.get_train_dataloader(args, patches=model.is_patch_based())
        self.target_loader = data_helper.get_val_dataloader(args, patches=model.is_patch_based())
        self.test_loaders = {"val": self.val_loader, "test": self.target_loader}
        self.len_dataloader = len(self.source_loader)
        print("Dataset size: train %d, val %d, test %d" % (len(self.source_loader.dataset),
                                                           len(self.val_loader.dataset),
                                                           len(self.target_loader.dataset)))

        self.optimizer, self.scheduler = get_optim_and_scheduler(model, network=args.network, epochs=args.epochs,
                                                                 lr=args.learning_rate, nesterov=args.nesterov)

        # this part is to create optimizers and schedules flexibly
        if args.stage2_one_stage_flag == 1:
            step_radio = 0.8
        else:
            step_radio = 1.0
        self.optimizers_scatter, self.schedulers_scatter = \
            get_optim_and_scheduler_scatter(model, network=args.network, epochs=args.epochs_PLACE,
                                            lr=args.learning_rate_style, nesterov=args.nesterov, step_radio=step_radio)

        self.n_classes = args.n_classes

        self.val_best = 0.0
        self.test_corresponding = 0.0

        # Create the directories for stage 1 and 2
        if self.args.stage1_flag == 1:
            self.base_result_path = self.args.result_path + "/" + args.data + "/"
            self.base_result_path += args.network

            if self.args.SwapStyle_flag == 1:
                line_layer = ""
                for i in self.args.SwapStyle_Layers:
                    line_layer += str(i)
                self.base_result_path += "_SwapStyle" + line_layer + "_random"
                if self.args.SwapStyle_detach_flag == 1:
                    self.base_result_path += "_detach"
            if self.args.gray_flag == 0:
                self.base_result_path += "_noGray"

            if self.args.RandAug_flag == 1:
                self.base_result_path += "_randaug" + "_m" + str(self.args.m) + "_n" + str(self.args.n)

            self.base_result_path += "_lr" + str(self.args.learning_rate) + "_batch" + str(self.args.batch_size)
            self.base_result_path += "/" + self.args.target + str(self.args.time) + "/"

            if not os.path.exists(self.base_result_path):
                os.makedirs(self.base_result_path)
        print("The directory of stage1 has been created.")

        if self.args.stage2_flag == 1:
            if self.args.stage1_flag == 1:
                self.stage2_base_path = self.base_result_path + "/"
            else:
                self.stage2_base_path = args.stage1_result_path + "/" + self.args.target + str(self.args.time) + "/"

            if args.stage2_one_stage_flag == 1:
                self.base_stage2_path = self.stage2_base_path + "OneStage_" + str(args.epochs_PLACE) + "dropout"
            else:
                self.base_stage2_path = self.stage2_base_path + "dropout"
            self.stage2_base_path += "_epochs" + str(self.args.epochs_PLACE) + "_lr" + \
                                     str(self.args.learning_rate_style)

            layer_line = ""
            for layer in args.stage2_layers:
                layer_line += str(layer)
            self.base_stage2_path = self.base_stage2_path + layer_line

            if args.stage2_stage1_LastOrBest == 0:
                self.base_stage2_path += "_lastEpoch"
            else:
                self.base_stage2_path += "_bestEpoch"

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

            if self.args.baseline_dropout_flag == 1:
                self.base_stage2_path += "_all" + str(self.args.baseline_dropout_p)
                if self.args.baseline_progressive_flag == 1:
                    self.base_stage2_path += "_progressive"
                    self.base_stage2_path += "_velocity" + str(self.args.velocity)
                if self.args.dropout_mode == 0:
                    self.base_stage2_path += "_normal"
                else:
                    if self.args.ChannelorSpatial == 0:
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

            self.base_stage2_path += "/"

            if not os.path.exists(self.base_stage2_path):
                os.makedirs(self.base_stage2_path)

    def get_optimizer(self, layer_selected=None):
        # this function is for getting optimizer according to the hype-parameters
        if self.args.update_parameters_method == 0:
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

    def _do_epoch_stage1(self, epoch=None):
        criterion = nn.CrossEntropyLoss()

        self.model.train()
        CE_loss = 0.0

        batch_num = 0.0
        class_right = 0.0
        class_total = 0.0

        for it, ((data, data_randaug, class_l, domain_l), d_idx) in enumerate(self.source_loader):
            data, data_randaug, class_l, domain_l, d_idx = data.to(self.device), data_randaug.to(self.device), \
                                                           class_l.to(self.device), domain_l.to(self.device), \
                                                           d_idx.to(self.device)
            self.optimizer.zero_grad()
            if self.args.RandAug_flag == 1:
                data = torch.cat((data, data_randaug))
                class_l = torch.cat((class_l, class_l))
                # domain_l = torch.cat((domain_l, domain_l))

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
        CE_loss = float(CE_loss / batch_num)
        class_acc = float(class_right / class_total)

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
                class_correct, class_loss = self.do_test(loader, stage=1)

                class_acc = float(class_correct) / total
                val_test_acc.append(class_acc)

                result = phase + ": Epoch: " + str(epoch) + ", CELoss: " + str(format(class_loss.item(), '.4f')) + \
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
        criterion = nn.CrossEntropyLoss()
        self.model.train()
        CE_loss = 0.0

        batch_num = 0.0
        class_right = 0.0
        class_total = 0.0

        for it, ((data, data_randaug, class_l, domain_l), d_idx) in enumerate(self.source_loader):
            data, data_randaug, class_l, domain_l, d_idx = data.to(self.device), data_randaug.to(self.device), \
                                                           class_l.to(self.device), domain_l.to(self.device), \
                                                           d_idx.to(self.device)

            if self.args.RandAug_flag == 1:
                data = torch.cat((data, data_randaug))
                class_l = torch.cat((class_l, class_l))
                # domain_l = torch.cat((domain_l, domain_l))

            layer_select = np.random.randint(len(self.args.stage2_layers), size=1)[0]
            if self.args.update_parameters_method == 0 or self.args.update_parameters_method == 1:
                # update the fixed parameters
                optimizer = self.get_optimizer(layer_selected=None)
            else:
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
        CE_loss = float(CE_loss / batch_num)
        class_acc = float(class_right / class_total)

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
                class_correct, class_loss = self.do_test(loader, stage=2)
                class_acc = float(class_correct) / total
                val_test_acc.append(class_acc)

                result = phase + ": Epoch: " + str(epoch) + ", CELoss: " + str(format(class_loss.item(), '.4f')) + \
                         ", ACC: " + str(format(class_acc, '.4f')) + "\n"
                with open(self.base_stage2_path + "/" + phase + ".txt", "a") as f:
                    f.write(result)
                self.logger.log_test(phase, {"class": class_acc})
                self.results[phase][self.current_epoch] = class_acc
            if val_test_acc[0] >= self.val_best:
                self.val_best = val_test_acc[0]
                self.save_best_model(self.base_stage2_path)

    def save_last_model(self, base_path):
        model_path = base_path + "models/"
        if not os.path.exists(model_path):
            os.makedirs(model_path)
        torch.save(self.model.state_dict(), os.path.join(model_path, "model_last.pt"))

    def save_best_model(self, base_path):
        model_path = base_path + "models/"
        if not os.path.exists(model_path):
            os.makedirs(model_path)
        torch.save(self.model.state_dict(), os.path.join(model_path, "model_best.pt"))

    def do_test(self, loader, stage=None):
        class_correct = 0
        class_loss = 0
        criterion = nn.CrossEntropyLoss()

        batch_num = 0
        for it, ((data, _, class_l, domain_l), _) in enumerate(loader):
            data, class_l = data.to(self.device), class_l.to(self.device)
            class_logit = self.model(data, class_l, mode="test", stage=stage)
            _, cls_pred = class_logit.max(dim=1)
            class_loss += criterion(class_logit, class_l)
            class_correct += torch.sum(cls_pred == class_l.data)
            batch_num += 1

        return class_correct, class_loss

    def do_training_stage1(self):
        self.logger = Logger(self.args, update_frequency=300)
        self.results = {"val": torch.zeros(self.args.epochs), "test": torch.zeros(self.args.epochs)}
        for self.current_epoch in range(self.args.epochs):
            self.logger.new_epoch(self.scheduler.get_last_lr())
            start_time = time.time()
            self._do_epoch_stage1(self.current_epoch)
            self.scheduler.step()
            end_time = time.time()
            print("Time for one epoch is " + str(format(end_time - start_time, '.0f')) + "s")
        self.save_last_model(self.base_result_path)

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
        self.results = {"val": torch.zeros(self.args.epochs_PLACE), "test": torch.zeros(self.args.epochs_PLACE)}

        for self.current_epoch in range(self.args.epochs_PLACE):
            if self.args.update_parameters_method == 0 or self.args.update_parameters_method == 1:
                scheduler_stage2 = self.get_scheduler(layer_selected=None)
            else:
                scheduler_stage2 = self.get_scheduler(layer_selected=self.args.stage2_layers[0])
            self.logger.new_epoch(scheduler_stage2[0].get_last_lr())
            start_time = time.time()
            self._do_epoch_stage2(self.current_epoch)
            for scl in scheduler_stage2:
                scl.step()
            end_time = time.time()
            print("Time for one epoch is " + str(format(end_time - start_time, '.0f')) + "s")
        self.save_last_model(self.base_stage2_path)
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
    # 'miniDomainNet': ['clipart', 'painting', 'real', 'sketch'],
    # 'digits_dg': ['mnist', 'mnist_m', 'svhn', 'syn'],
}

classes_map = {
    'PACS': 7,
    'PACS_random_split': 7,
    'OfficeHome': 65,
    # 'miniDomainNet': 126,
    'VLCS': 5,
    # 'digits_dg': 10,
}

domains_map = {
    'PACS': 3,
    'PACS_random_split': 3,
    'OfficeHome': 3,
    # 'miniDomainNet': 3,
    'VLCS': 3,
    # 'digits_dg': 3,
}

val_size_map = {
    'PACS': 0.1,
    'PACS_random_split': 0.1,
    'OfficeHome': 0.1,
    # 'miniDomainNet': 0.1,
    'VLCS': 0.3,
    # 'digits_dg': 0.2,
}

image_size_map = {
    'PACS': 224,
    'PACS_random_split': 224,
    'OfficeHome': 224,
    # 'miniDomainNet': 96,
    'VLCS': 224,
    # 'digits_dg': 32,
}

max_scale_map = {
    'PACS': 1.0,
    'PACS_random_split': 1.0,
    'OfficeHome': 1.0,
    # 'miniDomainNet': 1.25,
    'VLCS': 1.0,
    # 'digits_dg': 1.0,
}


def get_domain(name):
    if name not in domain_map:
        raise ValueError('Name of dataset unknown %s' % name)
    return domain_map[name]


def main():
    args = get_args()

    domain = get_domain(args.data)
    args.target = domain.pop(args.target)
    args.source = domain
    print("Target domain: {}".format(args.target))

    if "PACS" in args.data:
        args.data_root = os.path.join(args.data_root, "PACS")
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

    trainer = Trainer(args, device)
    if args.stage1_flag:
        print("Start Stage1 training.")
        trainer.do_training_stage1()
    # model load
    if args.stage2_flag:
        stage1_model_name = "model_last.pt" if args.stage2_stage1_LastOrBest == 0 else "model_best.pt"
        if args.stage1_flag:
            model_path = trainer.base_result_path + "/models/" + stage1_model_name
        else:
            model_path = args.stage1_result_path + "/models/" + stage1_model_name
        if args.stage2_one_stage_flag == 0:
            trainer.model.load_state_dict(torch.load(model_path, map_location='cuda'), strict=False)
        trainer.do_training_stage2()


if __name__ == "__main__":
    torch.backends.cudnn.benchmark = True
    main()
