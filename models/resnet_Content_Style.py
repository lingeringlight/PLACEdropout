from torch import nn
from torch.utils import model_zoo
from torchvision.models.resnet import BasicBlock, model_urls, Bottleneck
import torch
from torch import nn as nn
from torch.autograd import Variable
import numpy.random as npr
import numpy as np
import torch.nn.functional as F
import random
import math
from.MixStyle import MixStyle
from sklearn.cluster import KMeans
from .Pooling import GeMPooling, AddAvgMaxPooling, CatAvgMaxPooling
from .layer_discriminator import Discriminator
from .DropBlock import DropBlock2D



class ResNet(nn.Module):
    def __init__(self, block, layers, device, jigsaw_classes=1000, classes=100,
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
                 MixStyle_mix_flag=0,

                 test_dropout_flag=0,
                 train_diversity_flag=0,
                 test_diversity_flag=0,
                 regularizer_weight=1):
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

        layer_channel = [3, 64, 128, 256, 512]
        layer_H_W = [224, 56, 28, 14, 7]

        self.stage2_epochs_layer = stage2_epochs_layer
        self.stage2_reverse_flag = stage2_reverse_flag
        self.stage2_layers = stage2_layers
        self.dropout_epoch_stop = dropout_epoch_stop

        # self.update_parameters_method = update_parameters_method
        self.random_dropout_layers_flag = random_dropout_layers_flag

        self.layers = [self.layer1, self.layer2, self.layer3, self.layer4, self.class_classifier]

        self.device = device

        self.initial_KL = 0
        self.initial_CS = 0

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
            self.MixStyle = MixStyle(p=MixStyle_p, mix='random', style_detach=MixStyle_detach_flag, mix_flag=MixStyle_mix_flag)
        self.train_diversity_flag = train_diversity_flag
        self.test_diversity_flag = test_diversity_flag
        self.regularizer_weight = regularizer_weight

        self.test_dropout_flag = test_dropout_flag

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def decoupling_params(self):
        # params = []
        pass

    def layer_pooling_methods(self, method, kernel_size, stride):
        # method:
        # 0: AvgPooling and AvgPooling
        # 1: Avg+Max
        # 2: Avg, Max
        # 3: GeMPooling and GeMPooling
        if method == 0:
            layer_class_pool = nn.AvgPool2d(kernel_size=kernel_size, stride=stride)
            layer_domain_pool = nn.AvgPool2d(kernel_size=kernel_size, stride=stride)
        elif method == 1:
            layer_class_pool = nn.AvgPool2d(kernel_size=kernel_size, stride=stride)
            layer_domain_pool = nn.MaxPool2d(kernel_size=kernel_size, stride=stride)
        elif method == 2:
            layer_class_pool = AddAvgMaxPooling(kernel_size=kernel_size, stride=stride)
            layer_domain_pool = AddAvgMaxPooling(kernel_size=kernel_size, stride=stride)
        elif method == 3:
            layer_class_pool = CatAvgMaxPooling(kernel_size=kernel_size, stride=stride)
            layer_domain_pool = CatAvgMaxPooling(kernel_size=kernel_size, stride=stride)
        else:
            layer_class_pool = GeMPooling(self.GeM_p)
            layer_domain_pool = GeMPooling(self.GeM_p)
        return layer_class_pool, layer_domain_pool


    def style_params(self, layer, adjust_single_layer=0):
        if layer == -1:
            return self.parameters()
        else:
            params = []
            if adjust_single_layer == 0:
                for m in self.layers[layer:]:
                    params += [p for p in m.parameters()]
            else:
                if layer == 3:
                    for m in self.layers[layer:]:
                        params += [p for p in m.parameters()]
                else: # 1 2 4
                    # for m in self.layers[layer: layer+1]:
                    m = self.layers[layer]
                    params = [p for p in m.parameters()]
            return params

    def deepall_params(self):
        params = []
        for m in self.layers:
            params += [p for p in m.parameters()]
        return params


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
                if self.random_dropout_layers_flag != 0:
                    if self.random_dropout_layers_flag == 1:
                        # randomly select one
                        dropout_layers = [self.stage2_layers[layer_select]]
                        dropout_epochs = [epoch, epoch, epoch]
                    elif self.random_dropout_layers_flag == 2:
                        dropout_layers = self.stage2_layers
                        dropout_epochs = [epoch, epoch, epoch]
                    else:
                        #Bernoulli
                        pass
                else:
                    #################
                    # This place need to be notice because I do not consider the Stage 1
                    stage2_layer = int(epoch / self.stage2_epochs_layer)
                    #################
                    if self.stage2_reverse_flag == 1:
                        dropout_layers = [self.stage2_layers[len(self.stage2_layers) - i - 1] for i in range(stage2_layer + 1)]
                    else:
                        dropout_layers = [self.stage2_layers[i] for i in range(stage2_layer + 1)]

                    dropout_epochs = [epoch, epoch - self.stage2_epochs_layer, epoch - self.stage2_epochs_layer * 2]
                    if self.dropout_epoch_stop == 1:
                        for i in range(len(dropout_epochs)):
                            dropout_epochs[i] = self.stage2_epochs_layer if dropout_epochs[i] > self.stage2_epochs_layer else dropout_epochs[i]

            if self.MixStyle_flag == 1 and 0 in self.MixStyle_Layers:
                x = self.MixStyle(x)
            for i, layer in enumerate([self.layer1, self.layer2, self.layer3, self.layer4]):
                x = layer(x)
                if self.MixStyle_flag == 1 and i+1 in self.MixStyle_Layers:
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

            x = self.avgpool(x)
            x = x.view(x.size(0), -1)
            y = self.class_classifier(x)

            return y

        else:
            channel_diversity_layers = [0., 0., 0., 0.]
            KL_divergence_layers = [0., 0., 0., 0.]

            for i, layer in enumerate([self.layer1, self.layer2, self.layer3, self.layer4]):
                x = layer(x)
                if self.test_diversity_flag == 1:
                    channel_diversity = self.response_diversity(layer_outputs=x)
                    channel_diversity_layers[i] += channel_diversity
                    KL_divergence = self.KL_divergence(layer_outputs=x)
                    KL_divergence_layers[i] += KL_divergence
                if stage == 2:
                    if self.test_dropout_flag:
                        # dropout in test time
                        if self.baseline_dropout_flag:
                            x = self.random_dropout(x, channel_drop=self.baseline_dropout_p,
                                                    spatial_drop=self.baseline_dropout_p, current_epoch=epoch)

            x = self.avgpool(x)
            x = x.view(x.size(0), -1)
            y = self.class_classifier(x)

            return y, channel_diversity_layers, KL_divergence_layers

    def CutFeature(self, x, mode=None):
        # this function is used to cut the input into two part
        pass

    def save_gradient1(self, grad):
        self.gradient1.append(grad)
    def save_gradient2(self, grad):
        self.gradient2.append(grad)
    def save_gradient3(self, grad):
        self.gradient3.append(grad)
    def save_gradient4(self, grad):
        self.gradient4.append(grad)

    def get_gradients(self, layers):
        gradients = [self.gradient1, self.gradient2, self.gradient3, self.gradient4]
        gradients_layer = [gradients[i-1] for i in layers]
        return gradients_layer

    def CAM_forward(self, x, layers):
        self.gradient1 = []
        self.gradient2 = []
        self.gradient3 = []
        self.gradient4 = []
        activations = []

        save_gradient_functions = [self.save_gradient1, self.save_gradient2, self.save_gradient3, self.save_gradient4]

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        # x = self.layer1(x)
        # x = self.layer2(x)
        # x = self.layer3(x)
        # x = self.layer4(x)

        for i, layer in enumerate([self.layer1, self.layer2, self.layer3, self.layer4]):
            x = layer(x)
            if i + 1 in layers:
                x.register_hook(save_gradient_functions[i])
                activations.append(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        y = self.class_classifier(x)

        return activations, y

    def KLCS_forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x_3 = self.layer3(x)
        x_4 = self.layer4(x_3)

        return x_3, x_4

    def DomainGap_forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        features = []
        for i, layer in enumerate([self.layer1, self.layer2, self.layer3, self.layer4]):
            x = layer(x)
            features.append(x)

        return features

    def feature_forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        for i, layer in enumerate([self.layer1, self.layer2, self.layer3, self.layer4]):
            x = layer(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        return x

    def feature_operate(self, x, labels):
        choose_one = random.randint(1, 10)
        if choose_one <= self.LCD_prob:
            L_SR_response, L_SR_filter = self.L_spatial_regularizations(layer_index=self.LCD_layer, layer_outputs=x)
            x = self.channels_dropout_mask(layer_outputs=x) if self.CD_flag else x
            x = x * self.spatial_attention_module(x).view(x.shape[0], x.shape[1], 1, 1) if self.Attention_flag else x
        else:
            L_SR_response = torch.tensor(0.).to(self.device)
            L_SR_filter = torch.tensor(0.).to(self.device)
        return x, L_SR_response, L_SR_filter

    def filters_diversity(self, parameters):
        # C1xC2xHxW
        # print(parameters.requires_grad)
        filter_size = parameters.shape[0]
        channel_size = parameters.shape[1]
        H = parameters.shape[2]
        W = parameters.shape[3]

        # version 1: as formula 1
        # # comput the diversity of HxW among the channel of different image
        para_F_C_HW = parameters.view(filter_size, channel_size, H * W)
        # 对每个channel上的元素进行normalization
        para_F_C_HW_norm = F.normalize(para_F_C_HW, p=2, dim=2)  # 不指明p，即为Frobenius范数
        # 计算channel p上，每个filter之间的cosine相似度
        para_C_F_HW_norm = para_F_C_HW_norm.permute(1, 0, 2)
        para_C_F_HW_norm_t = para_C_F_HW_norm.permute(0, 2, 1)
        para_C_F_F = torch.matmul(para_C_F_HW_norm, para_C_F_HW_norm_t)
        # 对每个channel上的所有元素求均值，并且取绝对值
        para_F_F = torch.abs(torch.mean(para_C_F_F, dim=0))

        # # version 2: as the Anchornet code
        # ## compute the diversity of CxHxW of filters
        # para_F_CHW = parameters.view(filter_size, channel_size*H*W)
        # para_F_CHW_norm = F.normalize(para_F_CHW, p=2, dim=1)
        # para_F_CHW_norm_t = para_F_CHW.permute(1,0)
        # para_F_F = torch.abs(torch.matmul(para_F_CHW_norm, para_F_CHW_norm_t))

        # 对除了对角线外的所有元素求和
        para_diag = para_F_F.diag().diag_embed()
        para_F_F_zeor_diag = para_F_F - para_diag
        L_SR_filter = para_F_F_zeor_diag.mean()
        return L_SR_filter

    def response_diversity(self, layer_outputs, group_flag=0):

        batch_size = layer_outputs.shape[0]
        channel_size = layer_outputs.shape[1]
        H = layer_outputs.shape[2]
        W = layer_outputs.shape[3]

        # whether divide channels into groups
        if group_flag == 1:
            # BxCxHxW =》 BxGxHxW
            group_num = self.LSR_response_group
            group_size = int(channel_size / group_num)
            initial_index = np.array(range(0, channel_size, group_size))
            relative_index = np.random.randint(0, group_size, group_num)
            channel_index = initial_index + relative_index
            x = layer_outputs[:, channel_index, :, :] # CxGxHxW
        else:
            x = layer_outputs

        # 将HxW展开为一维HW
        outputs_B_C_HW = x.view(batch_size, -1, H*W) # BxGxHW
        # 对每个channel上的元素进行normalization
        outputs_B_C_HW_norm = F.normalize(outputs_B_C_HW, p=2, dim=2)
        # 计算不同channel之间的相似度
        outputs_B_C_HW_norm_t = outputs_B_C_HW_norm.permute(0, 2, 1)
        outputs_B_C_C = torch.matmul(outputs_B_C_HW_norm, outputs_B_C_HW_norm_t).abs()
        outputs_B_C_C_diag = outputs_B_C_C.diagonal(dim1=1, dim2=2).diag_embed(dim1=1, dim2=2)
        outputs_B_C_C_zero_diag = outputs_B_C_C - outputs_B_C_C_diag
        L_SR_response = outputs_B_C_C_zero_diag.mean()
        return L_SR_response

    def CorrMatrix(self, x):
        # return the Similarity Matrix whose diagonal is zero
        batch_size = x.shape[0]
        channel_size = x.shape[1]
        H = x.shape[2]
        W = x.shape[3]

        # 将HxW展开为一维HW
        outputs_B_C_HW = x.view(batch_size, -1, H * W)  # BxGxHW
        # 对每个channel上的元素进行normalization
        outputs_B_C_HW_norm = F.normalize(outputs_B_C_HW, p=2, dim=2)
        # 计算不同channel之间的相似度
        outputs_B_C_HW_norm_t = outputs_B_C_HW_norm.permute(0, 2, 1)
        outputs_B_C_C = torch.matmul(outputs_B_C_HW_norm, outputs_B_C_HW_norm_t)
        outputs_B_C_C_diag = outputs_B_C_C.diagonal(dim1=1, dim2=2).diag_embed(dim1=1, dim2=2)
        outputs_B_C_C_zero_diag = outputs_B_C_C - outputs_B_C_C_diag

        return outputs_B_C_C_zero_diag


    def group_diversity(self, group1_output, group2_output):
        # this function is used to compute the similarity between the two group channels
        # you should notice whether you need .clone().detach() or not before you use this function
        batch_size = group1_output.shape[0]
        H = group1_output.shape[2]
        W = group1_output.shape[3]

        group1_B_C_HW = group1_output.view(batch_size, -1, H*W)
        group2_B_C_HW = group2_output.view(batch_size, -1, H*W)
        group1_B_C_HW_norm = F.normalize(group1_B_C_HW, p=2, dim=2)
        group2_B_C_HW_norm = F.normalize(group2_B_C_HW, p=2, dim=2)
        group2_B_C_HW_norm_t = group2_B_C_HW_norm.permute(0, 2, 1)
        group_B_C1_C2 = torch.matmul(group1_B_C_HW_norm, group2_B_C_HW_norm_t)

        group_similarity = group_B_C1_C2.mean()
        return group_similarity



    def L_spatial_regularizations(self, layer_index, layer_outputs):
        L_SR_response = torch.tensor(0.).to(self.device)
        if self.LSR_response_flag != 0:
            group_flag = 0 if self.LSR_response_flag == 1 else 1
            L_SR_response = self.response_diversity(layer_outputs, group_flag=group_flag)

        L_SR_filter = torch.tensor(0.).to(self.device)
        if self.LSR_filter_flag == 1:
            layer_name = "layer" + str(layer_index)
            if self.Conv_or_convs_flag == 0:
                layer_name += ".1.conv2.weight"
            for name, param in self.named_parameters():
                if layer_name in name and "conv" in name:
                    parameters = param
                    L_SR_filter += self.filters_diversity(parameters)

        # L_SR = self.alpha * L_SR_filter + self.beta * L_SR_response
        return L_SR_response, L_SR_filter

    def KL_divergence(self, layer_outputs):

        # output_sum = layer_outputs.sum()
        # out_norm = (layer_outputs / output_sum) + 1e-20
        outputs = layer_outputs
        out_norm = F.normalize(outputs, p=1, dim=None) + 1e-20
        uniform_tensor = torch.ones(out_norm.shape).to(self.device)
        uni_norm = uniform_tensor / torch.sum(uniform_tensor)
        kl_divergence = F.kl_div(input=out_norm.log(), target=uni_norm, reduction='batchmean')
        kl_loss = kl_divergence * self.regularizer_weight

        return kl_loss

    def channels_dropout_mask(self, layer_outputs):
        # 生成batchsize个gama
        gama_min = self.CD_drop_min
        gama_max = self.CD_drop_max

        batchsize = layer_outputs.shape[0]
        channel_num = layer_outputs.shape[1]
        H = layer_outputs.shape[2]
        W = layer_outputs.shape[3]
        gama_min_num = int(channel_num * gama_min)
        gama_max_num = int(channel_num * gama_max)
        gama_array = [random.randint(gama_min_num, gama_max_num+1) for i in range(batchsize)]
        # gama_array = random.randint(low=gama_min_num, high=gama_max_num, batchsize)

        mask = torch.ones_like(layer_outputs).to(self.device) #NCHW
        for i in range(batchsize):
            if self.CD_sample_flag:
                if random.randint(1, 3) == 1:
                    zero_channel_index = random.sample(range(0, channel_num), gama_array[i])
                    mask[i][zero_channel_index] = torch.zeros([H, W]).to(self.device)
            else:
                zero_channel_index = random.sample(range(0, channel_num), gama_array[i])
                mask[i][zero_channel_index] = torch.zeros([H, W]).to(self.device)

        layer_outputs_mask = layer_outputs * mask
        # print(layer_outputs_mask == layer_outputs)
        # print(torch.equal(layer_outputs,layer_outputs_mask))
        return layer_outputs_mask

    def setBlockZero_ODD(self, mask, x, y, i):
        H = mask.shape[1]
        W = mask.shape[2]

        mask[:, x, y] = torch.tensor(0.).to(self.device)
        if y - i > -1:
            mask[:, x, y - i] = torch.tensor(0.).to(self.device)
        if y + i < H:
            mask[:, x, y + i] = torch.tensor(0.).to(self.device)

        # the before column
        if x - i > -1:
            mask[:, x - i, y] = torch.tensor(0.).to(self.device)
            if y - i > -1:
                mask[:, x - i, y - i] = torch.tensor(0.).to(self.device)
            if y + i < H:
                mask[:, x - i, y + i] = torch.tensor(0.).to(self.device)

        if x + i < W:
            mask[:, x + i, y] = torch.tensor(0.).to(self.device)
            if y - i > -1:
                mask[:, x + i, y - i] = torch.tensor(0.).to(self.device)
            if y + i < H:
                mask[:, x + i, y + i] = torch.tensor(0.).to(self.device)

        return mask

    def setBlockZero_2(self, mask, x, y):
        H = mask.shape[1]
        W = mask.shape[2]

        mask[:, x, y] = torch.tensor(0.).to(self.device)
        if y + 1 < H:
            mask[:, x, y + 1] = torch.tensor(0.).to(self.device)

        if x + 1 < W:
            mask[:, x + 1, y] = torch.tensor(0.).to(self.device)
            if y + 1 < H:
                mask[:, x + 1, y + 1] = torch.tensor(0.).to(self.device)

        return mask


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
            # spatial_num = H * W
            # dropout_spatial_num = int(spatial_num * spatial_drop)
            # dropout_spatial_num = dropout_spatial_num // (self.spatialBlock * self.spatialBlock)
            #
            # mask = torch.ones_like(x).to(self.device)
            # for i in range(batch_size):
            #     zero_spatial_index = random.sample(range(0, spatial_num), dropout_spatial_num)
            #     x_index = [index // H for index in zero_spatial_index]
            #     y_index = [index % H for index in zero_spatial_index]
            #     for x_i, y_i in zip(x_index, y_index):
            #         if self.spatialBlock % 2 == 1: # ODD
            #             s_i = (self.spatialBlock - 1) // 2
            #             mask[i] = self.setBlockZero_ODD(mask[i], x_i, y_i, s_i)
            #         elif self.spatialBlock == 2:
            #             mask[i] = self.setBlockZero_2(mask[i], x_i, y_i)
            #
            # x = x * mask
            # if self.dropout_recover_flag:
            #     dropout_radio = dropout_spatial_num * self.spatialBlock * self.spatialBlock / spatial_num
            #     x = x * 1 / (1 - dropout_radio)
            block_size = random.choice(self.spatialBlock)
            x = self.drop_block(input=x, keep_prob=1 - spatial_drop, block_size=block_size)
            return x


    def CorrDropout(self, x, sample_drop=0., channel_drop=0., spatial_drop=0., current_epoch=0):
        x_detach = x.clone().detach()

        channel_num = x_detach.shape[1]

        corr_m = self.CorrMatrix(x_detach)
        corr_m_abs = corr_m.abs()
        f_matrix = torch.mean(corr_m_abs, dim=2) # BxC

        f_matrix_norm = F.softmax(f_matrix, dim=1)
        gamma = torch.ones_like(f_matrix_norm) - f_matrix_norm
        m_mask = torch.bernoulli(gamma) # BxC

        numel_m = channel_num * torch.ones_like(m_mask) # BxC
        sum_m = m_mask.sum(dim=1, keepdim=True) * torch.ones_like(m_mask) # BxC

        p = channel_drop + 0.001
        p_cap = p * numel_m / (numel_m - sum_m)
        p_cap = torch.ones_like(p_cap) - p_cap
        b_mask = torch.bernoulli(p_cap)

        s_mask = m_mask | b_mask    # BxC
        x = x * s_mask

        if self.dropout_recover_flag:
            sum_s = s_mask.sum(dim=1) # B
            numel_s = channel_num * torch.ones_like(sum_s) # B
            recover_radio = numel_s / sum_s # B
            x = x * recover_radio
        return x

    def CorrDropout_sort(self, x, sample_drop=0., channel_drop=0., spatial_drop=0., current_epoch=0):
        x_detach = x.clone().detach()

        channel_num = x_detach.shape[1]

        # channel_drop = 0.33
        corr_m = self.CorrMatrix(x_detach)
        corr_m_abs = corr_m.abs()
        f_matrix = torch.mean(corr_m_abs, dim=2) # BxC

        ######## 调整descending！
        sorted_f, _ = torch.sort(f_matrix, dim=1, descending=True)
        threshold_f = sorted_f[:, int(channel_num * channel_drop)] # Bx1
        threshold_f = threshold_f.unsqueeze(dim=1).expand_as(f_matrix)
        m_mask = torch.where(f_matrix < threshold_f, torch.tensor(1).cuda(), torch.tensor(0).cuda()) # BxC
        #########
        m_mask_BC11 = m_mask.view(-1, channel_num, 1, 1)  #BxCx1x1
        x = x * m_mask_BC11

        if self.dropout_recover_flag:
            sum_m = m_mask.sum(dim=1) # B
            numel_m = channel_num * torch.ones_like(sum_m) # B
            recover_radio = (numel_m / sum_m).unsqueeze(dim=1) # Bx1
            recover_radio = recover_radio.expand_as(f_matrix) # BxC
            recover_radio = recover_radio.view(-1, channel_num, 1 , 1)
            x = x * recover_radio
        return x


    def progressive_dropout_one_layer(self, x, current_epoch):
        # channel_dropout = (current_epoch / 10.) * 1. / 10. # 0%->10%->20%%
        channel_dropout = self.progressive_dropout_radio[int(current_epoch / self.progressive_stage_epoch)]
        x_dropout = self.random_dropout(x, channel_drop=channel_dropout)
        return x_dropout


    def progressive_dropout_metric(self, x, current_epoch, pmax, velocity=4):
        channel_dropout = pmax * 2 / np.pi * np.arctan(current_epoch/velocity)

        if self.dropout_mode == 0:
            x = nn.functional.dropout(x, p=channel_dropout, training=self.training)
        else:
            x = self.random_dropout(x, channel_drop=channel_dropout, spatial_drop=channel_dropout,
                                current_epoch=current_epoch)
        return x, channel_dropout

    def progressive_dropout_Corr(self, x, current_epoch, pmax):
        channel_dropout = pmax * 2 / np.pi * np.arctan(current_epoch/4)
        x = self.CorrDropout(x, channel_drop=channel_dropout, current_epoch=current_epoch)
        return x, channel_dropout

    def progressive_dropout_Corr_sort(self, x, current_epoch, pmax):
        channel_dropout = pmax * 2 / np.pi * np.arctan(current_epoch/4)
        x = self.CorrDropout_sort(x, channel_drop=channel_dropout, current_epoch=current_epoch)
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

def resnet101(pretrained=True, **kwargs):
    """Constructs a ResNet-101 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 4, 23, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet50']), strict=False)
    return model
