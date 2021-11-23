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


class ResNet(nn.Module):
    def __init__(self, block, layers, device, jigsaw_classes=1000, classes=100,
                 CD_drop_max=1,
                 CD_drop_min=0,
                 CD_flag=0,
                 CD_sample_flag=0,
                 LSR_filter_flag=0,
                 LSR_response_flag=0,
                 LSR_response_group=32,
                 LCD_Layer=None,
                 Conv_or_Convs_flag=0,
                 Attention_flag=1,
                 LCD_prob=1,
                 mixup_flag=0,
                 progressive_dropout_flag=0,
                 progressive_dropout_radio=None,
                 progressive_dropout_all_PCD_flag=0,
                 progressive_dropout_all_epoch=None,
                 progressive_dropout_all_radio=None,
                 progressive_dropout_linear_epoch=None,
                 progressive_dropout_linear_epoch_radio=None,

                 progressive_dropout_metric_mode=None,
                 progressive_dropout_p_max=None,
                 progressive_dropout_metric_delta=None,
                 progressive_dropout_metric_sample_flag=0,

                 group_CD_flag=0,
                 group_channel_batch_average_flag=0,
                 drop_group_num=30,
                 group_progressive_mode=0,
                 progressive_dropout_recover_flag=0,
                 MixStyle_flag=0,
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
        self.avgpool = nn.AvgPool2d(7, stride=1)
        # self.jigsaw_classifier = nn.Linear(512 * block.expansion, jigsaw_classes)
        self.class_classifier = nn.Linear(512 * block.expansion, classes)
        #self.domain_classifier = nn.Linear(512 * block.expansion, domains)

        # layer_planes = [64, 128, 256, 512]
        # feature_map = [56, 28, 14, 7]
        # self.SA_conv = nn.Conv2d(layer_planes[LCD_Layer-1], layer_planes[LCD_Layer-1], kernel_size=1, stride=1, bias=False)
        # self.SA_FC1 = nn.Linear(layer_planes[LCD_Layer-1] * feature_map[LCD_Layer-1] * feature_map[LCD_Layer-1], layer_planes[LCD_Layer-1])
        # self.SA_GAP = nn.AvgPool2d(feature_map[LCD_Layer-1], stride=1)
        # self.SA_FC2 = nn.Linear(layer_planes[LCD_Layer-1], layer_planes[LCD_Layer-1])
        # self.sigmoid = torch.nn.Sigmoid()


        self.pecent = 1/3
        self.device = device
        # parameters for LCD
        self.CD_drop_max = CD_drop_max
        self.CD_drop_min = CD_drop_min
        self.CD_flag = CD_flag
        self.CD_sample_flag = CD_sample_flag
        self.LSR_filter_flag = LSR_filter_flag
        self.LSR_response_flag = LSR_response_flag
        self.LSR_response_group = LSR_response_group

        self.LCD_layer = LCD_Layer
        self.regularizer_weight = regularizer_weight
        self.Conv_or_convs_flag = Conv_or_Convs_flag
        self.Attention_flag = Attention_flag
        self.LCD_prob = LCD_prob
        self.mixup_flag = mixup_flag
        self.progressive_dropout_flag = progressive_dropout_flag
        self.progressive_dropout_radio = progressive_dropout_radio

        self.progressive_dropout_all_PCD_flag = progressive_dropout_all_PCD_flag
        self.progressive_dropout_all_epoch = progressive_dropout_all_epoch
        self.progressive_dropout_all_radio = progressive_dropout_all_radio

        self.progressive_dropout_linear_epoch = progressive_dropout_linear_epoch
        self.progressive_dropout_linear_epoch_radio = progressive_dropout_linear_epoch_radio

        self.progressive_dropout_metric_mode = progressive_dropout_metric_mode
        self.progressive_dropout_p_max = progressive_dropout_p_max
        self.progressive_dropout_metric_delta = progressive_dropout_metric_delta
        self.progressive_dropout_metric_sample_flag = progressive_dropout_metric_sample_flag

        self.group_CD_flag = group_CD_flag
        self.group_channel_batch_average_flag = group_channel_batch_average_flag
        self.drop_group_num = drop_group_num
        self.group_progressive_mode = group_progressive_mode

        self.progressive_dropout_recover_flag = progressive_dropout_recover_flag

        if self.progressive_dropout_radio != None:
            self.progressive_stage_epoch = int(30 / len(self.progressive_dropout_radio))

        self.initial_KL = 0
        self.initial_CS = 0

        self.MixStyle_flag = MixStyle_flag
        if MixStyle_flag != 0:
            self.MixStyle = MixStyle(mix='random')

        self.train_diversity_flag = train_diversity_flag
        self.test_diversity_flag = test_diversity_flag


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

    def spatial_attention_module(self, x):
        x_conv = self.SA_conv(x)
        # batch_size = x_conv.shape[0]
        # channel_num = x_conv.shape[1]
        # H = x_conv.shape[2]
        # W = x_conv.shape[3]
        # x_conv_new = x_conv.view(batch_size, channel_num * H * W)
        # x_fc1 = self.SA_FC1(x_conv_new)
        x_fc1 = self.SA_GAP(x_conv)
        x_fc1 = x_fc1.view(x_fc1.size(0), -1)
        x_fc2 = self.SA_FC2(x_fc1)
        # attention = nn.functional.softmax(x_fc2, dim=1) * x_conv.shape[1]
        # attention = 1 - nn.functional.softmax(x_fc2, dim=1)
        attention = self.sigmoid(x_fc2)
        # attention = x_fc2
        return attention

    def is_patch_based(self):
        return False

    def forward(self, x, gt=None, epoch=None, mode=None):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        if mode == "train":
            if self.progressive_dropout_flag == 0:
                L_SR_response = torch.tensor(0.).to(self.device)
                L_SR_filter = torch.tensor(0.).to(self.device)
                index = None
                lam = None

                x = self.layer1(x)
                if 1 in self.LCD_layer:
                # if self.LCD_layer == 1:
                    x, L_SR_response, L_SR_filter, index, lam = self.feature_operate(x, gt)

                x = self.layer2(x)
                if 2 in self.LCD_layer:
                    x, L_SR_response, L_SR_filter, index, lam = self.feature_operate(x, gt)

                x = self.layer3(x)
                if 3 in self.LCD_layer:
                    Lx, L_SR_response, L_SR_filter, index, lam = self.feature_operate(x, gt)

                x = self.layer4(x)
                if 4 in self.LCD_layer:
                    x, L_SR_response, L_SR_filter, index, lam = self.feature_operate(x, gt)

                x = self.avgpool(x)
                x = x.view(x.size(0), -1)
                return self.class_classifier(x), L_SR_response, L_SR_filter, index, lam

            elif self.progressive_dropout_flag == 1 or self.progressive_dropout_flag == 3:
                # progressive dropout for one layer
                channel_diversity_layers = [0., 0., 0., 0.]
                KL_divergence_layers = [0., 0., 0., 0.]

                x = self.layer1(x)
                if self.MixStyle_flag == 1:
                    x = self.MixStyle(x)

                if 1 in self.LCD_layer:
                    if self.group_CD_flag == 1:
                        x = self.progressive_dropout_group(x, current_epoch=epoch)
                    elif self.progressive_dropout_flag == 1:
                        x = self.progressive_dropout_one_layer(x, current_epoch=epoch)
                    else:
                        x = self.progressive_dropout_linear(x, current_epoch=epoch)

                if self.train_diversity_flag == 1:
                    channel_diversity = self.response_diversity(layer_outputs=x)
                    KL_divergence = self.KL_divergence(layer_outputs=x)
                    channel_diversity_layers[1-1] += channel_diversity
                    KL_divergence_layers[1-1] += KL_divergence


                x = self.layer2(x)
                if self.train_diversity_flag == 1:
                    x = self.MixStyle(x)

                if 2 in self.LCD_layer:
                    if self.group_CD_flag == 1:
                        x = self.progressive_dropout_group(x, current_epoch=epoch)
                    elif self.progressive_dropout_flag == 1:
                        x = self.progressive_dropout_one_layer(x, current_epoch=epoch)
                    else:
                        x = self.progressive_dropout_linear(x, current_epoch=epoch)

                if self.train_diversity_flag == 1:
                    channel_diversity = self.response_diversity(layer_outputs=x)
                    # KL_divergence = self.KL_divergence(layer_outputs=x)
                    channel_diversity_layers[2-1] += channel_diversity
                    # KL_divergence_layers[2-1] += KL_divergence


                x = self.layer3(x)
                if self.MixStyle_flag == 1:
                    x = self.MixStyle(x)
                if 3 in self.LCD_layer:
                    if self.group_CD_flag == 1:
                        x = self.progressive_dropout_group(x, current_epoch=epoch)
                    elif self.progressive_dropout_flag == 1:
                        x = self.progressive_dropout_one_layer(x, current_epoch=epoch)
                    else:
                        x = self.progressive_dropout_linear(x, current_epoch=epoch)

                if self.train_diversity_flag == 1:
                    channel_diversity = self.response_diversity(layer_outputs=x)
                    # KL_divergence = self.KL_divergence(layer_outputs=x)
                    channel_diversity_layers[3-1] += channel_diversity
                    # KL_divergence_layers[3-1] += KL_divergence


                x = self.layer4(x)
                if 4 in self.LCD_layer:
                    if self.group_CD_flag == 1:
                        x = self.progressive_dropout_group(x, current_epoch=epoch)
                    elif self.progressive_dropout_flag == 1:
                        x = self.progressive_dropout_one_layer(x, current_epoch=epoch)
                    else:
                        x = self.progressive_dropout_linear(x, current_epoch=epoch)

                if self.train_diversity_flag == 1:
                    channel_diversity = self.response_diversity(layer_outputs=x)
                    # KL_divergence = self.KL_divergence(layer_outputs=x)
                    channel_diversity_layers[4-1] += channel_diversity
                    # KL_divergence_layers[4-1] += KL_divergence

                x = self.avgpool(x)
                x = x.view(x.size(0), -1)
                if self.train_diversity_flag == 1:
                    return self.class_classifier(x), channel_diversity_layers, KL_divergence_layers
                else:
                    return self.class_classifier(x)
            elif self.progressive_dropout_flag == 4:

                x = self.layer1(x)
                if 1 in self.LCD_layer:
                    x, radio = self.progressive_dropout_metric(x, current_epoch=epoch)

                x = self.layer2(x)
                if 2 in self.LCD_layer:
                    x, radio = self.progressive_dropout_metric(x, current_epoch=epoch)

                x = self.layer3(x)
                if 3 in self.LCD_layer:
                    x, radio = self.progressive_dropout_metric(x, current_epoch=epoch)

                x = self.layer4(x)
                if 4 in self.LCD_layer:
                    x, radio = self.progressive_dropout_metric(x, current_epoch=epoch)

                x = self.avgpool(x)
                x = x.view(x.size(0), -1)
                return self.class_classifier(x), radio
            else:
                # progressive dropout for all layers
                x = self.layer1(x)
                x = self.progressive_dropout_all(x, current_epoch=epoch, layer_index=1)

                x = self.layer2(x)
                x = self.progressive_dropout_all(x, current_epoch=epoch, layer_index=2)

                x = self.layer3(x)
                x = self.progressive_dropout_all(x, current_epoch=epoch, layer_index=3)

                x = self.layer4(x)
                x = self.progressive_dropout_all(x, current_epoch=epoch, layer_index=4)

                x = self.avgpool(x)
                x = x.view(x.size(0), -1)
                return self.class_classifier(x)

        else:
            channel_diversity_layers = [0., 0., 0., 0.]
            KL_divergence_layers = [0., 0., 0., 0.]

            x = self.layer1(x)
            if self.test_diversity_flag == 1:
                channel_diversity = self.response_diversity(layer_outputs=x)
                channel_diversity_layers[1-1] += channel_diversity
                KL_divergence = self.KL_divergence(layer_outputs=x)
                KL_divergence_layers[1-1] += KL_divergence

            x = self.layer2(x)
            if self.test_diversity_flag == 1:
                channel_diversity = self.response_diversity(layer_outputs=x)
                channel_diversity_layers[2 - 1] += channel_diversity
                KL_divergence = self.KL_divergence(layer_outputs=x)
                KL_divergence_layers[2 - 1] += KL_divergence

            x = self.layer3(x)
            if self.test_diversity_flag == 1:
                channel_diversity = self.response_diversity(layer_outputs=x)
                channel_diversity_layers[3 - 1] += channel_diversity
                KL_divergence = self.KL_divergence(layer_outputs=x)
                KL_divergence_layers[3 - 1] += KL_divergence

            x = self.layer4(x)
            if self.test_diversity_flag == 1:
                channel_diversity = self.response_diversity(layer_outputs=x)
                channel_diversity_layers[4 - 1] += channel_diversity
                KL_divergence = self.KL_divergence(layer_outputs=x)
                KL_divergence_layers[4 - 1] += KL_divergence

            x = self.avgpool(x)
            x = x.view(x.size(0), -1)

            if self.test_diversity_flag == 1:
                return self.class_classifier(x), channel_diversity_layers, KL_divergence_layers
            else:
                return self.class_classifier(x)


    def feature_operate(self, x, labels):
        choose_one = random.randint(1, 10)
        if choose_one <= self.LCD_prob:
            L_SR_response, L_SR_filter = self.L_spatial_regularizations(layer_index=self.LCD_layer, layer_outputs=x)
            x = self.channels_dropout_mask(layer_outputs=x) if self.CD_flag else x
            x = x * self.spatial_attention_module(x).view(x.shape[0], x.shape[1], 1, 1) if self.Attention_flag else x
            # x, index, lam = self.channel_mixup(x, labels, alpha=1)
        else:
            L_SR_response = torch.tensor(0.).to(self.device)
            L_SR_filter = torch.tensor(0.).to(self.device)
            # index = None
            # lam = None
        if self.mixup_flag:
            x, index, lam = self.channel_mixup(x, labels, alpha=1)
        else:
            x, index, lam = x, None, None
        return x, L_SR_response, L_SR_filter, index, lam


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
        outputs_B_C_HW_norm_t = outputs_B_C_HW_norm.permute(0,2,1)
        outputs_B_C_C = torch.matmul(outputs_B_C_HW_norm, outputs_B_C_HW_norm_t)
        outputs_B_C_C_diag = outputs_B_C_C.diagonal(dim1=1, dim2=2).diag_embed(dim1=1, dim2=2)
        outputs_B_C_C_zero_diag = outputs_B_C_C - outputs_B_C_C_diag
        L_SR_response = outputs_B_C_C_zero_diag.mean()
        return L_SR_response


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

    def ramdom_dropout(self, x, sample_drop=0, channel_drop=0, spatial_drop=0, current_epoch=0):
        batch_size = x.shape[0]
        channel_num = x.shape[1]
        H = x.shape[2]
        W = x.shape[3]
        dropout_channel_num = int(channel_num * channel_drop)

        # smooth: same or different mask for each samples
        mask = torch.ones_like(x).to(self.device)

        prob_sample = 30. + current_epoch * 2. if self.progressive_dropout_metric_sample_flag else 100.
        # different
        for i in range(batch_size):
            p = random.randint(0, 99)
            if p < prob_sample:
                zero_channel_index = random.sample(range(0, channel_num), dropout_channel_num)
                mask[i][zero_channel_index] = torch.zeros([H, W]).to(self.device)

        x = x * mask
        if self.progressive_dropout_recover_flag:
            x = x * 1/(1 - channel_drop * prob_sample / 100)
        return x

    def group_feature_divide(self, x):
        # divide x into groups by selected clustering method
        # input: channel, CxHxW
        # output: group

        channel_num = x.shape[0]
        H = x.shape[1]
        W = x.shape[2]
        x_channel = x.view(channel_num, H*W)
        x_C_HW_norm = F.normalize(x_channel, p=2, dim=1)

        # 对x_channel进行聚类
        kmeans = KMeans(n_clusters=self.group_num).fit(x_C_HW_norm)
        group_labels = kmeans.labels_
        # group_labels_count = []
        # for i in range(self.group_num):
        #     group_labels_count.append(group_labels.count(i))

        return group_labels

    def group_feature_similarity(self, x):
        # get the similarity matrix of channel to channel
        # feature:NxCxHxW
        self.eval()

        x_detach = x.clone().detach()
        batch_size = x_detach.shape[0]
        channel_size = x_detach.shape[1]
        H = x_detach.shape[2]
        W = x_detach.shape[3]
        # 将HxW展开为一维HW
        outputs_B_C_HW = x_detach.view(batch_size, channel_size, H * W)
        # 对每个channel上的元素进行normalization
        outputs_B_C_HW_norm = F.normalize(outputs_B_C_HW, p=2, dim=2)
        # 计算不同channel之间的相似度
        outputs_B_C_HW_norm_t = outputs_B_C_HW_norm.permute(0, 2, 1)
        outputs_B_C_C = torch.matmul(outputs_B_C_HW_norm, outputs_B_C_HW_norm_t)
        # outputs_B_C_C_diag = outputs_B_C_C.diagonal(dim1=1, dim2=2).diag_embed(dim1=1, dim2=2)

        # NxCxC, diagonal is zero that means the similarity between channel and itself is 1
        similarity_matrix = outputs_B_C_C

        self.train()
        return similarity_matrix

    def group_dropout(self, x, group_drop=0):
        self.eval()
        x_detach = x.clone().detach()
        batch_num = x_detach.shape[0]
        channel_num = x_detach.shape[1]
        H = x_detach.shape[2]
        W = x_detach.shape[3]
        # 整个batch做mask or 每个样本做mask
        mask = torch.ones_like(x_detach).to(self.device)
        dropout_group_num = self.drop_group_num
        if self.group_channel_batch_average_flag == 1:
            # average the whole batch
            x_batch = torch.mean(x_detach, dim=0, keepdim=True) # BxCxHxW -> 1xCxHxW
            similarity_matrix = self.group_feature_similarity(x_batch) # 1xCxC
            zero_group_index = random.sample(range(0, channel_num), dropout_group_num)  # drop_group_num
            # 对similarity-matrix按dim=1降序排序

            dropout_group_center_similarity_matrix = torch.index_select(similarity_matrix, dim=1,
                                                                        index=torch.tensor(zero_group_index).cuda()) #1xNxC
            dropout_group_center_similarity_matrix_NC = dropout_group_center_similarity_matrix.view(1, dropout_group_num * channel_num) # 1xNC
            dropout_num = math.ceil(channel_num * group_drop) + 1
            thresholds = torch.sort(dropout_group_center_similarity_matrix_NC, dim=1, descending=True)[0][:, dropout_num]
            # 在similarity matrix中做mask
            thresholds = thresholds.unsqueeze(dim=1).expand(1, dropout_group_num * channel_num).view(1, dropout_group_num, channel_num)
            vector = torch.where(dropout_group_center_similarity_matrix > thresholds,
                                 torch.ones(dropout_group_center_similarity_matrix.shape).cuda(),
                                 torch.zeros(dropout_group_center_similarity_matrix.shape).cuda())

            zero_channel_index = vector.nonzero(as_tuple=True)[2]
            # print(zero_channel_index)
            if len(zero_channel_index) != 0:
            # print(zero_channel_index)
                fact_dropout_num = len(zero_channel_index)
                if fact_dropout_num != dropout_num:
                    # print(zero_channel_index)
                    temp = []
                    for j in range(dropout_num - fact_dropout_num):
                        temp.append(zero_channel_index[-1])
                    zero_channel_index = torch.cat([zero_channel_index, torch.stack(temp)], dim=0)
                    # print(zero_channel_index)
                zero_channel_index = zero_channel_index.view(1, dropout_num).unique(dim=1)
                mask[:][zero_channel_index] = torch.zeros([H, W]).to(self.device)

            self.train()
            x = x * mask
            if self.progressive_dropout_recover_flag:
                # channel_drop = float(zero_channel_index_count) / float(channel_num)
                x = x * 1 / (1 - group_drop)
            return x

        else:
            similarity_matrix = self.group_feature_similarity(x_detach) # BxCxC
            zero_group_index = random.sample(range(0, channel_num), dropout_group_num)  # N
            dropout_group_center_similarity_matrix = torch.index_select(similarity_matrix, dim=1,
                                                                        index=torch.tensor(zero_group_index).cuda())  # BxNxC
            dropout_group_center_similarity_matrix_NC = dropout_group_center_similarity_matrix.view(batch_num, dropout_group_num * channel_num) # BxNC
            dropout_num = math.ceil(channel_num * group_drop) + 1
            thresholds = torch.sort(dropout_group_center_similarity_matrix_NC, dim=1, descending=True)[0][:, dropout_num]
            thresholds = thresholds.unsqueeze(dim=1).expand(batch_num, dropout_group_num * channel_num).view(batch_num, dropout_group_num, channel_num)
            vector = torch.where(dropout_group_center_similarity_matrix > thresholds,
                                 torch.ones(dropout_group_center_similarity_matrix.shape).cuda(),
                                 torch.zeros(dropout_group_center_similarity_matrix.shape).cuda()) #BxNxC
            # print(len(vector.nonzero(as_tuple=True)[0]))
            indexes = vector.nonzero(as_tuple=True)
            B_index = indexes[0]
            C_index = indexes[2]
            if len(C_index) != batch_num * dropout_num:
                BxN_lenght = len(B_index)
                pre = -1
                zero_channel_index = [] # to B x selected channels
                temp = []
                for i in range(BxN_lenght):
                    current = B_index[i]
                    if pre != current and pre != -1:
                        for j in range(dropout_num - len(temp)):
                            temp.append(C_index[i-1])
                        zero_channel_index.append(temp)
                        temp = []
                    temp.append(C_index[i])
                    if i == BxN_lenght-1:
                        for j in range(dropout_num - len(temp)):
                            temp.append(C_index[i])
                        zero_channel_index.append(temp)
                    pre = current
                zero_channel_index = torch.tensor(zero_channel_index).cuda().unique(dim=1)
            else:
                zero_channel_index = C_index.view(batch_num, dropout_num).unique(dim=1)

            # zero_channel_index = torch.tensor(zero_channel_index).cuda().unique(dim=1)
            # print(zero_channel_index[0])

            for i in range(len(zero_channel_index)):
                mask[i][zero_channel_index[i]] = torch.zeros([H, W]).to(self.device)

            self.train()
            x = x * mask
            if self.progressive_dropout_recover_flag:
                # channel_drop = float(zero_channel_index_count) / float(channel_num)
                x = x * 1 / (1 - group_drop)
            return x


    def progressive_dropout_one_layer(self, x, current_epoch):
        # channel_dropout = (current_epoch / 10.) * 1. / 10. # 0%->10%->20%%
        channel_dropout = self.progressive_dropout_radio[int(current_epoch / self.progressive_stage_epoch)]
        x_dropout = self.ramdom_dropout(x, channel_drop=channel_dropout)
        return x_dropout

    def progressive_dropout_all(self, x, current_epoch, layer_index):
        if current_epoch >= self.progressive_dropout_all_epoch[layer_index-1]:
            if self.progressive_dropout_all_PCD_flag == 1:
                epoch = current_epoch - self.progressive_dropout_all_epoch[layer_index-1]
                channel_dropout = self.progressive_dropout_p_max * 2 / np.pi * np.arctan(epoch/4)
            else:
                channel_dropout = self.progressive_dropout_all_radio
            x_dropout = self.ramdom_dropout(x, channel_drop=channel_dropout)
            return x_dropout
        else:
            return x

    def progressive_dropout_linear(self, x, current_epoch):
        if current_epoch < self.progressive_dropout_linear_epoch:
            channel_dropout = current_epoch * self.progressive_dropout_linear_epoch_radio
            x = self.ramdom_dropout(x, channel_drop=channel_dropout)
        return x

    def progressive_dropout_metric(self, x, current_epoch):
        # x_detach = x.clone().detach()
        # if current_epoch == 0:
        #     self.initial_KL = self.KL_divergence(x_detach)
        #     self.initial_CS = self.response_diversity(x_detach)
        #
        # delta_metric = torch.abs(self.KL_divergence(x_detach) - self.initial_KL) if self.progressive_dropout_metric_mode == 0 \
        #     else torch.abs(self.response_diversity(x_detach)-self.initial_CS)
        # radio = delta_metric / self.progressive_dropout_metric_delta if delta_metric < self.progressive_dropout_metric_delta else torch.tensor(1.0)
        # channel_dropout = self.progressive_dropout_p_max * radio

        channel_dropout = self.progressive_dropout_p_max * 2 / np.pi * np.arctan(current_epoch/4)
        x = self.ramdom_dropout(x, channel_drop=channel_dropout, current_epoch=current_epoch)
        return x, channel_dropout

    def progressive_dropout_group(self, x, current_epoch):
        group_dropout = self.progressive_dropout_p_max * 2 / np.pi * np.arctan(current_epoch/4)
        x = self.group_dropout(x, group_drop=group_dropout)
        return x

    def channel_mixup(self, layer_outputs, labels, alpha):
        if self.mixup_flag:
            # label need to mixup
            batch_size = layer_outputs.shape[0]
            channel_num = layer_outputs.shape[1]
            H = layer_outputs.shape[2]
            W = layer_outputs.shape[3]

            # normal mixup
            index = torch.randperm(batch_size).to(self.device)
            lam = np.random.beta(alpha, alpha)

            ## feature mixup
            mixed_feature = lam * layer_outputs + (1-lam) * layer_outputs[index, :]
            # channel-wise exchange
            # mask = torch.ones_like(layer_outputs).to(self.device)  # NxCxHxW
            # channel_exchange = int(channel_num * (1-lam))
            # x_channel_index = random.sample(range(0, channel_num), channel_exchange)
            # mask[:, x_channel_index] = torch.zeros([H, W]).to(self.device)


        else:
            mixed_feature = layer_outputs
            index = None
            lam = None
        # label-aware channel mixup
        return mixed_feature, index, lam

def resnet18(pretrained=True, **kwargs):
    """Constructs a ResNet-18 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet18']), strict=False)
        # model.load_state_dict(torch.load("/data/gjt/RSC-master/RSC-master/Domain_Generalization/trainResults/FSR/sketch/model_best.pt", map_location='cuda:0'), strict=True)
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
