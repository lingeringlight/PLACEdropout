import torch
import numpy as np
import torch.nn.functional as F


# 测试传入model的神经元多样性
class NeuronDiversity:
    def __init__(self, model, device, regularizer_weight=0.005, save_path=None):

        self.model = model
        self.device = device
        self.regularizer_weight = regularizer_weight
        self.batch_num = 0
        self.save_path = save_path

        # initialize the neuron activation map
        layer_names = ["1", "2", "3", "4"]
        self.layer_names = layer_names
        layer_shape = [(64,56,56), (128,28,28), (256,14,14), (512,7,7)]
        self.layer_shape = layer_shape

        self.ND = [0. for i in range(len(self.layer_names))]
        self.CS = [0. for i in range(len(self.layer_names))]

    def reset_divergence(self):
        self.ND = [0. for i in range(len(self.layer_names))]
        self.batch_num = 0

    def norm_divergence(self, layer_index, layer_outputs):
        output_sum = layer_outputs.sum()
        out_norm = (layer_outputs / output_sum) + 1e-20
        uniform_tensor = torch.ones(out_norm.shape).to(self.device)
        uni_norm = uniform_tensor / torch.sum(uniform_tensor)
        # uni_norm = uniform_tensor / torch.prod(torch.tensor(out_norm.shape))
        divergence = F.kl_div(input=out_norm.log(), target=uni_norm, reduction='sum')
        divergence = divergence * self.regularizer_weight
        # print("layer:", layer_index, " divergence:", divergence)
        self.ND[layer_index] += divergence
        del output_sum, out_norm, uni_norm, divergence

    def channel_similarity(self, layer_index, layer_outputs):
        # N * C * H * W
        # feature:NxCxHxW
        batch_size = layer_outputs.shape[0]
        channel_size = layer_outputs.shape[1]
        H = layer_outputs.shape[2]
        W = layer_outputs.shape[3]
        # 将HxW展开为一维HW
        outputs_B_C_HW = layer_outputs.view(batch_size, channel_size, H * W)
        # 对每个channel上的元素进行normalization
        outputs_B_C_HW_norm = F.normalize(outputs_B_C_HW, p=2, dim=2)
        # 计算不同channel之间的相似度
        outputs_B_C_HW_norm_t = outputs_B_C_HW_norm.permute(0, 2, 1)
        outputs_B_C_C = torch.matmul(outputs_B_C_HW_norm, outputs_B_C_HW_norm_t)
        outputs_B_C_C_diag = outputs_B_C_C.diagonal(dim1=1, dim2=2).diag_embed(dim1=1, dim2=2)
        outputs_B_C_C_zero_diag = outputs_B_C_C - outputs_B_C_C_diag
        L_SR_response = outputs_B_C_C_zero_diag.mean()

        self.CS[layer_index] += L_SR_response

    def fill_divergence(self, input):
        x = self.model.conv1(input)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)
        x = self.model.layer1(x)
        self.norm_divergence(layer_index=0, layer_outputs=x)
        self.channel_similarity(layer_index=0, layer_outputs=x)
        x = self.model.layer2(x)
        self.norm_divergence(layer_index=1, layer_outputs=x)
        self.channel_similarity(layer_index=1, layer_outputs=x)
        x = self.model.layer3(x)
        self.norm_divergence(layer_index=2, layer_outputs=x)
        self.channel_similarity(layer_index=2, layer_outputs=x)
        x = self.model.layer4(x)
        self.norm_divergence(layer_index=3, layer_outputs=x)
        self.channel_similarity(layer_index=3, layer_outputs=x)

        # x = self.model.resnet.conv1(input)
        # x = self.model.resnet.bn1(x)
        # x = self.model.resnet.relu(x)
        # x = self.model.resnet.maxpool(x)
        # x = self.model.resnet.layer1(x)
        # self.norm_divergence(layer_index=0, layer_outputs=x)
        # x = self.model.resnet.layer2(x)
        # self.norm_divergence(layer_index=1, layer_outputs=x)
        # x = self.model.resnet.layer3(x)
        # self.norm_divergence(layer_index=2, layer_outputs=x)
        # x = self.model.resnet.layer4(x)
        # self.norm_divergence(layer_index=3, layer_outputs=x)

        self.batch_num += 1

    def show_divergence(self, mode=None):
        file_path = self.save_path + "/" + "divergence_" + str(mode) + ".txt"
        with open(file_path, "w") as f:
            divergence_all = 0.0
            for i, divergence in enumerate(self.ND):
                divergence_batch = divergence / self.batch_num
                f.write(str(divergence_batch) + "\n")
                print("layer"+str(i), " divergence:", divergence_batch)
                divergence_all += divergence_batch
            f.write(str(divergence_all) + "\n")
            print("diversity for all layers:", divergence_all)

    def show_similarity(self, mode=None):
        file_path = self.save_path + "/" + "similarity_" + str(mode) + ".txt"
        with open(file_path, "w") as f:
            similarity_all = 0.0
            for i, similarity in enumerate(self.CS):
                similarity_batch = similarity / self.batch_num
                f.write(str(similarity_batch) + "\n")
                print("layer"+str(i), " similarity:", similarity_batch)
                similarity_all += similarity_batch
            f.write(str(similarity_all) + "\n")
            print("Similarity for all layers:", similarity_all)

    def eval_nd(self, data_loader, device, mode=None):
        self.model.eval()
        self.reset_divergence()
        with torch.no_grad():
            for it, ((data, jig_l, class_l), d_idx) in enumerate(data_loader):
                data = data.to(device)
                self.fill_divergence(data)
        print("Divergence tracker has been updated!")
        self.show_divergence(mode=mode)
        self.show_similarity(mode=mode)

        # divergence 写入文件
        # NC写入方式修改



