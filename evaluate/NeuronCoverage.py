import torch
import numpy as np
import os


# 测试传入model的神经元覆盖率
class NeuronCoverage:
    def __init__(self, model, threshold=None, record_path=None, max_min_path=None):

        self.model = model
        self.threshold = threshold
        self.thresholds = [0.005, 0.01, 0.02, 0.04, 0.08, 0.16, 0.32, 0.64]

        self.record_path = record_path
        self.max_min_path = max_min_path

        # initialize the neuron activation map
        layer_names = ["1", "2", "3", "4"]
        self.layer_names = layer_names
        layer_shape = [(64,56,56), (128,28,28), (256,14,14), (512,7,7)]
        self.layer_shape = layer_shape
        self.layer_neurons = [np.prod(i) for i in layer_shape]

        # self.sample_num = 0
        self.coverage_tracker = []
        self.coverage_image = []
        self.coverage_image_threshold = []

        self.image_max = []
        self.image_min = []

        try:
            for index_name in range(len(self.layer_names)):
                channel_val = np.zeros(self.layer_shape[index_name])
                self.coverage_tracker.append(channel_val)

                self.coverage_image.append([])

                # coverage_thresholds = []
                # for index_threshold in range(len(self.thresholds)):
                #     coverage_thresholds.append([])
                # self.coverage_image_threshold.append(coverage_thresholds)
                self.coverage_image_threshold.append([])

        except Exception as ex:
            raise Exception(f"Error while checking model layer to initialize neuron coverage tracker: {ex}")
        self.coverage_tracker = np.array(self.coverage_tracker)

    def reset_coverage_tracker(self):
        for index_name in range(len(self.layer_names)):
            self.coverage_tracker[index_name] = np.zeros(self.layer_shape[index_name])

    def reset_coverage_image(self):
        for index_name in range(len(self.layer_names)):
            self.coverage_image[index_name] = []

    def reset_coverage_image_thresholds(self):
        for index_name in range(len(self.layer_names)):
            coverage_thresholds = []
            for index_threshold in range(len(self.thresholds)):
                coverage_thresholds.append([])
            self.coverage_image[index_name] = coverage_thresholds

    def scale(self, layer_outputs, rmax=1, rmin=0):
        # BxCxHxW
        output_max = layer_outputs.max()
        output_min = layer_outputs.min()
        output_std = (layer_outputs - output_min) / (output_max - output_min)
        output_scaled = output_std * (rmax - rmin) + rmin

        self.image_max.append(output_max)
        self.image_min.append(output_min)
        return output_scaled

    def save_max_min(self, mode=None):
        max_path = os.path.join(self.max_min_path, mode, "max.txt")
        min_path = os.path.join(self.max_min_path, mode, "min.txt")
        with open(max_path, "w") as f:
            for value in self.image_max:
                f.write(str(value.cpu().numpy())+"\n")
        with open(min_path, "w") as f:
            for value in self.image_min:
                f.write(str(value.cpu().numpy())+"\n")


    def update_coverage(self, layer_outputs, layer_index):
        outputs = torch.squeeze(torch.sum(layer_outputs, dim=0))
        scaled_outputs = self.scale(outputs) # CxHxW
        self.coverage_tracker[layer_index] = np.where(scaled_outputs.cpu() > self.threshold, 1,
                                                      self.coverage_tracker[layer_index])

    def image_coverage(self, layer_outputs, layer_index):
        activate_all = 0
        for image_output in layer_outputs:
            # image_output: CxHxW
            scale_outputs = self.scale(image_output)
            activate = np.where(scale_outputs.cpu() > self.threshold, 1, 0)
            activate_sum = activate.sum()
            activate_all += activate_sum
        batch_size = len(layer_outputs)
        activate_image = activate_all / batch_size
        self.coverage_image[layer_index].append(activate_image)

    def image_coverage_threshold(self, layer_outputs, layer_index):
        # test the image coverage for each threshold
        activate_all = [0 for i in range(len(self.thresholds))]
        for image_output in layer_outputs:
            # image_output: CxHxW
            scale_outputs = self.scale(image_output)

            for i, threshold in enumerate(self.thresholds):
                activate = np.where(scale_outputs.cpu() > threshold, 1, 0)
                activate_sum = activate.sum()
                activate_all[i] += activate_sum
        batch_size = len(layer_outputs) # 接下来要修改的是rate函数，用于计算每个threshold对应的覆盖率
        activate_image = np.divide(activate_all, batch_size * self.layer_neurons[layer_index])
        self.coverage_image_threshold[layer_index].append(activate_image)

    def coverage_rate(self):
        # this function is for computing the neuron coverage rate
        # it should be run after filling the coverage_tracker
        activate_sum = 0
        all_sum = 0
        for i in range(len(self.layer_names)):
            layer_activate = self.coverage_tracker[i].sum()
            layer_all = np.prod(self.layer_shape[i])
            # layer_rate = layer_activate/layer_all
            # print(self.layer_names[i]+":" + str(layer_activate) + " / " + str(layer_all) + " = " + str(layer_rate))
            activate_sum += layer_activate
            all_sum += layer_all
        rate = activate_sum/all_sum
        print("The total coverage rate is " + str(activate_sum)+" / "+str(all_sum)+" = " + str(rate))

    def image_coverage_rate(self):
        # this function is for computing the average neuron coverage rate for each sample
        image_layer = []
        for index, coverage in enumerate(self.coverage_image):
            all_neuron_num = np.prod(self.layer_shape[index])
            activate_image = np.mean(coverage)

            coverage_rate = activate_image / all_neuron_num
            image_layer.append(coverage_rate)
            print("Threshold: {}, layer: {}, average coverage rate: {}".format(self.threshold, index, coverage_rate))

    def image_coverage_rate_thresholds(self):
        # this function is for computing the average neuron coverage rate for each sample with different threshold
        # coverage_image_threshold:[[......],[],[],[]] 4xnx7
        for index, coverage in enumerate(self.coverage_image_threshold):
            average_image = np.mean(coverage, axis=0, keepdims=False) # dim=7
            print("Layer: {}, average coverage rate: {}".format(index, average_image))
            self.record_file(index, average_image)

    def record_file(self, layer_index, layer_NC):
        with open(self.record_path + "layer" + str(layer_index) + ".txt", "a") as f:
            for nc in layer_NC:
                f.write(str(nc)+" ")
            f.write("\n")

    def fill_coverage_tracker(self, input):
        x = self.model.conv1(input)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)
        x = self.model.layer1(x)
        self.update_coverage(layer_index=0, layer_outputs=x)
        x = self.model.layer2(x)
        self.update_coverage(layer_index=1, layer_outputs=x)
        x = self.model.layer3(x)
        self.update_coverage(layer_index=2, layer_outputs=x)
        x = self.model.layer4(x)
        self.update_coverage(layer_index=3, layer_outputs=x)

        # x = self.model.resnet.conv1(input)
        # x = self.model.resnet.bn1(x)
        # x = self.model.resnet.relu(x)
        # x = self.model.resnet.maxpool(x)
        # x = self.model.resnet.layer1(x)
        # self.update_coverage(layer_index=0, layer_outputs=x)
        # x = self.model.resnet.layer2(x)
        # self.update_coverage(layer_index=1, layer_outputs=x)
        # x = self.model.resnet.layer3(x)
        # self.update_coverage(layer_index=2, layer_outputs=x)
        # x = self.model.resnet.layer4(x)
        # self.update_coverage(layer_index=3, layer_outputs=x)

        # x = self.model.base_model.conv1(input)
        # x = self.model.base_model.bn1(x)
        # x = self.model.base_model.relu(x)
        # x = self.model.base_model.maxpool(x)
        #
        # x = self.model.base_model.layer1(x)
        # self.update_coverage(layer_index=0, layer_outputs=x)
        #
        # x = self.model.base_model.layer2(x)
        # self.update_coverage(layer_index=1, layer_outputs=x)
        #
        # x = self.model.base_model.layer3(x)
        # self.update_coverage(layer_index=2, layer_outputs=x)
        #
        # x = self.model.base_model.layer4(x)
        # self.update_coverage(layer_index=3, layer_outputs=x)

    def fill_coverage_image(self, input):
        x = self.model.resnet.conv1(input)
        x = self.model.resnet.bn1(x)
        x = self.model.resnet.relu(x)
        x = self.model.resnet.maxpool(x)
        x = self.model.resnet.layer1(x)
        self.image_coverage(layer_index=0, layer_outputs=x)
        x = self.model.resnet.layer2(x)
        self.image_coverage(layer_index=1, layer_outputs=x)
        x = self.model.resnet.layer3(x)
        self.image_coverage(layer_index=2, layer_outputs=x)
        x = self.model.resnet.layer4(x)
        self.image_coverage(layer_index=3, layer_outputs=x)

    def fill_coverage_image_thresholds(self, input):
        x = self.model.resnet.conv1(input)
        x = self.model.resnet.bn1(x)
        x = self.model.resnet.relu(x)
        x = self.model.resnet.maxpool(x)
        x = self.model.resnet.layer1(x)
        self.image_coverage_threshold(layer_index=0, layer_outputs=x)
        x = self.model.resnet.layer2(x)
        self.image_coverage_threshold(layer_index=1, layer_outputs=x)
        x = self.model.resnet.layer3(x)
        self.image_coverage_threshold(layer_index=2, layer_outputs=x)
        x = self.model.resnet.layer4(x)
        self.image_coverage_threshold(layer_index=3, layer_outputs=x)

    def eval_nc(self, data_loader, device):
        self.model.eval()
        self.reset_coverage_tracker()
        with torch.no_grad():
            for it, ((data, jig_l, class_l), d_idx) in enumerate(data_loader):
                data = data.to(device)
                self.fill_coverage_tracker(data)
        print("coverage tracker has been updated!")
        self.coverage_rate()

    def eval_nc_image(self, data_loader, device, mode=None):
        self.model.eval()
        self.reset_coverage_image()
        with torch.no_grad():
            for it, ((data, jig_l, class_l), d_idx) in enumerate(data_loader):
                data = data.to(device)
                self.fill_coverage_image(data)
        print("image coverage tracker has been updated!")
        self.image_coverage_rate()
        # self.save_max_min(mode=mode)

    def eval_nc_image_thresholds(self, data_loader, device):
        self.model.eval()
        self.reset_coverage_image_thresholds()
        with torch.no_grad():
            for it, ((data, jig_l, class_l), d_idx) in enumerate(data_loader):
                data = data.to(device)
                self.fill_coverage_image_thresholds(data)
        print("image coverage tracker has been updated!")
        self.image_coverage_rate_thresholds()



# def SimCot(Cot1, Cot2):
#     # 分层计算每层神经元覆盖的相似度
#     perc_00 = []
#     perc_01 = []
#     perc_10 = []
#     perc_11 = []
#     for layer in range(len(Cot1)):
#         hot1 = Cot1[layer]
#         hot2 = Cot2[layer]
#         len_all = len(hot1)
#         len_00 = 0
#         len_01 = 0
#         len_10 = 0
#         len_11 = 0
#         for i in range(len_all):
#             if hot1[i] == 1 and hot2[i] == 1:
#                 len_11 += 1
#             elif hot1[i] == 1 and hot2[i] == 0:
#                 len_10 += 1
#             elif hot1[i] == 0 and hot2[i] == 1:
#                 len_01 += 1
#             else:
#                 len_00 += 1
#         len_all = float(len_all)
#         perc_00.append(len_00/len_all)
#         perc_01.append(len_01/len_all)
#         perc_10.append(len_10/len_all)
#         perc_11.append(len_11/len_all)
#
#     return perc_00, perc_01, perc_10, perc_11
#
# import matplotlib.pyplot as plt
# def drawBar(cots, mode, path):
#     for i in range(len(cots)):
#         plt.bar(range(len(cots[i])), cots[i], color='blue')
#         plt.savefig(path + mode + "_layer_" + str(i+1) + ".png")
#         plt.clf()



