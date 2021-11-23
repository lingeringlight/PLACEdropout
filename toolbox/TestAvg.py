# 统计test.txt中的结果，并且输出在一个文件中
import os
import numpy as np

data_index = 2
datasets = ["PACS", "VLCS", "OfficeHome", "digits_dg", "miniDomainNet"]
datasets_dict = [
    {"photo":[], "art_painting":[], "cartoon":[], "sketch":[]},
    {"CALTECH":[], "LABELME":[], "PASCAL":[], "SUN":[]},
    {'Art':[], 'Clipart':[], 'Product':[], 'RealWorld':[]},
    {'mnist':[], 'mnist_m':[], 'svhn':[], 'syn':[]},
    {'clipart':[], 'painting':[], 'real':[], 'sketch':[]},
]
result_dict = datasets_dict[data_index]
data = datasets[data_index]

# path = "/data/gjt/RSC-master/LCD/" + data + "/"
# path = "/data/gjt/RSC-master/LCD/exploreDiversity/" + data + "/"
path = "/data/gjt/RSC-master/Content_Style/" + data + "/Deepall_RandAug_MixStyle/" + "/"
# path = "/data/gjt/RSC-master/RSC/VLCS/"
# path = "/data/gjt/RSC-master/Content_Style/VLCS_V100/VLCS/" + "/Deepall_RandAug_MixStyle/"
# path = "/data/gjt/RSC-master/MixStyle/" + data + "/"
dirs = os.listdir(path)

dropout_methods = [

    # "dropout34_lastEpoch_random_one_layer_behind_first_layer_all0.33_progressive_velocity4_channel_spatial3",
    # "dropout3_lastEpoch_random_one_layer_behind_first_layer_all0.33_progressive_velocity4_spatial4",
    # "dropout4_lastEpoch_random_one_layer_behind_first_layer_all0.33_progressive_velocity4_spatial4",
    # "dropout34_lastEpoch_random_one_layer_behind_first_layer_all0.33_progressive_velocity4_spatial5",
    # "dropout3_lastEpoch_random_one_layer_behind_first_layer_all0.33_progressive_velocity4_spatial35",
    # "dropout4_lastEpoch_random_one_layer_behind_first_layer_all0.33_progressive_velocity4_spatial35",
    # "dropout34_lastEpoch_random_one_layer_behind_first_layer_all0.33_progressive_velocity4_spatial35",
    # "dropout34_lastEpoch_random_one_layer_behind_first_layer_all0.33_progressive_velocity4_channel_labelSmooth",

    # "dropout34_epoch15_lr0.001_lastEpoch_random_one_layer_behind_first_layer_all0.33_progressive",
    # "dropout34_lastEpoch_random_one_layer_behind_first_layer_all0.33_progressive_velocity4_channel",
    # "dropout34_bestEpoch_random_one_layer_behind_first_layer_all0.33_progressive",
    # "dropout34_lastEpoch_random_one_layer_behind_first_layer_all0.25_progressive",
    
    # "dropout34_lastEpoch_all_layer_behind_first_layer_all0.33_channel",
    # "dropout34_lastEpoch_all_layer_behind_first_layer_all0.33_normal",
    # "dropout34_lastEpoch_random_one_layer_behind_first_layer_all0.33_normal",
    # "dropout34_lastEpoch_random_one_layer_behind_first_layer_all0.33_channel",

    # "dropout34_lastEpoch_random_one_layer_behind_first_layer_all0.33_progressive_velocity4_channel",


    # "dropout34_lastEpoch_all_layer_behind_first_layer_all0.33_progressive_velocity4_channel",
    # "dropout34_lastEpoch_random_one_layer_behind_first_layer_all0.33_progressive_velocity4_channel",
    # "dropout34_lastEpoch_all_layer_behind_first_layer_all0.33_progressive_velocity4_normal",

    # "dropout34_lastEpoch_random_one_layer_behind_first_layer_all0.33_progressive_velocity4_normal",
    # "dropout34_lastEpoch_random_one_layer_behind_first_layer_all0.33_progressive_velocity4_channel",


    # "dropout34_lastEpoch_random_one_layer_whole_network_all0.33_progressive_velocity4_channel",
    # "OneStage_30dropout34_lastEpoch_random_one_layer_whole_network_all0.33_progressive_velocity4_channel",
    # "OneStage_60dropout34_lastEpoch_random_one_layer_whole_network_all0.33_progressive_velocity4_channel",

    # "dropout1_epoch15_lr0.001_lastEpoch_random_one_layer_behind_first_layer_all0.33_progressive",
    # "dropout2_epoch15_lr0.001_lastEpoch_random_one_layer_behind_first_layer_all0.33_progressive",
    # "dropout3_epoch15_lr0.001_lastEpoch_random_one_layer_behind_first_layer_all0.33_progressive",
    # "dropout4_epoch15_lr0.001_lastEpoch_random_one_layer_behind_first_layer_all0.33_progressive",
    # "dropout12_epoch15_lr0.001_lastEpoch_random_one_layer_behind_first_layer_all0.33_progressive",
    # "dropout23_epoch15_lr0.001_lastEpoch_random_one_layer_behind_first_layer_all0.33_progressive",
    # "dropout34_epoch15_lr0.001_lastEpoch_random_one_layer_behind_first_layer_all0.33_progressive",
    # "dropout123_epoch15_lr0.001_lastEpoch_random_one_layer_behind_first_layer_all0.33_progressive",
    # "dropout234_epoch15_lr0.001_lastEpoch_random_one_layer_behind_first_layer_all0.33_progressive",
    # "dropout1234_epoch15_lr0.001_lastEpoch_random_one_layer_behind_first_layer_all0.33_progressive",

    # "dropout34_lastEpoch_random_one_layer_behind_first_layer_all0.33_progressive_velocity4_normal",
    # "dropout34_lastEpoch_random_one_layer_behind_first_layer_all0.33_progressive_velocity4_channel",

    "dropout34_lastEpoch_random_one_layer_behind_first_layer_all0.2_progressive_velocity4_channel",
    # "dropout34_lastEpoch_random_one_layer_behind_first_layer_all0.25_progressive_velocity4_channel",
    "dropout34_lastEpoch_random_one_layer_behind_first_layer_all0.33_progressive_velocity4_channel",
    # # "dropout34_lastEpoch_random_one_layer_behind_first_layer_all0.33_progressive",
    "dropout34_lastEpoch_random_one_layer_behind_first_layer_all0.4_progressive_velocity4_channel",

# "dropout34_bestEpoch_random_one_layer_behind_first_layer_all0.2_progressive_velocity4_channel",
#     "dropout34_bestEpoch_random_one_layer_behind_first_layer_all0.25_progressive_velocity4_channel",
    # "dropout34_lastEpoch_random_one_layer_behind_first_layer_all0.33_progressive_velocity4_channel",
    # "dropout34_lastEpoch_random_one_layer_behind_first_layer_all0.33_progressive",
    # "dropout34_bestEpoch_random_one_layer_behind_first_layer_all0.4_progressive_velocity4_channel",

]

for i in range(len(dropout_methods)):
    dropout_name = dropout_methods[i]

    for name, result in result_dict.items():
        result.clear()
    dropout_test_flag = 1
    # mode = "best val"
    mode = "last epoch"
    seed = 2
    for file in dirs:
        # if "RSC" in file:
        # if "RSC" == file:
        if "resnet_layer_3_randaug_m4_n8_stage1_w10.0_w20.0_radio0.5_Conv1BNReLUDropout0.0_stage2_epochs_style30_learning_rate_style0.001_lr0.001_batch64" in file:
            print("aaa")
            method_path = path + "/" + file + "/"
            # method_path = path + "/"
            # print(method_path)
            # 写在这个文件中
            for domain in sorted(os.listdir(method_path)):
                # if domain != "results.txt" and domain != "results_last_epoch.txt" and domain != "results_0.txt" \
                #         and domain != "results_last_epoch_0.txt" and (seed == -1 or (seed != -1 and str(seed) in domain)) and "4" not in domain:
                # and str(2) not in domain and str(4) not in domain
                if "results" not in domain and ((seed == -1) or (seed != -1 and str(seed) in domain)):
                    domain_name = domain[:-1]
                    print(domain_name)
                    test_result = method_path + domain + "/" + "test.txt" if dropout_test_flag == 0 \
                        else method_path + domain + "/" + dropout_name + "/" + "test.txt"
                    with open(test_result, "r") as f:
                        line = f.readlines()
                        # print(line[-1])

                        if mode == "best val":
                            # read the result of best val
                            result = line[-1].strip("\n").replace(domain_name+": ", "")
                            result = result.replace("Best val ", "").replace(", corresponding test", "").replace(" - best test:", "").replace(", best epoch:","").replace("- best test:", "")
                            results = result.split(" ")

                            test_acc = float(results[1])

                        elif mode == "last epoch":
                            # read the result of the last epoch
                            result = line[-2].strip("\n")
                            # result = result.replace("test: Epoch: ", "").replace(", CELoss:", "").replace(", ACC:", "")
                            result = result.replace("test: Epoch: ", "").replace(", layer CELoss:", "").replace(", CELoss:", "").replace(", ACC:", "")
                            results = result.split(" ")
                            test_acc = float(results[2])
                        else:
                            pass

                    result_dict[domain_name].append(test_acc)

            model_results = []
            results_line = ""
            for key, value in result_dict.items():
                domain_mean = format(np.mean(value) * 100, ".2f")
                domain_var = format(np.std(value) * 100, ".2f")
                model_results.append(value)

                results_line += key + " : " + str(domain_mean) + "+-" + str(domain_var) + "\n"
            model_results = np.mean(model_results, axis=0)
            for i, result in enumerate(model_results):
                results_line += "random seed: " + str(i) + ", model result: " + str(result) + "\n"
            model_results_mean = format(np.mean(model_results) * 100, ".2f")
            model_results_var = format(np.std(model_results) * 100, ".2f")
            results_line += "Mean" + " : " + str(model_results_mean) + "+-" + str(model_results_var) + "\n"

            if dropout_test_flag == 0:
                if mode == "best val":
                    temp_path = "/results"
                    if seed != -1:
                        temp_path += "_" + str(seed)
                    temp_path += ".txt"
                    with open(method_path + temp_path, "w") as f:
                        f.write(results_line)
                elif mode == "last epoch":
                    temp_path = "/results_last_epoch"
                    if seed != -1:
                        temp_path += "_" + str(seed)
                    temp_path += ".txt"
                    with open(method_path + temp_path, "w") as f:
                        f.write(results_line)
            else:
                if mode == "best val":
                    temp_path = "/results"
                    if seed != -1:
                        temp_path += "_" + str(seed)

                    with open(method_path + temp_path + "_" + dropout_name + ".txt", "w") as f:
                        f.write(results_line)
                elif mode == "last epoch":
                    temp_path = "/results_last_epoch"
                    if seed != -1:
                        temp_path += "_" + str(seed)
                    with open(method_path + temp_path + "_" + dropout_name + ".txt", "w") as f:
                        f.write(results_line)
