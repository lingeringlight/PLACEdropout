# PLACEdropout

## Requirements

* Python == 3.7.3
* Pytorch == 1.8.1
* Cuda == 10.1
* Torchvision == 0.4.2
* Tensorflow == 1.14.0
* GPU == RTX 2080Ti

## DataSets
Please download PACS dataset from [here](https://drive.google.com/drive/folders/0B6x7gtvErXgfUU1WcGY5SzdwZVk?resourcekey=0-2fvpQY_QSyJf2uIECzqPuQ).
Make sure you use the official train/val/test split in [PACS paper](https://openaccess.thecvf.com/content_iccv_2017/html/Li_Deeper_Broader_and_ICCV_2017_paper.html).
Take `/data/DataSets/` as the saved directory for example:
```
images -> /data/DataSets/PACS/kfold/art_painting/dog/pic_001.jpg, ...
splits -> /data/DataSets/PACS/pacs_label/art_painting_crossval_kfold.txt, ...
```
Then set the `"data_root"` as `"/data/DataSets/"` and `"data"` as `"PACS"` in `train_PLACE.py`.

## Training
For training the model, please set the `"result_path"` where the results are saved.
Then simply running the code to train a ResNet-18:
```
python train_PLACE.py --target [domain_index] --device [GPU_index]
```
The `domain_index` denotes the index of target domain, and `GPU_index` denotes the GPU device number.
```
domain_index: [0:'photo', 1:'art_painting', 2:'cartoon', 3:'sketch']
```
Or run the `PLACE.sh` directly.

## Test
To test a ResNet-18, you can download the trained model below:

Target domain  | Acc(%) | models |
:----:  | :----: | :----: |
Photo | 96.65 | download |
Art | 85.79 | download |
Cartoon | 80.20 | download |
Sketch | 83.69 | download |

