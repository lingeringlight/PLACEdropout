import torch
import cv2
import sys
import numpy as np
import matplotlib.pyplot as plt
from models.resnet_Content_Style import resnet18

class GradCam():
 """
 GradCam主要执行
 1.提取特征（调用FeatureExtractor)
 2.反向传播求目标层梯度
 3.实现目标层的CAM图
 """
 def __init__(self, model, target_layer_names):
  self.model = model

  self.extractor = FeatureExtractor(self.model, target_layer_names)

 def forward(self, input):
  return self.model(input)

 def __call__(self, input):
  features, output = self.extractor(input)  # 这里的feature 对应的就是目标层的输出， output是图像经过分类网络的输出
  output.data
  one_hot = output.max()  # 取1000个类中最大的值

  self.model.features.zero_grad()  # 梯度清零
  self.model.classifier.zero_grad()  # 梯度清零
  one_hot.backward(retain_graph=True)  # 反向传播之后，为了取得目标层梯度

  grad_val = self.extractor.get_gradients()[-1].data.numpy()
  # 调用函数get_gradients(),  得到目标层求得的梯

  target = features[-1]
  # features 目前是list 要把里面relu层的输出取出来, 也就是我们要的目标层 shape(1, 512, 14, 14)
  target = target.data.numpy()[0, :]  # (1, 512, 14, 14) > (512, 14, 14)

  weights = np.mean(grad_val, axis=(2, 3))[0, :]  # array shape (512, ) 求出relu梯度的 512层 每层权重

  cam = np.zeros(target.shape[1:])  # 做一个空白map，待会将值填上
  # (14, 14)  shape(512, 14, 14)tuple  索引[1:] 也就是从14开始开始

  # for loop的方式将平均后的权重乘上目标层的每个feature map， 并且加到刚刚生成的空白map上
  for i, w in enumerate(weights):
   cam += w * target[i, :, :]
   # w * target[i, :, :]
   # target[i, :, :] = array:shape(14, 14)
   # w = 512个的权重均值 shape(512, )
   # 每个均值分别乘上target的feature map
   # 在放到空白的14*14上（cam)
   # 最终 14*14的空白map 会被填满

  cam = cv2.resize(cam, (224, 224))  # 将14*14的featuremap 放大回224*224
  cam = cam - np.min(cam)
  cam = cam / np.max(cam)
  return cam