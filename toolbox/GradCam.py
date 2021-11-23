import torch
import cv2
import sys
import numpy as np
import matplotlib.pyplot as plt
from models.resnet_Content_Style import resnet18

class GradCam():
 def __init__(self, model, target_layer_names):
  self.model = model

  self.extractor = FeatureExtractor(self.model, target_layer_names)

 def forward(self, input):
  return self.model(input)

 def __call__(self, input):
  features, output = self.extractor(input)
  output.data
  one_hot = output.max()

  self.model.features.zero_grad()
  self.model.classifier.zero_grad()
  one_hot.backward(retain_graph=True)

  grad_val = self.extractor.get_gradients()[-1].data.numpy()

  target = features[-1]
  target = target.data.numpy()[0, :]  # (1, 512, 14, 14) > (512, 14, 14)

  weights = np.mean(grad_val, axis=(2, 3))[0, :]

  cam = np.zeros(target.shape[1:])

  for i, w in enumerate(weights):
   cam += w * target[i, :, :]
  cam = cv2.resize(cam, (224, 224))
  cam = cam - np.min(cam)
  cam = cam / np.max(cam)
  return cam