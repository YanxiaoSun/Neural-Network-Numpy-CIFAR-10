"""
可视化
1. 绘制各层参数的直方图
2. 绘制各层参数的热力图
"""
import os
import sys
import json
from src.ThreeLayerNN import ThreeLayerNN


import matplotlib.pyplot as plt

import src.Layers as L


if not os.path.exists("images"):
    os.makedirs("images")

ckpt_path = "models_2/model_epoch_35.pkl"
nn_architecture = json.load(open(ckpt_path.replace(".pkl", ".json"), "r"))

layer_sizes = [3072, 256, 64, 10]
activation_functions = ["relu", "relu", "softmax"]
model = ThreeLayerNN(layer_sizes, activation_functions) 
for i, layer in enumerate(model.layers):
    if isinstance(layer, L.Linear):
        plt.figure()
        plt.hist(layer.W.flatten(), bins=100)
        plt.title(f"Layer {i + 1} Weight Distribution")
        plt.savefig(f"images/layer_{i + 1}_weight_distribution_init.png")

        plt.figure()
        plt.imshow(layer.W, cmap="hot", interpolation="nearest")
        plt.title(f"Layer {i + 1} Weight Matrix")
        plt.colorbar()
        plt.savefig(f"images/layer_{i + 1}_weight_matrix_init.png")

model.load_model_dict(path=ckpt_path)
for i, layer in enumerate(model.layers):
    if isinstance(layer, L.Linear):
        plt.figure()
        plt.hist(layer.W.flatten(), bins=100)
        plt.title(f"Layer {i + 1} Weight Distribution")
        plt.savefig(f"images/layer_{i + 1}_weight_distribution.png")

        plt.figure()
        plt.imshow(layer.W, cmap="hot", interpolation="nearest")
        plt.title(f"Layer {i + 1} Weight Matrix")
        plt.colorbar()
        plt.savefig(f"images/layer_{i + 1}_weight_matrix.png")
