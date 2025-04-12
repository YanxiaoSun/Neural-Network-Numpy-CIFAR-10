import json
import pickle

import numpy as np

import src.Layers as L


class ThreeLayerNN:
    def __init__(self, layer_sizes, activation_functions):
        self.layer_sizes = layer_sizes
        self.activation_functions = activation_functions
        self.layers = []
        for i in range(len(layer_sizes) - 1):
            self.layers.append(L.Linear(layer_sizes[i], layer_sizes[i+1]))
            self.layers.append(L.Activation(activation_functions[i]))

    def forward(self, x):
        for layer in self.layers:
            x = layer.forward(x)
        return x

    def backward(self, grad):
        for layer in reversed(self.layers):
            if isinstance(layer, L.Linear):
                dW, db, grad = layer.backward(grad, layer.input_cache)
                layer.dW = dW
                layer.db = db
            elif isinstance(layer, L.Activation):
                grad = layer.backward(grad, layer.input_cache)
        return grad

    def predict(self, x):
        return np.argmax(self.forward(x), axis=1)

    def deep_copy(self):
        model_copy = ThreeLayerNN(self.layer_sizes, self.activation_functions)
        for i, layer in enumerate(self.layers):
            if isinstance(layer, L.Linear):
                model_copy.layers[i].W = layer.W.copy()
                model_copy.layers[i].b = layer.b.copy()
        return model_copy

    def save_model_dict(self, path):
        model_dict = {}
        for i, layer in enumerate(self.layers):
            if isinstance(layer, L.Linear):
                model_dict[f"layer_{i}_W"] = layer.W
                model_dict[f"layer_{i}_b"] = layer.b
        with open(path, "wb") as f:
            pickle.dump(model_dict, f)
        model_json = {}
        model_json["layer_sizes"] = self.layer_sizes
        model_json["activation_functions"] = self.activation_functions
        with open(path.replace(".pkl", ".json"), "w") as f:
            json.dump(model_json, f)

    def load_model_dict(self, path):
        with open(path.replace(".pkl", ".json"), "r") as f:
            model_json = json.load(f)
        layer_sizes = model_json["layer_sizes"]
        activation_functions = model_json["activation_functions"]   
        self.__init__(layer_sizes, activation_functions)
        with open(path, "rb") as f:
            model_dict = pickle.load(f)
        for i, layer in enumerate(self.layers):
            if isinstance(layer, L.Linear):
                layer.W = model_dict[f"layer_{i}_W"]
                layer.b = model_dict[f"layer_{i}_b"]
                layer.zero_grad()
