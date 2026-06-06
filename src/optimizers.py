import numpy as np
from typing import List
from src.base import Layer

class Adam:
    def __init__(self, layers: List[Layer], lr: float = 0.001, beta1: float = 0.9, beta2: float = 0.999, eps: float = 1e-8) -> None:
        # Filters network graph to track only trainable layers
        self.layers = [layer for layer in layers if layer.trainable]
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        # Initializes time step counter and moment tracking maps
        self.t = 0
        self.m: dict = {}
        self.v: dict = {}
        
        # Allocates zeroed tracking arrays using layer object IDs
        for layer in self.layers:
            self.m[id(layer)] = [np.zeros_like(p) for p in layer.params]
            self.v[id(layer)] = [np.zeros_like(p) for p in layer.params]

    def step(self) -> None:
        # Increments current optimization training step
        self.t += 1
        for layer in self.layers:
            layer_id = id(layer)
            for i in range(len(layer.params)):
                # Updates running mean of past gradients (momentum)
                self.m[layer_id][i] = self.beta1 * self.m[layer_id][i] + (1 - self.beta1) * layer.grads[i]
                # Updates running variance of gradients (adaptive scaling)
                self.v[layer_id][i] = self.beta2 * self.v[layer_id][i] + (1 - self.beta2) * (layer.grads[i] ** 2)
                
                # Computes bias adjustments for early training steps
                m_hat = self.m[layer_id][i] / (1 - self.beta1 ** self.t)
                v_hat = self.v[layer_id][i] / (1 - self.beta2 ** self.t)
                
                # Adjusts network parameters using scaled step sizes
                layer.params[i] -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)