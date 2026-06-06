import numpy as np
from typing import List
from src.base import Layer

class Dense(Layer):
    def __init__(self, in_features: int, out_features: int) -> None:
        super().__init__()
        self.trainable = True
        
        # Scales random weights to prevent gradient explosion
        limit = np.sqrt(2.0 / in_features)
        self.weights = np.random.randn(in_features, out_features) * limit
        # Initializes biases to zero for each output feature
        self.biases = np.zeros((1, out_features))
        
        # Groups parameters and allocates zeroed gradient buffers
        self.params = [self.weights, self.biases]
        self.grads = [np.zeros_like(self.weights), np.zeros_like(self.biases)]
        # Placeholder array to cache input data during forward pass
        self.inputs: np.ndarray = np.empty((0,))

    def forward(self, inputs: np.ndarray) -> np.ndarray:
        # Ensures incoming data is a 2D batch matrix
        assert inputs.ndim == 2
        self.inputs = inputs
        # Computes linear combination of inputs, weights, and biases
        return (inputs @ self.weights) + self.biases

    def backward(self, output_gradient: np.ndarray) -> np.ndarray:
        # Computes weight gradients using input matrix transposition
        self.grads[0] = self.inputs.T @ output_gradient
        # Sums error gradients across the batch for biases
        self.grads[1] = np.sum(output_gradient, axis=0, keepdims=True)
        
        # Passes input error gradient back to the previous layer
        return output_gradient @ self.weights.T

class ReLU(Layer):
    def __init__(self) -> None:
        super().__init__()
        # Placeholder array to cache activation states
        self.activated_states: np.ndarray = np.empty((0,))

    def forward(self, inputs: np.ndarray) -> np.ndarray:
        # Replaces all negative input values with zero
        self.activated_states = np.maximum(0, inputs)
        return self.activated_states

    def backward(self, output_gradient: np.ndarray) -> np.ndarray:
        # Zeroes out gradients where the forward input was negative
        return output_gradient * (self.activated_states > 0)