import numpy as np
from typing import List

# Base class that defines the structure for all network layers
class Layer:
    def __init__(self) -> None:
        # Tracks if the layer updates its weights during training
        self.trainable: bool = False
        # Stores weight and bias arrays for the layer
        self.params: List[np.ndarray] = []
        # Stores calculated gradients for parameters
        self.grads: List[np.ndarray] = []

    # Forces child layers to implement their own forward pass
    def forward(self, inputs: np.ndarray) -> np.ndarray:
        raise NotImplementedError()

    # Forces child layers to implement their own backward pass
    def backward(self, output_gradient: np.ndarray) -> np.ndarray:
        raise NotImplementedError()