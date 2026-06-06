import unittest
import numpy as np
from src.layers import Dense, ReLU

class TestNetworkDimensions(unittest.TestCase):
    def test_dense_forward_shape(self):
        # Verifies that input dimensions properly map to configured output features
        batch_size = 32
        in_features = 10
        out_features = 5
        
        layer = Dense(in_features, out_features)
        dummy_input = np.random.randn(batch_size, in_features)
        output = layer.forward(dummy_input)
        
        self.assertEqual(output.shape, (batch_size, out_features))

    def test_dense_backward_shape(self):
        # Verifies that backpropagated gradients match incoming input coordinates
        batch_size = 32
        in_features = 10
        out_features = 5
        
        layer = Dense(in_features, out_features)
        dummy_input = np.random.randn(batch_size, in_features)
        _ = layer.forward(dummy_input)
        
        dummy_gradient = np.random.randn(batch_size, out_features)
        input_gradient = layer.backward(dummy_gradient)
        
        self.assertEqual(input_gradient.shape, dummy_input.shape)

if __name__ == "__main__":
    unittest.main()