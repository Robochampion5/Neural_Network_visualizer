# Neural Network Visualizer

A clean, from-scratch neural network implementation in NumPy that learns to classify a 2D spiral dataset and visualizes the learned decision boundaries.

## Overview

This project demonstrates a simple deep learning pipeline built without PyTorch or TensorFlow. It includes:

- A custom `Dense` neural layer with trainable weights and biases
- The `ReLU` activation function
- A combined softmax + cross-entropy loss function
- The Adam optimizer
- A streaming batch data pipeline
- Training progress visualization and decision boundary plotting

## What the Project Does

The `main.py` script:

1. Generates a synthetic 2D spiral dataset with 3 classes
2. Splits the data into training and test sets
3. Builds a small neural network:
   - `Dense(2, 64)`
   - `ReLU()`
   - `Dense(64, 3)`
4. Trains the model using mini-batch gradient descent
5. Evaluates the model on unseen test data
6. Saves and displays a plot showing:
   - training loss over epochs
   - decision regions learned by the network

## Repository Structure

- `main.py` — data generation, model training loop, evaluation, and visualization trigger
- `visualize.py` — training loss plot and decision boundary visualization
- `requirements.txt` — required Python packages
- `data/pipeline.py` — batch streaming utility for training
- `src/base.py` — base `Layer` class and shared layer structure
- `src/layers.py` — `Dense` and `ReLU` layer implementations
- `src/losses.py` — `SoftmaxCrossEntropy` loss implementation
- `src/optimizers.py` — `Adam` optimizer implementation

## Installation

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Usage

Run the training script with:

```bash
python3 main.py
```

After training, the script will generate `model_performance_report.png` containing the loss curve and decision boundary plot.

## Key Concepts

- **Dense layer**: matrix multiplication plus bias
- **ReLU activation**: zeroes negative values to introduce non-linearity
- **Softmax + Cross-Entropy**: converts logits to probabilities and computes classification loss
- **Adam optimizer**: adaptive optimization with momentum and variance scaling
- **Mini-batch training**: processes data in batches for stable learning

## Requirements

- Python 3.8+
- NumPy
- Matplotlib

Install with:

```bash
pip install -r requirements.txt
```

## Notes

- The dataset is synthetic and designed to demonstrate non-linear decision boundaries.
- The visualization helps confirm that the network is learning meaningful class separation.
- The code is intentionally modular to make it easy to extend with new layers, losses, or optimizers.

## License

This project is provided as-is for learning and experimentation.
