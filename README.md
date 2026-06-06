# Building a Neural Network Framework From Scratch (NumPy)

A modular, readable deep learning framework built entirely from scratch using only NumPy. Instead of using pre-made tools like PyTorch or TensorFlow, this project builds the underlying mathematical layers, error-tracking, and optimization systems completely from first principles.

## 🎯 What This Project Does

This framework solves a classic machine learning problem: classifying data that cannot be separated by straight lines. 

The code generates a non-linear 2D spiral dataset containing three distinct classes. A standard linear equation cannot solve this. Our custom network processes the coordinates, automatically learns the curves of the dataset, and creates a visual report (`model_performance_report.png`) mapping out its learned decision boundaries.

## 🧱 The Code

The project is split into isolated, reusable blocks so you can add new features without breaking the core engine:

* **`src/base.py`**: The master blueprint. It forces every layer we create to handle data moving forward (predictions) and backward (learning).
* **`src/layers.py`**: The structural components. Contains the `Dense` layer (which connects every input to every output using scaled random weights) and the `ReLU` layer (which acts as an on/off switch to handle complex curves).
* **`src/losses.py`**: The scoring system. It measures how wrong the network's guesses are and scales down giant numbers so the computer's memory doesn't overflow.
* **`src/optimizers.py`**: The teacher. It uses the Adam optimization algorithm to track past mistakes, smooth out noisy updates, and adjust the weights in the right direction.
* **`data/pipeline.py`**: The delivery truck. It slices the dataset into small batches for the network to read, managing computer memory efficiently.

## 🚀 Quick Start Guide

### 1. Set Up the Environment
Clone the repository and set up a clean, isolated Python environment:
```bash
git clone [https://github.com/Robochampion5/numpy-deep-learning.git](https://github.com/Robochampion5/numpy-deep-learning.git)
cd numpy-deep-learning
python3 -m venv .venv
source .venv/bin/activate  # On Windows use: .venv\Scripts\activate
pip install -r requirements.txt# Neural_Network_visualizer
