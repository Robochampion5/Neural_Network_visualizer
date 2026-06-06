import matplotlib.pyplot as plt
import numpy as np
from typing import List
from src.base import Layer

def plot_training_results(X: np.ndarray, Y: np.ndarray, layers: List[Layer], loss_history: List[float]) -> None:
    # Initializes side-by-side subplot tracking figures
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(14, 6))
    
    # Generates loss convergence profile line plot
    axes[0].plot(loss_history, color='#2c3e50', linewidth=2.5)
    axes[0].set_title("Loss Convergence Tracker", fontsize=12, fontweight='bold')
    axes[0].set_xlabel("Epoch Iterations")
    axes[0].set_ylabel("Cross-Entropy Loss")
    axes[0].grid(True, linestyle='--', alpha=0.6)

    # Defines evaluation space boundary plot ranges
    axes[1].set_title("Learned Decision Boundaries", fontsize=12, fontweight='bold')
    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
    # Builds standard 2D validation coordinates grid
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02), np.arange(y_min, y_max, 0.02))
    
    # Flattens grid data to pass through network layers
    grid_data = np.c_[xx.ravel(), yy.ravel()]
    out = grid_data
    for layer in layers:
        out = layer.forward(out)
        
    # Maps top predicted classes back to 2D grid dimensions
    predictions = np.argmax(out, axis=1).reshape(xx.shape)
    axes[1].contourf(xx, yy, predictions, cmap='Spectral', alpha=0.3)
    
    # Renders original data scatter markers over the background colors
    labels_idx = np.argmax(Y, axis=1)
    scatter = axes[1].scatter(X[:, 0], X[:, 1], c=labels_idx, cmap='Spectral', edgecolors='k', s=40)
    
    # Configures chart spacing layouts and saves static output image
    fig.colorbar(scatter, ax=axes[1])
    plt.tight_layout()
    plt.savefig("model_performance_report.png", dpi=150)
    plt.show()