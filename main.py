import numpy as np
from typing import Tuple
from src.layers import Dense, ReLU
from src.losses import SoftmaxCrossEntropy
from src.optimizers import Adam
from data.pipeline import stream_batches
from visualize import plot_training_results

def generate_spiral_data(samples_per_class: int = 300, classes: int = 3) -> Tuple[np.ndarray, np.ndarray]:
    # Establishes data scaling variables and structures empty placeholder arrays.
    N = samples_per_class
    D = 2
    K = classes
    X = np.zeros((N * K, D))
    y = np.zeros(N * K, dtype='uint8')
    
    for j in range(K):
        # Calculates structural indexing intervals for each unique branch arm.
        ix = range(N * j, N * (j + 1))
        r = np.linspace(0.0, 1, N)
        t = np.linspace(j * 4, (j + 1) * 4, N) + np.random.randn(N) * 0.2
        # Merges mathematical sine and cosine array calculations into our coordinate tracking matrix columns.
        X[ix] = np.c_[r * np.sin(t), r * np.cos(t)]
        y[ix] = j
    
    # Instantiates a clean zeroed target classification row matrix.
    one_hot_targets = np.zeros((N * K, K))
    # Places binary flag markers (1s) at specific class index columns for every row.
    one_hot_targets[np.arange(N * K), y] = 1
    return X, one_hot_targets

def main() -> None:
    # Anchors random engine states and creates data arrays
    np.random.seed(42)
    X_all, Y_all = generate_spiral_data()

    # CRITICAL HUMAN PATTERN: Manual train/test partitioning (80% train, 20% test)
    num_samples = X_all.shape[0]
    indices = np.arange(num_samples)
    np.random.shuffle(indices)
    
    split_idx = int(num_samples * 0.8)
    train_idx, test_idx = indices[:split_idx], indices[split_idx:]
    
    X_train, Y_train = X_all[train_idx], Y_all[train_idx]
    X_test, Y_test = X_all[test_idx], Y_all[test_idx]

    # Constructs sequential execution layer sequence objects
    layers = [
        Dense(in_features=2, out_features=64),
        ReLU(),
        Dense(in_features=64, out_features=3)
    ]
    # Initializes objective metrics loss and optimization engines
    loss_fn = SoftmaxCrossEntropy()
    optimizer = Adam(layers, lr=0.01)
    loss_history = []
    
    # Starts standard network epoch training iteration loops
    for epoch in range(150):
        epoch_loss = 0.0
        batches = 0
        
        # Streams batch variations sequentially through network topology
        for batch_X, batch_Y in stream_batches(X_train, Y_train, batch_size=64, shuffle=True):
            out = batch_X
            for layer in layers:
                # Routes data matrices forward through layer node functions sequentially.
                out = layer.forward(out)
                
            # Computes evaluation scores for data variations and aggregates loss trackers.
            loss = loss_fn.forward(out, batch_Y)
            epoch_loss += loss
            batches += 1
            
            # Extracts error derivatives and runs backpropagation sequences in inverse layer order.
            loss_grad = loss_fn.backward()
            for layer in reversed(layers):
                loss_grad = layer.backward(loss_grad)
                
            # Runs parameter value tuning step adjustments
            optimizer.step()
            
        # Appends mean epoch results and logs checkpoint status
        avg_loss = epoch_loss / batches
        loss_history.append(avg_loss)
        if epoch % 15 == 0:
            print(f"Epoch {epoch:03d} | Train Loss: {avg_loss:.5f}")

    # CRITICAL HUMAN PATTERN: Evaluate performance on completely unseen test data
    test_out = X_test
    for layer in layers:
        test_out = layer.forward(test_out)
    final_test_loss = loss_fn.forward(test_out, Y_test)
    print(f"\n[FINAL EVALUATION] Unseen Test Dataset Loss: {final_test_loss:.5f}")

    # Fires chart generation engines to verify accuracy models
    plot_training_results(X_all, Y_all, layers, loss_history)

if __name__ == "__main__":
    main()