import numpy as np
from typing import Generator, Tuple

def stream_batches(X: np.ndarray, y: np.ndarray, batch_size: int, shuffle: bool = True) -> Generator[Tuple[np.ndarray, np.ndarray], None, None]:
    # Creates an ordered index sequence matching data length
    num_samples = X.shape[0]
    indices = np.arange(num_samples)
    
    # Shuffles index positions randomly without copying raw datasets
    if shuffle:
        np.random.shuffle(indices)
        
    # Loops through dataset indices in chunks of batch size
    for i in range(0, num_samples, batch_size):
        batch_idx = indices[i:i + batch_size]
        # Yields temporary index slices directly from data arrays
        yield X[batch_idx], y[batch_idx]