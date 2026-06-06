import numpy as np

class SoftmaxCrossEntropy:
    def __init__(self) -> None:
        # Placeholders for network probabilities and target labels
        self.probabilities: np.ndarray = np.empty((0,))
        self.targets: np.ndarray = np.empty((0,))

    def forward(self, logits: np.ndarray, targets: np.ndarray) -> float:
        self.targets = targets
        
        # Subtracts max value per row to prevent exponential overflow
        shifted_logits = logits - np.max(logits, axis=-1, keepdims=True)
        exp_logits = np.exp(shifted_logits)
        # Divides exponentials by their sum to get probabilities
        self.probabilities = exp_logits / np.sum(exp_logits, axis=-1, keepdims=True)
        
        # Clips values slightly away from 0 and 1 to protect log math
        eps = 1e-15
        clipped_probs = np.clip(self.probabilities, eps, 1.0 - eps)
        # Calculates mean cross-entropy loss for the batch
        batch_loss = -np.sum(targets * np.log(clipped_probs)) / logits.shape[0]
        return float(batch_loss)

    def backward(self) -> np.ndarray:
        # Gets total batch size to normalize gradients
        batch_size = self.targets.shape[0]
        # Returns simplified loss gradient w.r.t linear inputs
        return (self.probabilities - self.targets) / batch_size