import numpy as np

class LinearRegression:
    def __init__(self, num_features: int):
        self.weights = None
        self.bias = None

    def forward(self, X: np.ndarray):
        """
        Compute predictions.
        """
        pass

    def compute_loss(self, y_pred: np.ndarray, y_true: np.ndarray):
        """
        Mean Squared Error loss.
        """
        pass

    def backward(self, X: np.ndarray, y_pred: np.ndarray, y_true: np.ndarray):
        """
        Compute gradients for weights and bias.
        """
        pass

    def step(self, lr: float):
        """
        Update parameters using gradients.
        """
        pass

