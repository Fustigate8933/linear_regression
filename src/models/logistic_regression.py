import numpy as np
from src.models.base import BaseModel

class LogisticRegression(BaseModel):
    # def __init__(self, num_features: int, reg_lambda: float = 0.0):
    #     super().__init__(num_features, reg_lambda)
    #
    def forward(self, X: np.ndarray):
        """
        Compute predictions.
        """
        linear_output = X @ self.weights.T + self.bias  # (n_samples, 1)
        y_pred = self.sigmoid(linear_output)
        return y_pred.reshape(-1) # (n_samples,)

    def predict(self, X: np.ndarray):
        """
        same as forward
        """
        return self.forward(X)

    def compute_loss(self, y_pred: np.ndarray, y_true: np.ndarray):
        """
        binary cross-entropy loss with L2 regularization.
        """
        n = y_true.shape[0]
        # clip predictions to avoid log(0)
        y_pred_clipped = np.clip(y_pred, 1e-15, 1 - 1e-15)
        # binary cross-entropy loss
        bce_loss = - (1/n) * np.sum(y_true * np.log(y_pred_clipped) + (1 - y_true) * np.log(1 - y_pred_clipped))
        # L2 regularization
        l2_reg = self.reg_lambda * np.sum(self.weights ** 2)
        total_loss = bce_loss + l2_reg
        return total_loss

    def backward(self, X: np.ndarray, y_pred: np.ndarray, y_true: np.ndarray) -> None:
        """
        Compute gradients.
        """
        n = y_true.shape[0]
        error = y_pred - y_true  # (n_samples,)

        self.grad_w = (1/n) * (error.reshape(-1, 1) * X).sum(axis=0).reshape(1, -1) + 2 * self.reg_lambda * self.weights
        self.grad_b = (1/n) * np.sum(error)

    def step(self, lr: float) -> None:
        """
        Update parameters using gradients.
        """
        self.weights -= lr * self.grad_w
        self.bias -= lr * self.grad_b

    def sigmoid(self, z: np.ndarray) -> np.ndarray:
        """
        Sigmoid activation function.
        """
        return 1 / (1 + np.exp(-z))
