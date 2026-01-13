import numpy as np

class BaseModel:
    def __init__(self, num_features: int, reg_lambda: float = 0.0):
        self.weights = np.random.randn(1, num_features) * 0.01
        self.bias = np.random.randn(1)
        self.grad_w = np.zeros_like(self.weights)
        self.grad_b = 0.0
        self.reg_lambda = reg_lambda  # L2 regularization strength

    def forward(self, X: np.ndarray) -> np.ndarray:
        pass

    def predict(self, X: np.ndarray):
        return self.forward(X)

    def compute_loss(self, y_pred: np.ndarray, y_true: np.ndarray):
        pass

    def backward(self, X: np.ndarray, y_pred: np.ndarray, y_true: np.ndarray) -> None:
        pass

    def step(self, lr: float) -> None:
        self.weights -= lr * self.grad_w
        self.bias -= lr * self.grad_b
