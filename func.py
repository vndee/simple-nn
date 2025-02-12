import numpy as np

from numpy.typing import NDArray

def sigmoid(x: NDArray[np.float64]) -> NDArray[np.float64]:
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x: NDArray[np.float64]) -> NDArray[np.float64]:
    return sigmoid(x) * (1 - sigmoid(x))

def relu(x: NDArray[np.float64]) -> NDArray[np.float64]:
    return np.maximum(0, x)

def relu_derivative(x: NDArray[np.float64]) -> NDArray[np.float64]:
    return np.where(x > 0, 1, 0)

def softmax(x: NDArray[np.float64]) -> NDArray[np.float64]:
    return np.exp(x) / np.sum(np.exp(x), axis=1, keepdims=True)

def softmax_derivative(x: NDArray[np.float64]) -> NDArray[np.float64]:
    return softmax(x) * (1 - softmax(x))

def cross_entropy(y_true: NDArray[np.float64], y_pred: NDArray[np.float64]) -> float:
    return -np.sum(y_true * np.log(y_pred))

def cross_entropy_derivative(y_true: NDArray[np.float64], y_pred: NDArray[np.float64]) -> NDArray[np.float64]:
    return y_pred - y_true

def mse(y_true: NDArray[np.float64], y_pred: NDArray[np.float64]) -> float:
    return np.mean((y_true - y_pred) ** 2)

def mse_derivative(y_true: NDArray[np.float64], y_pred: NDArray[np.float64]) -> NDArray[np.float64]:
    return 2 * (y_pred - y_true) / len(y_true)

