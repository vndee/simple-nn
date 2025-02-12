"""
Simple Neural Network in Python with OOP-style
"""

import numpy as np
from typing import Callable, List, Dict, Optional

from numpy.typing import NDArray

from func import (
    sigmoid, sigmoid_derivative,
    relu, relu_derivative,
    cross_entropy, cross_entropy_derivative
)


class Layer:
    def __init__(self, input_size: int, output_size: int, activation_function: str):
        # He initialization for ReLU, Xavier for sigmoid
        if activation_function == "relu":
            scale = np.sqrt(2.0 / input_size)
        else:  # sigmoid
            scale = np.sqrt(1.0 / input_size)
            
        self.weights = np.random.randn(output_size, input_size) * scale
        self.biases = np.zeros((output_size, 1))  # Initialize biases to zero
        self.activation_name = activation_function
        self.activation_function = self._get_activation_function(activation_function)
        self.activation_derivative = self._get_activation_derivative(activation_function)
        
        # Cache for backpropagation
        self.input: Optional[NDArray[np.float64]] = None
        self.output: Optional[NDArray[np.float64]] = None
        self.z: Optional[NDArray[np.float64]] = None

    @staticmethod
    def _get_activation_function(name: str) -> Callable[[NDArray[np.float64]], NDArray[np.float64]]:
        if name == "sigmoid":
            return sigmoid
        elif name == "relu":
            return relu
        else:
            raise ValueError(f"Activation function {name} not found")

    @staticmethod
    def _get_activation_derivative(name: str) -> Callable[[NDArray[np.float64]], NDArray[np.float64]]:
        if name == "sigmoid":
            return sigmoid_derivative
        elif name == "relu":
            return relu_derivative
        else:
            raise ValueError(f"Activation derivative {name} not found")

    def forward(self, x: NDArray[np.float64]) -> NDArray[np.float64]:
        self.input = x
        self.z = self.weights @ x + self.biases
        self.output = self.activation_function(self.z)
        return self.output
    
    def backward(self, grad_output: NDArray[np.float64]) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
        """
        Compute gradients for backpropagation
        Returns:
            - grad_input: gradient with respect to input
            - grad_weights: gradient with respect to weights
            - grad_biases: gradient with respect to biases
        """
        if self.input is None or self.z is None:
            raise ValueError("Forward pass must be called before backward pass")
        
        grad_activation = grad_output * self.activation_derivative(self.z)
        grad_weights = grad_activation @ self.input.T
        grad_biases = np.sum(grad_activation, axis=1, keepdims=True)
        grad_input = self.weights.T @ grad_activation
        
        return grad_input, grad_weights, grad_biases


class NeuralNetwork:
    def __init__(
        self,
        layers: List[tuple[int, int, str]],
        learning_rate: float = 0.01,
    ):
        self.layers = []
        for input_size, output_size, activation_function in layers:
            self.layers.append(Layer(input_size, output_size, activation_function))
        self.learning_rate = learning_rate

    def forward(self, x: NDArray[np.float64]) -> NDArray[np.float64]:
        """Forward pass through all layers"""
        current_output = x.T  # Convert to shape (input_size, batch_size)
        for layer in self.layers:
            current_output = layer.forward(current_output)
        return current_output

    def backward(self, x: NDArray[np.float64], y: NDArray[np.float64]) -> Dict[str, float]:
        """
        Backward pass through all layers
        Returns dictionary with loss and accuracy
        """
        batch_size = x.shape[0]
        y = y.T  # Convert to shape (output_size, batch_size)
        
        # Forward pass
        output = self.forward(x)
        
        # Calculate initial gradient from loss
        grad_output = cross_entropy_derivative(y, output)
        loss = cross_entropy(y, output) / batch_size
        
        # Backward pass through all layers
        for layer in reversed(self.layers):
            grad_input, grad_weights, grad_biases = layer.backward(grad_output)
            
            # Update weights and biases with normalization by batch size
            layer.weights -= (self.learning_rate / batch_size) * grad_weights
            layer.biases -= (self.learning_rate / batch_size) * grad_biases
            
            grad_output = grad_input
            
        # Calculate accuracy for multi-class classification
        predictions = np.argmax(output, axis=0)
        true_labels = np.argmax(y, axis=0)
        accuracy = np.mean(predictions == true_labels)
        
        return {"loss": loss, "accuracy": accuracy}

    def train(self, x: NDArray[np.float64], y: NDArray[np.float64], epochs: int = 1000, batch_size: int = 32) -> List[Dict[str, float]]:
        """Train the neural network"""
        history = []
        n_samples = x.shape[0]
        
        # Create progress bar
        epoch_range = range(epochs)
        try:
            from tqdm import tqdm
            epoch_range = tqdm(epoch_range, desc="Training")
        except ImportError:
            pass
            
        for epoch in epoch_range:
            # Shuffle the data
            indices = np.random.permutation(n_samples)
            x_shuffled = x[indices]
            y_shuffled = y[indices]
            
            epoch_metrics = {"loss": 0.0, "accuracy": 0.0}
            n_batches = 0
            
            # Mini-batch training
            for i in range(0, n_samples, batch_size):
                batch_x = x_shuffled[i:i + batch_size]
                batch_y = y_shuffled[i:i + batch_size]
                
                if batch_x.shape[0] < 2:  # Skip last batch if too small
                    continue
                    
                metrics = self.backward(batch_x, batch_y)
                epoch_metrics["loss"] += metrics["loss"]
                epoch_metrics["accuracy"] += metrics["accuracy"]
                n_batches += 1
            
            # Average metrics over batches
            if n_batches > 0:
                epoch_metrics["loss"] /= n_batches
                epoch_metrics["accuracy"] /= n_batches
                
                history.append(epoch_metrics)
            
        return history

    def predict(self, x: NDArray[np.float64]) -> NDArray[np.float64]:
        """Make predictions"""
        output = self.forward(x)
        return output.T  # Convert back to (batch_size, output_size)

    def evaluate(self, x: NDArray[np.float64], y: NDArray[np.float64]) -> Dict[str, float]:
        """Evaluate the model on test data"""
        output = self.forward(x)
        y = y.T
        
        loss = cross_entropy(y, output) / x.shape[0]
        predictions = np.argmax(output, axis=0)
        true_labels = np.argmax(y, axis=0)
        accuracy = np.mean(predictions == true_labels)
        
        return {"loss": loss, "accuracy": accuracy}


if __name__ == "__main__":
    # Example: XOR problem
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y = np.array([[1, 0], [0, 1], [0, 1], [1, 0]])  # One-hot encoded
    
    nn = NeuralNetwork(
        layers=[
            (2, 4, "relu"),     # Input layer -> Hidden layer
            (4, 2, "sigmoid")   # Hidden layer -> Output layer
        ],
        learning_rate=0.1
    )
    
    history = nn.train(X, y, epochs=1000, batch_size=4)
    for epoch, epoch_metrics in enumerate(history):
        print(f"Epoch {epoch + 1}: Loss={epoch_metrics['loss']:.4f}, Accuracy={epoch_metrics['accuracy']:.4f}")

    # Test predictions
    predictions = nn.predict(X)
    print("\nFinal predictions:")
    for input_x, true_y, pred_y in zip(X, y, predictions):
        print(f"Input: {input_x}, True: {np.argmax(true_y)}, Predicted: {np.argmax(pred_y)}")
