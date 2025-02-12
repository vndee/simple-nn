"""
Simple Neural Network in Python with Functional Programming style
"""

import numpy as np
from typing import Callable, List, Dict, Tuple, NamedTuple
from dataclasses import dataclass
from numpy.typing import NDArray

from func import (
    sigmoid, sigmoid_derivative,
    relu, relu_derivative,
    cross_entropy, cross_entropy_derivative
)


@dataclass(frozen=True)
class LayerParams:
    """Immutable layer parameters"""
    weights: NDArray[np.float64]
    biases: NDArray[np.float64]
    activation_name: str


@dataclass(frozen=True)
class LayerCache:
    """Cache for layer computations during forward pass"""
    input: NDArray[np.float64]
    z: NDArray[np.float64]
    output: NDArray[np.float64]


@dataclass(frozen=True)
class NetworkParams:
    """Immutable network parameters"""
    layers: List[LayerParams]
    learning_rate: float


@dataclass(frozen=True)
class TrainingMetrics:
    """Training metrics for one epoch"""
    loss: float
    accuracy: float


def get_activation_function(name: str) -> Callable[[NDArray[np.float64]], NDArray[np.float64]]:
    """Get activation function by name"""
    if name == "sigmoid":
        return sigmoid
    elif name == "relu":
        return relu
    else:
        raise ValueError(f"Activation function {name} not found")


def get_activation_derivative(name: str) -> Callable[[NDArray[np.float64]], NDArray[np.float64]]:
    """Get activation function derivative by name"""
    if name == "sigmoid":
        return sigmoid_derivative
    elif name == "relu":
        return relu_derivative
    else:
        raise ValueError(f"Activation derivative {name} not found")


def init_layer_params(input_size: int, output_size: int, activation_name: str) -> LayerParams:
    """Initialize layer parameters"""
    # He initialization for ReLU, Xavier for sigmoid
    if activation_name == "relu":
        scale = np.sqrt(2.0 / input_size)
    else:  # sigmoid
        scale = np.sqrt(1.0 / input_size)
        
    weights = np.random.randn(output_size, input_size) * scale
    biases = np.zeros((output_size, 1))
    
    return LayerParams(weights=weights, biases=biases, activation_name=activation_name)


def init_network_params(layer_sizes: List[tuple[int, int, str]], learning_rate: float) -> NetworkParams:
    """Initialize network parameters"""
    layers = [
        init_layer_params(input_size, output_size, activation_name)
        for input_size, output_size, activation_name in layer_sizes
    ]
    return NetworkParams(layers=layers, learning_rate=learning_rate)


def forward_layer(
    x: NDArray[np.float64],
    params: LayerParams
) -> LayerCache:
    """Forward pass through one layer"""
    z = params.weights @ x + params.biases
    output = get_activation_function(params.activation_name)(z)
    return LayerCache(input=x, z=z, output=output)


def forward_network(
    x: NDArray[np.float64],
    params: NetworkParams
) -> Tuple[NDArray[np.float64], List[LayerCache]]:
    """Forward pass through the network"""
    current_output = x.T  # Convert to shape (input_size, batch_size)
    layer_caches = []
    
    for layer_params in params.layers:
        cache = forward_layer(current_output, layer_params)
        layer_caches.append(cache)
        current_output = cache.output
    
    return current_output, layer_caches


def backward_layer(
    grad_output: NDArray[np.float64],
    params: LayerParams,
    cache: LayerCache,
    learning_rate: float,
    batch_size: int
) -> Tuple[NDArray[np.float64], LayerParams]:
    """Backward pass through one layer"""
    # Compute gradients
    grad_activation = grad_output * get_activation_derivative(params.activation_name)(cache.z)
    grad_weights = grad_activation @ cache.input.T
    grad_biases = np.sum(grad_activation, axis=1, keepdims=True)
    grad_input = params.weights.T @ grad_activation
    
    # Update parameters
    new_weights = params.weights - (learning_rate / batch_size) * grad_weights
    new_biases = params.biases - (learning_rate / batch_size) * grad_biases
    new_params = LayerParams(
        weights=new_weights,
        biases=new_biases,
        activation_name=params.activation_name
    )
    
    return grad_input, new_params


def backward_network(
    x: NDArray[np.float64],
    y: NDArray[np.float64],
    params: NetworkParams,
) -> Tuple[TrainingMetrics, NetworkParams]:
    """Backward pass through the network"""
    batch_size = x.shape[0]
    y = y.T  # Convert to shape (output_size, batch_size)
    
    # Forward pass
    output, layer_caches = forward_network(x, params)
    
    # Calculate loss and initial gradient
    loss = cross_entropy(y, output) / batch_size
    grad_output = cross_entropy_derivative(y, output)
    
    # Backward pass through all layers
    new_layers = []
    for layer_params, cache in zip(reversed(params.layers), reversed(layer_caches)):
        grad_output, new_layer_params = backward_layer(
            grad_output, layer_params, cache,
            params.learning_rate, batch_size
        )
        new_layers.insert(0, new_layer_params)
    
    # Calculate accuracy
    predictions = np.argmax(output, axis=0)
    true_labels = np.argmax(y, axis=0)
    accuracy = np.mean(predictions == true_labels)
    
    new_params = NetworkParams(
        layers=new_layers,
        learning_rate=params.learning_rate
    )
    metrics = TrainingMetrics(loss=loss, accuracy=accuracy)
    
    return metrics, new_params


def train_network(
    x: NDArray[np.float64],
    y: NDArray[np.float64],
    params: NetworkParams,
    epochs: int = 1000,
    batch_size: int = 32
) -> Tuple[List[TrainingMetrics], NetworkParams]:
    """Train the network"""
    history = []
    current_params = params
    n_samples = x.shape[0]
    
    # Create progress bar
    epoch_range = range(epochs)
    try:
        from tqdm import tqdm
        epoch_range = tqdm(epoch_range, desc="Training")
    except ImportError:
        pass
    
    for _ in epoch_range:
        # Shuffle the data
        indices = np.random.permutation(n_samples)
        x_shuffled = x[indices]
        y_shuffled = y[indices]
        
        epoch_metrics = TrainingMetrics(loss=0.0, accuracy=0.0)
        n_batches = 0
        
        # Mini-batch training
        for i in range(0, n_samples, batch_size):
            batch_x = x_shuffled[i:i + batch_size]
            batch_y = y_shuffled[i:i + batch_size]
            
            if batch_x.shape[0] < 2:  # Skip last batch if too small
                continue
            
            metrics, current_params = backward_network(batch_x, batch_y, current_params)
            epoch_metrics = TrainingMetrics(
                loss=epoch_metrics.loss + metrics.loss,
                accuracy=epoch_metrics.accuracy + metrics.accuracy
            )
            n_batches += 1
        
        # Average metrics over batches
        if n_batches > 0:
            epoch_metrics = TrainingMetrics(
                loss=epoch_metrics.loss / n_batches,
                accuracy=epoch_metrics.accuracy / n_batches
            )
            history.append(epoch_metrics)
    
    return history, current_params


def predict(
    x: NDArray[np.float64],
    params: NetworkParams
) -> NDArray[np.float64]:
    """Make predictions"""
    output, _ = forward_network(x, params)
    return output.T


def evaluate(
    x: NDArray[np.float64],
    y: NDArray[np.float64],
    params: NetworkParams
) -> TrainingMetrics:
    """Evaluate the model"""
    output, _ = forward_network(x, params)
    y = y.T
    
    # Calculate metrics
    loss = cross_entropy(y, output) / x.shape[0]
    predictions = np.argmax(output, axis=0)
    true_labels = np.argmax(y, axis=0)
    accuracy = np.mean(predictions == true_labels)
    
    return TrainingMetrics(loss=loss, accuracy=accuracy)


if __name__ == "__main__":
    # Example: XOR problem
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y = np.array([[1, 0], [0, 1], [0, 1], [1, 0]])  # One-hot encoded
    
    # Initialize network
    network_params = init_network_params(
        layer_sizes=[
            (2, 4, "relu"),     # Input layer -> Hidden layer
            (4, 2, "sigmoid")   # Hidden layer -> Output layer
        ],
        learning_rate=0.1
    )
    
    # Train network
    history, final_params = train_network(X, y, network_params, epochs=1000, batch_size=4)
    
    # Print training history
    for epoch, metrics in enumerate(history):
        print(f"Epoch {epoch + 1}: Loss={metrics.loss:.4f}, Accuracy={metrics.accuracy:.4f}")
    
    # Test predictions
    predictions = predict(X, final_params)
    print("\nFinal predictions:")
    for input_x, true_y, pred_y in zip(X, y, predictions):
        print(f"Input: {input_x}, True: {np.argmax(true_y)}, Predicted: {np.argmax(pred_y)}") 