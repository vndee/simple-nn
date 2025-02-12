"""
Compare performance between OOP and FP implementations of Neural Network
"""

import time
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple
from memory_profiler import memory_usage
import os
from functools import wraps

from nn_oop import NeuralNetwork as NeuralNetworkOOP
from nn_fp import (
    init_network_params,
    train_network as train_network_fp,
    evaluate as evaluate_fp,
    TrainingMetrics,
    NetworkParams
)


def measure_memory_usage(func):
    """Decorator to measure peak memory usage of a function"""
    @wraps(func)
    def wrapper(*args, **kwargs):
        # Measure memory usage and get result in a single run
        result = []
        def target():
            nonlocal result
            result = func(*args, **kwargs)
            return result

        # Get memory usage during execution
        mem_usage = memory_usage(
            (target, (), {}),
            interval=0.01,  # Smaller interval for more precise measurement
            include_children=True,  # Include child processes
            multiprocess=True,  # Handle multiprocessing
            max_usage=True,  # Get peak memory usage
        )
        
        # Get peak memory usage (max_usage=True returns just the peak)
        peak_memory = mem_usage[0] if isinstance(mem_usage, list) else mem_usage
        
        return result, peak_memory
    
    return wrapper


@measure_memory_usage
def train_oop(
    X_train: np.ndarray,
    y_train: np.ndarray,
    layer_sizes: List[tuple[int, int, str]],
    learning_rate: float,
    epochs: int,
    batch_size: int
) -> Tuple[NeuralNetworkOOP, List[Dict[str, float]]]:
    """Train OOP implementation and measure memory usage"""
    nn_oop = NeuralNetworkOOP(layers=layer_sizes, learning_rate=learning_rate)
    history = nn_oop.train(X_train, y_train, epochs=epochs, batch_size=batch_size)
    return nn_oop, history


@measure_memory_usage
def train_fp(
    X_train: np.ndarray,
    y_train: np.ndarray,
    layer_sizes: List[tuple[int, int, str]],
    learning_rate: float,
    epochs: int,
    batch_size: int
) -> Tuple[NetworkParams, List[TrainingMetrics]]:
    """Train FP implementation and measure memory usage"""
    network_params = init_network_params(layer_sizes, learning_rate)
    history, final_params = train_network_fp(
        X_train, y_train,
        params=network_params,
        epochs=epochs,
        batch_size=batch_size
    )
    return final_params, history


def time_training(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    layer_sizes: List[tuple[int, int, str]],
    learning_rate: float,
    epochs: int,
    batch_size: int
) -> Dict[str, Dict[str, float]]:
    """Time and measure memory usage for both implementations"""
    results = {}
    
    # Time and measure OOP implementation
    print("\nTraining OOP implementation...")
    start_time = time.time()
    (nn_oop, history_oop), mem_used_oop = train_oop(
        X_train, y_train,
        layer_sizes=layer_sizes,
        learning_rate=learning_rate,
        epochs=epochs,
        batch_size=batch_size
    )
    train_time_oop = time.time() - start_time
    
    metrics_oop = nn_oop.evaluate(X_test, y_test)
    results['oop'] = {
        'train_time': train_time_oop,
        'memory_used': mem_used_oop,
        'final_loss': history_oop[-1]['loss'],
        'final_accuracy': history_oop[-1]['accuracy'],
        'test_loss': metrics_oop['loss'],
        'test_accuracy': metrics_oop['accuracy']
    }
    
    # Time and measure FP implementation
    print("\nTraining FP implementation...")
    start_time = time.time()
    (final_params, history_fp), mem_used_fp = train_fp(
        X_train, y_train,
        layer_sizes=layer_sizes,
        learning_rate=learning_rate,
        epochs=epochs,
        batch_size=batch_size
    )
    train_time_fp = time.time() - start_time
    
    metrics_fp = evaluate_fp(X_test, y_test, final_params)
    results['fp'] = {
        'train_time': train_time_fp,
        'memory_used': mem_used_fp,
        'final_loss': history_fp[-1].loss,
        'final_accuracy': history_fp[-1].accuracy,
        'test_loss': metrics_fp.loss,
        'test_accuracy': metrics_fp.accuracy
    }
    
    return results


def plot_comparison(results: Dict[str, Dict[str, float]]):
    """Plot comparison of results"""
    metrics = ['train_time', 'memory_used', 'final_loss', 'final_accuracy', 'test_loss', 'test_accuracy']
    x = np.arange(len(metrics))
    width = 0.35
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    
    # Plot time and memory
    performance_metrics = ['train_time', 'memory_used']
    ax1.bar(x[:2] - width/2, [results['oop'][m] for m in performance_metrics], width, label='OOP')
    ax1.bar(x[:2] + width/2, [results['fp'][m] for m in performance_metrics], width, label='FP')
    ax1.set_ylabel('Value')
    ax1.set_title('Performance Comparison')
    ax1.set_xticks(x[:2])
    ax1.set_xticklabels(performance_metrics)
    ax1.legend()
    
    # Plot accuracy and loss
    learning_metrics = ['final_loss', 'final_accuracy', 'test_loss', 'test_accuracy']
    ax2.bar(x[2:] - width/2, [results['oop'][m] for m in learning_metrics], width, label='OOP')
    ax2.bar(x[2:] + width/2, [results['fp'][m] for m in learning_metrics], width, label='FP')
    ax2.set_ylabel('Value')
    ax2.set_title('Learning Metrics Comparison')
    ax2.set_xticks(x[2:])
    ax2.set_xticklabels(learning_metrics)
    ax2.legend()
    
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()


def print_detailed_comparison(results: Dict[str, Dict[str, float]]):
    """Print detailed comparison of results"""
    print("\nDetailed Comparison:")
    print("-" * 60)
    
    # Performance metrics
    print("\nPerformance Metrics:")
    print(f"{'Metric':<15} {'OOP':<15} {'FP':<15} {'Difference':<15}")
    print("-" * 60)
    
    for metric in ['train_time', 'memory_used']:
        oop_val = results['oop'][metric]
        fp_val = results['fp'][metric]
        diff = oop_val - fp_val
        diff_percent = (diff / oop_val) * 100 if oop_val != 0 else 0
        
        print(f"{metric:<15} {oop_val:>15.4f} {fp_val:>15.4f} {diff:>+15.4f} ({diff_percent:>+.1f}%)")
    
    # Learning metrics
    print("\nLearning Metrics:")
    print(f"{'Metric':<15} {'OOP':<15} {'FP':<15} {'Difference':<15}")
    print("-" * 60)
    
    for metric in ['final_loss', 'final_accuracy', 'test_loss', 'test_accuracy']:
        oop_val = results['oop'][metric]
        fp_val = results['fp'][metric]
        diff = oop_val - fp_val
        diff_percent = (diff / oop_val) * 100 if oop_val != 0 else 0
        
        print(f"{metric:<15} {oop_val:>15.4f} {fp_val:>15.4f} {diff:>+15.4f} ({diff_percent:>+.1f}%)")


def main():
    # Load MNIST data
    from mnist_test import load_mnist
    from sklearn.model_selection import train_test_split
    
    # Load a subset of MNIST for faster comparison
    print("Loading MNIST dataset...")
    X, y = load_mnist(n_samples=5000)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Network configuration
    layer_sizes = [
        (784, 128, "relu"),    # Input layer -> Hidden layer 1
        (128, 64, "relu"),     # Hidden layer 1 -> Hidden layer 2
        (64, 10, "sigmoid"),   # Hidden layer 2 -> Output layer
    ]
    
    # Training parameters
    params = {
        'learning_rate': 0.1,
        'epochs': 10,
        'batch_size': 32
    }
    
    # Run comparison
    print("\nRunning performance comparison...")
    results = time_training(
        X_train, y_train,
        X_test, y_test,
        layer_sizes=layer_sizes,
        **params
    )
    
    # Print detailed comparison
    print_detailed_comparison(results)
    
    # Plot comparison
    plot_comparison(results)


if __name__ == "__main__":
    main() 