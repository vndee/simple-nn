"""
Run comprehensive experiments comparing OOP and FP Neural Network implementations
"""

import numpy as np
from sklearn.datasets import load_breast_cancer, load_digits
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Callable
import time

from compare_implementations import time_training, print_detailed_comparison


def load_datasets() -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
    """Load real-world datasets"""
    datasets = {}
    
    # Breast Cancer dataset (smallest)
    print("Loading Breast Cancer dataset...")
    cancer = load_breast_cancer()
    X_cancer = cancer.data
    y_cancer = cancer.target
    y_cancer_onehot = np.zeros((y_cancer.size, 2))
    y_cancer_onehot[np.arange(y_cancer.size), y_cancer] = 1
    datasets['breast_cancer'] = (X_cancer, y_cancer_onehot)
    
    # Digits dataset (medium size)
    print("Loading Digits dataset...")
    digits = load_digits()
    # Replicate the digits dataset to make it larger
    X_digits = np.tile(digits.data, (10, 1))  # 10x larger
    y_digits = np.tile(digits.target, 10)
    y_digits_onehot = np.zeros((y_digits.size, 10))
    y_digits_onehot[np.arange(y_digits.size), y_digits] = 1
    datasets['digits'] = (X_digits, y_digits_onehot)
    
    # MNIST subset (largest)
    print("Loading MNIST dataset...")
    from mnist_test import load_mnist
    X_mnist, y_mnist = load_mnist(n_samples=50000)
    datasets['mnist'] = (X_mnist, y_mnist)
    
    return datasets


def get_network_config(input_size: int, output_size: int) -> List[tuple[int, int, str]]:
    """Get network configuration based on input and output size"""
    hidden_size = max(32, min(input_size, 128))
    return [
        (input_size, hidden_size, "relu"),
        (hidden_size, hidden_size // 2, "relu"),
        (hidden_size // 2, output_size, "sigmoid")
    ]


def preprocess_data(X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Preprocess dataset"""
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Split data
    return train_test_split(X_scaled, y, test_size=0.2, random_state=42)


def run_experiment(
    dataset_name: str,
    X: np.ndarray,
    y: np.ndarray,
    epochs: int = 20,
    batch_size: int = 32,
    learning_rate: float = 0.1
) -> Dict[str, Dict[str, float]]:
    """Run experiment on one dataset"""
    print(f"\nRunning experiment on {dataset_name} dataset...")
    print(f"Dataset shape: X={X.shape}, y={y.shape}")
    
    # Preprocess data
    X_train, X_test, y_train, y_test = preprocess_data(X, y)
    
    # Get network configuration
    layer_sizes = get_network_config(X.shape[1], y.shape[1])
    
    # Run comparison
    results = time_training(
        X_train, y_train,
        X_test, y_test,
        layer_sizes=layer_sizes,
        learning_rate=learning_rate,
        epochs=epochs,
        batch_size=batch_size
    )
    
    return results


def format_dataset_label(dataset_name: str, X: np.ndarray) -> str:
    """Format dataset label with size information"""
    return f"{dataset_name}\n(n={X.shape[0]}, d={X.shape[1]})"


def plot_all_results(all_results: Dict[str, Dict[str, Dict[str, float]]], all_datasets: Dict[str, Tuple[np.ndarray, np.ndarray]]):
    """Plot comparison across all datasets and save both individual and combined plots"""
    metrics = ['train_time', 'memory_used', 'final_accuracy', 'test_accuracy']
    datasets = list(all_results.keys())
    
    # Format dataset labels with size information
    dataset_labels = [format_dataset_label(name, all_datasets[name][0]) for name in datasets]
    
    # Create individual plots
    for i, metric in enumerate(metrics):
        plt.figure(figsize=(10, 6))
        x = np.arange(len(datasets))
        width = 0.35
        
        oop_values = [all_results[dataset]['oop'][metric] for dataset in datasets]
        fp_values = [all_results[dataset]['fp'][metric] for dataset in datasets]
        
        # Plot bars
        plt.bar(x - width/2, oop_values, width, label='OOP')
        plt.bar(x + width/2, fp_values, width, label='FP')
        
        # Add value labels on top of bars with appropriate formatting
        for j, v in enumerate(oop_values):
            if metric in ['memory_used', 'train_time']:
                label = f'{v:.1f}'
            else:
                label = f'{v:.3f}'
            plt.text(x[j] - width/2, v, label, ha='center', va='bottom')
            
        for j, v in enumerate(fp_values):
            if metric in ['memory_used', 'train_time']:
                label = f'{v:.1f}'
            else:
                label = f'{v:.3f}'
            plt.text(x[j] + width/2, v, label, ha='center', va='bottom')
        
        # Set labels and title
        if metric == 'memory_used':
            plt.ylabel('Memory (MB)')
        elif metric == 'train_time':
            plt.ylabel('Time (seconds)')
        else:
            plt.ylabel(metric)
        
        plt.title(f'{metric} Comparison')
        plt.xticks(x, dataset_labels, rotation=45, ha='right')
        plt.legend()
        
        # Add grid for better readability
        plt.grid(True, alpha=0.3)
        
        # Adjust y-axis limits to show all bars with some padding
        ymin, ymax = plt.ylim()
        plt.ylim(ymin, ymax * 1.15)  # Add 15% padding at the top
        
        plt.tight_layout()
        # Save individual plot
        plt.savefig(f'plots/{metric}_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    # Create combined figure
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    axes = axes.ravel()
    
    for i, metric in enumerate(metrics):
        ax = axes[i]
        x = np.arange(len(datasets))
        width = 0.35
        
        oop_values = [all_results[dataset]['oop'][metric] for dataset in datasets]
        fp_values = [all_results[dataset]['fp'][metric] for dataset in datasets]
        
        # Plot bars
        ax.bar(x - width/2, oop_values, width, label='OOP')
        ax.bar(x + width/2, fp_values, width, label='FP')
        
        # Add value labels on top of bars with appropriate formatting
        for j, v in enumerate(oop_values):
            if metric in ['memory_used', 'train_time']:
                label = f'{v:.1f}'
            else:
                label = f'{v:.3f}'
            ax.text(x[j] - width/2, v, label, ha='center', va='bottom')
            
        for j, v in enumerate(fp_values):
            if metric in ['memory_used', 'train_time']:
                label = f'{v:.1f}'
            else:
                label = f'{v:.3f}'
            ax.text(x[j] + width/2, v, label, ha='center', va='bottom')
        
        # Set labels and title
        if metric == 'memory_used':
            ax.set_ylabel('Memory (MB)')
        elif metric == 'train_time':
            ax.set_ylabel('Time (seconds)')
        else:
            ax.set_ylabel(metric)
        
        ax.set_title(f'{metric} Comparison')
        ax.set_xticks(x)
        ax.set_xticklabels(dataset_labels, rotation=45, ha='right')
        ax.legend()
        
        # Add grid for better readability
        ax.grid(True, alpha=0.3)
        
        # Adjust y-axis limits to show all bars with some padding
        ymin, ymax = ax.get_ylim()
        ax.set_ylim(ymin, ymax * 1.15)  # Add 15% padding at the top
    
    plt.tight_layout()
    # Save combined figure
    plt.savefig('plots/combined_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()
    plt.close()
    
    # Print relative comparisons
    print("\nRelative Comparisons (FP vs OOP):")
    print("-" * 80)
    print(f"{'Dataset':<15} {'Metric':<15} {'OOP':<12} {'FP':<12} {'Ratio (FP/OOP)':<15}")
    print("-" * 80)
    
    for dataset in datasets:
        for metric in metrics:
            oop_val = all_results[dataset]['oop'][metric]
            fp_val = all_results[dataset]['fp'][metric]
            ratio = fp_val / oop_val if oop_val != 0 else float('inf')
            
            if metric in ['memory_used', 'train_time']:
                print(f"{dataset:<15} {metric:<15} {oop_val:>12.1f} {fp_val:>12.1f} {ratio:>15.2f}")
            else:
                print(f"{dataset:<15} {metric:<15} {oop_val:>12.3f} {fp_val:>12.3f} {ratio:>15.3f}")


def print_dataset_info(all_datasets: Dict[str, Tuple[np.ndarray, np.ndarray]]):
    """Print detailed information about datasets"""
    print("\nDataset Information:")
    print("-" * 80)
    print(f"{'Dataset':<15} {'Samples':<10} {'Features':<10} {'Classes':<10} {'Total Size':<12} {'Memory (MB)':<12}")
    print("-" * 80)
    
    for name, (X, y) in all_datasets.items():
        n_samples = X.shape[0]
        n_features = X.shape[1]
        n_classes = y.shape[1]
        total_size = n_samples * n_features
        memory_mb = (X.nbytes + y.nbytes) / (1024 * 1024)  # Convert to MB
        
        print(f"{name:<15} {n_samples:<10} {n_features:<10} {n_classes:<10} {total_size:<12} {memory_mb:<12.2f}")


def main():
    # Load datasets
    print("Loading datasets...")
    all_datasets = load_datasets()
    
    # Print dataset information
    print_dataset_info(all_datasets)
    
    # Store all results
    all_results = {}
    
    # Run experiments on each dataset
    for dataset_name, (X, y) in all_datasets.items():
        results = run_experiment(
            dataset_name=dataset_name,
            X=X,
            y=y,
            epochs=20,
            batch_size=32,
            learning_rate=0.1
        )
        all_results[dataset_name] = results
        
        # Print results for this dataset
        print(f"\nResults for {dataset_name} dataset:")
        print("-" * 60)
        print_detailed_comparison(results)
    
    # Plot overall comparison
    plot_all_results(all_results, all_datasets)
    
    # Print summary statistics
    print("\nSummary Statistics:")
    print("-" * 60)
    
    metrics = ['train_time', 'memory_used', 'final_accuracy', 'test_accuracy']
    for metric in metrics:
        oop_values = [results['oop'][metric] for results in all_results.values()]
        fp_values = [results['fp'][metric] for results in all_results.values()]
        
        print(f"\n{metric}:")
        print(f"{'Statistic':<15} {'OOP':<15} {'FP':<15}")
        print("-" * 45)
        print(f"{'Mean':<15} {np.mean(oop_values):>15.4f} {np.mean(fp_values):>15.4f}")
        print(f"{'Std':<15} {np.std(oop_values):>15.4f} {np.std(fp_values):>15.4f}")
        print(f"{'Min':<15} {np.min(oop_values):>15.4f} {np.min(fp_values):>15.4f}")
        print(f"{'Max':<15} {np.max(oop_values):>15.4f} {np.max(fp_values):>15.4f}")


if __name__ == "__main__":
    main() 