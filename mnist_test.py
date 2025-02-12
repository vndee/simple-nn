"""
Test Neural Network implementation with MNIST dataset
"""

import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from tqdm import tqdm

from nn_oop import NeuralNetwork

def load_mnist(n_samples=None):
    """Load MNIST dataset"""
    print("Loading MNIST dataset...")
    X, y = fetch_openml('mnist_784', version=1, return_X_y=True, as_frame=False)
    
    if n_samples is not None:
        X = X[:n_samples]
        y = y[:n_samples]
    
    # Convert string labels to integers
    y = y.astype(np.int32)
    
    # Normalize pixel values
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    
    # Convert to float64
    X = X.astype(np.float64)
    
    # One-hot encode the labels
    y_onehot = np.zeros((y.shape[0], 10))
    y_onehot[np.arange(y.shape[0]), y] = 1
    
    return X, y_onehot

def plot_training_history(history):
    """Plot training history"""
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot([x['loss'] for x in history])
    plt.title('Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    
    plt.subplot(1, 2, 2)
    plt.plot([x['accuracy'] for x in history])
    plt.title('Training Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    
    plt.tight_layout()
    plt.show()

def main():
    # Load a subset of MNIST for faster testing
    X, y = load_mnist(n_samples=10000)
    
    # Split into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Create neural network
    nn = NeuralNetwork(
        layers=[
            (784, 128, "relu"),    # Input layer -> Hidden layer 1
            (128, 64, "relu"),     # Hidden layer 1 -> Hidden layer 2
            (64, 10, "sigmoid"),   # Hidden layer 2 -> Output layer
        ],
        learning_rate=0.1
    )
    
    # Train the network
    print("Training neural network...")
    history = nn.train(
        X_train, y_train,
        epochs=50,
        batch_size=32
    )
    
    # Plot training history
    plot_training_history(history)
    
    # Evaluate on test set
    print("\nEvaluating on test set...")
    metrics = nn.evaluate(X_test, y_test)
    print(f"Test Loss: {metrics['loss']:.4f}")
    print(f"Test Accuracy: {metrics['accuracy']:.4f}")
    
    # Show some example predictions
    print("\nExample predictions:")
    n_examples = 5
    predictions = nn.predict(X_test[:n_examples])
    true_labels = np.argmax(y_test[:n_examples], axis=1)
    pred_labels = np.argmax(predictions, axis=1)
    
    for i in range(n_examples):
        print(f"True: {true_labels[i]}, Predicted: {pred_labels[i]}")

if __name__ == "__main__":
    main() 