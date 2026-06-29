import sys
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml

from engine import Tensor
from nn import MLP
from optimizer import Adam
from loss import cross_entropy

from prune import DynamicPruner


RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)


# ─────────────────────────────────────────────────────────────────────────────
# Synthetic Dataset: Spirals
# ─────────────────────────────────────────────────────────────────────────────

def generate_spirals(samples_per_class: int, classes: int, noise: float = 0.2):
    """Generates a 2D spiral dataset for non-linear classification."""
    X = np.zeros((samples_per_class * classes, 2))
    y = np.zeros(samples_per_class * classes, dtype=int)
    
    for j in range(classes):
        ix = range(samples_per_class * j, samples_per_class * (j + 1))
        r = np.linspace(0.0, 1, samples_per_class)  # radius
        t = np.linspace(j * 4, (j + 1) * 4, samples_per_class) + np.random.randn(samples_per_class) * noise # theta
        X[ix] = np.c_[r * np.sin(t), r * np.cos(t)]
        y[ix] = j
        
    return X, y

def get_batches(X: np.ndarray, y: np.ndarray, batch_size: int):
    """Yields randomly shuffled mini-batches from the dataset."""
    
    indices = np.random.permutation(len(X))
    for i in range(0, len(X), batch_size):
        idx = indices[i : i + batch_size]
        yield X[idx], y[idx]


def generate_data_multiclass_clf(n_samples=200):
    """Generates a synthetic dataset for multi-class classification."""
    X = np.random.randn(n_samples, 20)
    Y = (np.sum(X**2, axis=1) > 20).astype(int) + (X[:,0] > 0).astype(int)
    return X, np.clip(Y, 0, 2)


def get_mnist_data(num_train=1000, num_test=200):
    """
    Downloads, normalizes, and splits the MNIST dataset.
    Returns subsets by default because scalar-based custom engines (like micrograd) 
    are computationally heavy and cannot efficiently process the full dataset.
    """
    print("Fetching MNIST dataset via OpenML...")
    mnist = fetch_openml('mnist_784', version=1, as_frame=False, parser='auto')
    
    # Extract data and labels
    X = mnist.data.astype(np.float32)
    y = mnist.target.astype(np.int64)

    # Normalize pixel values from [0, 255] to [0.0, 1.0]
    X /= 255.0

    # Split into standard train (60k) and test (10k) allocations
    X_train_full, X_test_full = X[:60000], X[60000:]
    y_train_full, y_test_full = y[:60000], y[60000:]

    # Slice smaller subsets for micrograd compatibility
    X_train = X_train_full[:num_train]
    y_train = y_train_full[:num_train]
    X_test = X_test_full[:num_test]
    y_test = y_test_full[:num_test]

    return X_train, y_train, X_test, y_test


def train_prune(prune=False, batch_size=50, epochs=500):
    
    # Separate data generation to allow fresh batching per epoch
    X_train, Y_train = generate_spirals(samples_per_class=200, classes=4) #generate_data_multiclass_clf()
    X_test, Y_test = generate_spirals(samples_per_class=200, classes=4)

    # Calculate total batches per epoch
    n_batches = len(X_train) // batch_size
    total_train_steps = n_batches * epochs
    prune_interval = total_train_steps // 4  
    
    model = MLP(2, [100, 100, 4])
    optimizer = Adam(model.parameters(), lr=0.001)
    
    if prune:
        pruner = DynamicPruner(model, optimizer, final_sparsity=0.9, prune_steps=total_train_steps, prune_interval=prune_interval)
            
    for ep in range(epochs):
        # Re-instantiate the generator for the new epoch
        train_loader = get_batches(X_train, Y_train, batch_size=batch_size)

        for xb, yb in train_loader:
            if prune:
                pruner.step_prune(xb, yb)
                
            optimizer.zero_grad()
            out = model(Tensor(xb))
            loss = cross_entropy(out, yb)
            loss.backward()
            optimizer.step()

        out = model(Tensor(X_test))
        preds = np.argmax(out.arr, axis=1)
        acc = np.mean(preds == Y_test)
    
        if prune:
            remaining_neurons = [model.layers[i].W.arr.shape[1] for i in range(pruner.n_hidden)]
            print(f"Final Sparse Configuration: Layer 1: {remaining_neurons[0]}/100, Layer 2: {remaining_neurons[1]}/100")
            
    return acc


def evaluate_pruning():
    print("Evaluating Dense Baseline Network...")
    dense_acc = train_prune(prune=False)
    
    print("\nEvaluating Gradient-Pruned Dynamic Network...")
    sparse_acc = train_prune(prune=True)

    print(f"\n--- Accuracy Report ---")
    print(f"Dense Network Target Accuracy: {dense_acc*100:.2f}%")
    print(f"90% Sparse Network Accuracy:   {sparse_acc*100:.2f}%")
    print(f"Accuracy Cost of Pruning:      {(dense_acc - sparse_acc)*100:.2f}%")



# ─────────────────────────────────────────────────────────────────────────────
# Mini-Batched Training Loop
# ─────────────────────────────────────────────────────────────────────────────

def train_and_evaluate_spiral_clf():

    # 1. Prepare Dataset
    N_SAMPLES = 100 
    CLASSES = 3
    BATCH_SIZE = 32
    EPOCHS = 150
    
    X_train, y_train = generate_spirals(N_SAMPLES, CLASSES)

    # Initialize Model and Optimizer
    # 2D input (x, y) -> Hidden 16 -> Hidden 16 -> 3 Classes
    model = MLP(nin=2, nouts=[16, 16, CLASSES])
    opt = Adam(model.parameters(), lr=1e-2, weight_decay=1e-4)

    print(model)
    print(opt, "\n")

    history_loss = []
    history_acc = []

    # Training Loop
    for epoch in range(1, EPOCHS + 1):
        epoch_loss = 0.0
        epoch_correct = 0
        batches = 0

        for X_batch, y_batch in get_batches(X_train, y_train, BATCH_SIZE):
            X_tensor = Tensor(X_batch)
            
            # Forward pass
            probs = model(X_tensor)
            loss = cross_entropy(probs, y_batch)
            
            # Backward pass
            opt.zero_grad()
            loss.backward()
            
            # Optimization step
            opt.step()

            # Record batch metrics
            epoch_loss += loss.arr.item()
            preds = np.argmax(probs.arr, axis=1)
            epoch_correct += np.sum(preds == y_batch)
            batches += 1

        # Calculate epoch metrics
        avg_loss = epoch_loss / batches
        avg_acc = epoch_correct / len(X_train)
        
        history_loss.append(avg_loss)
        history_acc.append(avg_acc)

        if epoch % 25 == 0 or epoch == 1:
            print(f"Epoch {epoch:3d}/{EPOCHS} | Loss: {avg_loss:.4f} | Acc: {avg_acc:.2%}")

    # ─────────────────────────────────────────────────────────────────────────────
    # Report Learning Curve
    # ─────────────────────────────────────────────────────────────────────────────
    
    fig, ax1 = plt.subplots(figsize=(8, 5))

    color = 'tab:red'
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Cross-Entropy Loss', color=color)
    ax1.plot(range(1, EPOCHS + 1), history_loss, color=color, label='Loss')
    ax1.tick_params(axis='y', labelcolor=color)

    ax2 = ax1.twinx()
    color = 'tab:blue'
    ax2.set_ylabel('Accuracy', color=color)
    ax2.plot(range(1, EPOCHS + 1), history_acc, color=color, label='Accuracy')
    ax2.tick_params(axis='y', labelcolor=color)
    
    # Restrict accuracy y-axis to valid percentage domain
    ax2.set_ylim([0.0, 1.05])

    fig.tight_layout()
    plt.title('Mini-Batch Training: Spirals Dataset')
    plt.grid(True, alpha=0.3)
    plt.show()

    return history_loss, history_acc


if __name__ == '__main__':
    if sys.argv[-1] == 'prune':
        evaluate_pruning()
    else:
        train_and_evaluate_spiral_clf()
    