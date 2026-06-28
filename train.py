import numpy as np
import matplotlib.pyplot as plt

from engine import Tensor
from nn import MLP
from optimizer import Adam
from loss import cross_entropy

RANDOM_STATE = 42

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

# ─────────────────────────────────────────────────────────────────────────────
# Mini-Batched Training Loop
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    np.random.seed(RANDOM_STATE)  # for reproducibility

    # 1. Prepare Dataset
    N_SAMPLES = 100 
    CLASSES = 3
    BATCH_SIZE = 32
    EPOCHS = 150
    
    X_train, y_train = generate_spirals(N_SAMPLES, CLASSES)

    # 2. Initialize Model and Optimizer
    # 2D input (x, y) -> Hidden 16 -> Hidden 16 -> 3 Classes
    model = MLP(nin=2, nouts=[16, 16, CLASSES])
    opt = Adam(model.parameters(), lr=1e-2, weight_decay=1e-4)

    print(model)
    print(opt, "\n")

    history_loss = []
    history_acc = []

    # 3. Training Loop
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