import os
import time
import sys
import csv
import json

import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from scipy import stats 

from engine import Tensor
from nn import MLP
from optimizer import Adam
from loss import cross_entropy

from prune import DynamicPruner


RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

def count_flops(model):
    """2*n_in*n_out per matmul (mult+add) + n_out for the bias add, per layer."""
    total = 0
    for layer in model.layers:
        n_in, n_out = layer.W.arr.shape
        total += 2 * n_in * n_out + n_out
    return total

def measure_inference_time(model, X, n_repeats=100, warmup=10):
    x = Tensor(X)
    for _ in range(warmup):
        model(x)
    times = []
    for _ in range(n_repeats):
        t0 = time.perf_counter()
        model(x)
        times.append(time.perf_counter() - t0)
    return float(np.mean(times)), float(np.std(times)) 


def run_seed_sweep(criterion, n_seeds=5, target_sparsity=0.9, epochs=500, batch_size=50):
    accs = []
    for seed in range(n_seeds):
        np.random.seed(seed)
        X_train, Y_train, X_test, Y_test = get_and_process_digits(num_train=1000, num_test=200, seed=seed)
        model = MLP(64, [100, 100, 10])
        optimizer = Adam(model.parameters(), lr=0.001)
        n_batches = len(X_train) // batch_size
        pruning_epochs = int(epochs * 0.7)
        prune_steps = n_batches * pruning_epochs
        prune_interval = prune_steps // 8
        pruner = DynamicPruner(model, optimizer, final_sparsity=target_sparsity,
                                prune_steps=prune_steps, prune_interval=prune_interval,
                                criterion=criterion)
        for ep in range(epochs):
            for xb, yb in get_batches(X_train, Y_train, batch_size):
                pruner.step_prune(xb, yb)
                optimizer.zero_grad()
                out = model(Tensor(xb))
                loss = cross_entropy(out, yb)
                loss.backward()
                optimizer.step()
        out = model(Tensor(X_test))
        acc = np.mean(np.argmax(out.arr, axis=1) == Y_test)
        accs.append(acc)
    return np.array(accs)


def compare_criteria(n_seeds=5):
    saliency_accs = run_seed_sweep('saliency', n_seeds=n_seeds)
    magnitude_accs = run_seed_sweep('magnitude', n_seeds=n_seeds)

    t_stat, p_value = stats.ttest_ind(saliency_accs, magnitude_accs, equal_var=False)  # Welch's

    print(f"Saliency:  {saliency_accs.mean():.4f} ± {saliency_accs.std():.4f}")
    print(f"Magnitude: {magnitude_accs.mean():.4f} ± {magnitude_accs.std():.4f}")
    print(f"Δ = {(saliency_accs.mean() - magnitude_accs.mean())*100:.2f} pts, p = {p_value:.4f}")
    return saliency_accs, magnitude_accs, p_value

def plot_pareto(results, path='pareto_curve.png'):
    sparsities = sorted(set(r['target_sparsity'] for r in results))
    means = [np.mean([r['accuracy'] for r in results if r['target_sparsity'] == s]) for s in sparsities]
    stds = [np.std([r['accuracy'] for r in results if r['target_sparsity'] == s]) for s in sparsities]

    plt.figure(figsize=(7, 5))
    plt.errorbar(sparsities, means, yerr=stds, marker='o', capsize=4)
    plt.xlabel('Target Sparsity')
    plt.ylabel('Test Accuracy')
    plt.title('Sparsity vs Accuracy (Pareto Curve)')
    plt.grid(True, alpha=0.3)
    plt.savefig(path, dpi=150)
    plt.close()


def run_sparsity_sweep(sparsities=(0.0, 0.5, 0.75, 0.9, 0.95), n_seeds=5,
                        epochs=500, batch_size=50, criterion='saliency'):
    results = []
    for target in sparsities:
        for seed in range(n_seeds):
            np.random.seed(seed)
            X_train, Y_train, X_test, Y_test = get_and_process_digits(num_train=1000, num_test=200)
            model = MLP(64, [100, 100, 10])
            optimizer = Adam(model.parameters(), lr=0.001)
            n_batches = len(X_train) // batch_size
            pruning_epochs = int(epochs * 0.7)
            prune_steps = n_batches * pruning_epochs
            prune_interval = max(1, prune_steps // 8)

            if target > 0:
                pruner = DynamicPruner(model, optimizer, final_sparsity=target,
                                        prune_steps=prune_steps, prune_interval=prune_interval,
                                        criterion=criterion)
            for ep in range(epochs):
                for xb, yb in get_batches(X_train, Y_train, batch_size):
                    if target > 0:
                        pruner.step_prune(xb, yb)
                    optimizer.zero_grad()
                    out = model(Tensor(xb))
                    loss = cross_entropy(out, yb)
                    loss.backward()
                    optimizer.step()

            out = model(Tensor(X_test))
            acc = float(np.mean(np.argmax(out.arr, axis=1) == Y_test))

            achieved = (1 - sum(m.sum() for m in pruner.masks) / sum(len(m) for m in pruner.masks)) if target > 0 else 0.0
            results.append({'target_sparsity': target, 'achieved_sparsity': achieved,
                             'seed': seed, 'accuracy': acc, 'criterion': criterion})
            print(f"  sparsity={target} seed={seed} acc={acc:.4f}")
    return results


def save_results_csv(results, path):
    with open(path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)
# ─────────────────────────────────────────────────────────────────────────────
# Dataset
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


def get_and_process_digits(num_train=1000, num_test=200, seed=RANDOM_STATE):
    """
    Processes the Digits dataset for training and testing.
    total samples: 1797, 8x8 images of digits (0-9)

    Args:
        num_train (int): Number of training samples to use.
        num_test (int): Number of testing samples to use.
    """
    X, y = load_digits(return_X_y=True)

    X = X / 16.0  # normalize
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=seed, stratify=y
    )

    return X_train[:num_train], y_train[:num_train], X_test[:num_test], y_test[:num_test]


def train_nn(prune=False, batch_size=50, epochs=500):
    
    # 10-class digit images (8×8 flattened → 64 features), 1797 samples
    # Tests your net on something image-like without images
    X_train, Y_train, X_test, Y_test = get_and_process_digits(num_train=1000, num_test=200)

    INPUT_SIZE = 64
    OUTPUT_SIZE = 10
    
    # Calculate total batches per epoch
    n_batches = len(X_train) // batch_size
    
    # Restrict pruning to the first 70% of epochs to guarantee a dense fine-tuning phase
    pruning_epochs = int(epochs * 0.7)
    prune_steps = n_batches * pruning_epochs
    
    # Ensure interval divides perfectly into prune_steps so the final pruning event hits exactly 1.0 sparsity
    prune_interval = prune_steps // 8
    
    model = MLP(INPUT_SIZE, [100, 100, OUTPUT_SIZE])
    optimizer = Adam(model.parameters(), lr=0.001)
    
    if prune:
        pruner = DynamicPruner(model, optimizer, final_sparsity=0.9, prune_steps=prune_steps, prune_interval=prune_interval)
            
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
            
    return acc, model


def evaluate_pruning():
    os.makedirs('results', exist_ok=True)

    SWEEP_EPOCHS = 150   # enough to show the shape of the curve, not full convergence
    FULL_EPOCHS = 500    

    # Part 1: Pareto curve
    print("Running sparsity sweep (saliency criterion)...")
    sweep_results = run_sparsity_sweep(criterion='saliency', epochs=SWEEP_EPOCHS)
    save_results_csv(sweep_results, 'results/pareto_raw.csv')
    plot_pareto(sweep_results, 'results/pareto_curve.png')

    # Part 2: real cost measurement at 90% sparsity
    print("\nMeasuring real cost (FLOPs, wall-clock) at 90% sparsity...")
    X_train, Y_train, X_test, Y_test = get_and_process_digits(num_train=1000, num_test=200)
    
    dense_acc, dense_model = train_nn(prune=False)
    pruned_acc, pruned_model = train_nn(prune=True)

    dense_flops = count_flops(dense_model)
    pruned_flops = count_flops(pruned_model)
    dense_t_mean, dense_t_std = measure_inference_time(dense_model, X_test)
    pruned_t_mean, pruned_t_std = measure_inference_time(pruned_model, X_test)

    cost_report = {
        'dense_flops': dense_flops, 'pruned_flops': pruned_flops,
        'flop_reduction_pct': 100 * (1 - pruned_flops / dense_flops),
        'dense_time_s': [dense_t_mean, dense_t_std],
        'pruned_time_s': [pruned_t_mean, pruned_t_std],
        'measurement_method': 'sparse-aware forward pass on structurally resized weight matrices, NOT dense-times-zero'
    }
    with open('results/cost_report.json', 'w') as f:
        json.dump(cost_report, f, indent=2)

    # Part 3: baseline comparison, saliency vs magnitude, at 90% sparsity
    print("\nComparing saliency vs magnitude pruning across seeds...")
    saliency_accs, magnitude_accs, p_value = compare_criteria(n_seeds=5)
    comparison = {
        'saliency_accs': saliency_accs.tolist(), 'magnitude_accs': magnitude_accs.tolist(),
        'saliency_mean': float(saliency_accs.mean()), 'saliency_std': float(saliency_accs.std()),
        'magnitude_mean': float(magnitude_accs.mean()), 'magnitude_std': float(magnitude_accs.std()),
        'p_value': float(p_value)
    }
    with open('results/baseline_comparison.json', 'w') as f:
        json.dump(comparison, f, indent=2)

    # Part 4: falsifiable claim, generated from the actual numbers above
    claim = (f"At 90% sparsity, saliency pruning retains {comparison['saliency_mean']*100:.2f}% "
             f"accuracy vs {comparison['magnitude_mean']*100:.2f}% for magnitude pruning "
             f"(mean over {len(saliency_accs)} seeds, p={p_value:.4f}).")
    print("\n" + claim)
    with open('results/claim.txt', 'w') as f:
        f.write(claim)



# ─────────────────────────────────────────────────────────────────────────────
# Mini-Batched Training Loop
# ─────────────────────────────────────────────────────────────────────────────

def train_and_evaluate_on_spiral_dataset():

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
        train_and_evaluate_on_spiral_dataset()
    