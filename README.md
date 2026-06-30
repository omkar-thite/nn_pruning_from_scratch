# Tiny Autograd + Dynamic Pruning

A from-scratch reverse-mode autograd engine (`engine.py`, a `Tensor` class with `.backward()`), a small MLP library (`nn.py`), an Adam optimizer (`optimizer.py`), a cross-entropy loss wired into the autograd graph (`loss.py`), and a Taylor-saliency / magnitude **dynamic structured pruner** (`prune.py`) that prunes hidden neurons during training. `train.py` ties everything together: a spiral-dataset training demo (Part 2) and a digits-dataset pruning evaluation (Parts 3–4) that writes its artifacts to `results/`.

## Project layout

```
engine.py      # Tensor autograd engine
nn.py          # Layer / MLP modules built on Tensor
optimizer.py   # Adam optimizer
loss.py        # cross_entropy loss (custom backward)
prune.py       # DynamicPruner: Taylor-saliency / magnitude structured pruning
train.py       # entry point: spiral demo + digits pruning evaluation
conftest.py    # pytest fixtures shared by the test suite
results/       # output of the Part 3/4 pruning evaluation (csv/json/png)
```

## Installation
cd into project root and run followig command:
```bash
uv sync
```


## Run instructions

### 1. Gradient-check tests

Verifies that every `Tensor` op's analytic backward pass  matches numerical finite-difference gradients.

```bash
pytest tests/test_gradient_check.py
```


### 2. Reproduce spiral training run

Trains a small MLP (`2 → 16 → 16 → 3`) on a synthetic 3-class spiral dataset and plots the loss/accuracy learning curves.

```bash
python train.py
```


### 3. Single self-pruning run

Trains one MLP (`64 → 100 → 100 → 10`) on the digits dataset with `DynamicPruner` enabled, pruning to 90% sparsity over the first 70% of training, then fine-tuning dense-free for the remainder. Prints the final retained-neuron count per layer and the test accuracy.

```bash
uv run python -c "from train import train_nn; train_nn(prune=True)"
```

(or: `python -c "from train import train_nn; train_nn(prune=True)"`)

### 4. Pareto sweep (sparsity vs. accuracy, cost, and criterion comparison)

Runs the full evaluation suite: a sparsity sweep (`0.0 → 0.95`) for the Pareto curve, a FLOPs/wall-clock cost comparison between the dense and 90%-pruned models, and a saliency-vs-magnitude criterion comparison over 5 seeds (Welch's t-test). Writes `results/pareto_raw.csv`, `results/pareto_curve.png`, `results/cost_report.json`, `results/baseline_comparison.json`, and `results/claim.txt`.

```bash
uv run python train.py prune
```

(or: `python train.py prune`)

> Note: this single command's `evaluate_pruning()` call covers the sweep needed for Part 4 (and reuses the same single-run pruning machinery from Part 3 internally for the cost report), so a fresh `results/` folder after running it contains everything Part 4 asks for.