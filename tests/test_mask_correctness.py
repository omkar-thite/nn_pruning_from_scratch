# test_correctness.py
import numpy as np

from engine import Tensor
from nn import MLP
from optimizer import Adam
from prune import DynamicPruner

def test_masked_weight_correctness():
    np.random.seed(42)
    
    # Initialize a 4 -> 4 -> 4 -> 2 network
    model = MLP(nin=4, nouts=[4, 4, 2])
    optimizer = Adam(model.parameters(), lr=0.01)
    
    # Configure pruner to drop to 50% sparsity immediately on step 1
    pruner = DynamicPruner(
        model=model, 
        optimizer=optimizer, 
        final_sparsity=0.5, 
        prune_steps=1, 
        prune_interval=1, 
        criterion='magnitude'
    )
    
    # Verify initial dense shapes
    assert model.layers[0].W.arr.shape == (4, 4)
    assert model.layers[1].W.arr.shape == (4, 4)
    assert model.layers[2].W.arr.shape == (4, 2)
    
    X_dummy = np.random.randn(2, 4)
    Y_dummy = np.array([0, 1])
    
    # Trigger structural pruning step
    pruner.step_prune(X_dummy, Y_dummy)
    
    # 1. Structural Correctness: The active weight matrices must be physically smaller
    # Layer 0 outputs 50% fewer neurons (4 -> 2)
    assert model.layers[0].W.arr.shape == (4, 2), f"Expected (4, 2), got {model.layers[0].W.arr.shape}"
    assert model.layers[0].b.arr.shape == (1, 2)
    
    # Layer 1 receives 2 inputs, outputs 2
    assert model.layers[1].W.arr.shape == (2, 2), f"Expected (2, 2), got {model.layers[1].W.arr.shape}"
    
    # Layer 2 receives 2 inputs, outputs 2 (classes)
    assert model.layers[2].W.arr.shape == (2, 2)
    
    # 2. Mask Verification: Exactly 2 neurons per hidden layer should remain true
    assert np.sum(pruner.masks[0]) == 2
    assert np.sum(pruner.masks[1]) == 2
    
    # 3. Master State Integrity: Pruned columns/rows in the master matrices must be neutralized
    pruned_idx_L0 = np.where(~pruner.masks[0])[0]
    
    # Optimizer moment arrays corresponding to pruned output columns must be zeroed
    assert np.all(pruner.master_t_W[0][:, pruned_idx_L0] == 0), "Pruned momentum states were not zeroed."
    
    # The subsequent layer's input rows corresponding to those same pruned neurons must also be zeroed
    assert np.all(pruner.master_t_W[1][pruned_idx_L0, :] == 0), "Downstream momentum states were not zeroed."
    
    print("Correctness Test Passed: Structural matrix reduction and master state neutralization are mathematically sound.")

if __name__ == '__main__':
    test_masked_weight_correctness()