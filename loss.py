import numpy as np  
from engine import Tensor

# ─────────────────────────────────────────────────────────────────────────────
# Cross-entropy loss — wires into the autograd graph via custom _backward
# ─────────────────────────────────────────────────────────────────────────────
 
def cross_entropy(probs: Tensor, targets: np.ndarray) -> Tensor:
    """
    Negative log-likelihood averaged over the batch.
 
    Args:
        probs   : (batch, C) Tensor — output of a softmax layer.
        targets : 1-D int ndarray of class indices, shape (batch,).
 
    Returns:
        Scalar Tensor whose .backward() propagates ∂L/∂probs.
 
    Gradient derivation:
        L = -1/N · Σᵢ log(pᵢ[yᵢ])
        ∂L/∂pᵢ[yᵢ]  = -1 / (N · pᵢ[yᵢ])
        ∂L/∂pᵢ[j≠yᵢ] = 0
    """
    batch  = probs.arr.shape[0]
    p_true = np.clip(probs.arr[np.arange(batch), targets], 1e-12, 1.0)
    loss   = Tensor(np.mean(-np.log(p_true)))   # scalar
 
    def _backward():
        grad = np.zeros_like(probs.arr)
        grad[np.arange(batch), targets] = -1.0 / (p_true * batch)
        probs.grad += grad
 
    loss._backward = _backward
    loss._parents  = (probs,)
    loss._op       = 'cross_entropy'
    return loss