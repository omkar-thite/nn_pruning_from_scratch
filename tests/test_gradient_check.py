import numpy as np
from engine import Tensor

np.random.seed(42)  # For reproducibility

def gradient_check(tensor, compute_loss, eps=1e-5, rtol=1e-5, atol=1e-8):
    """
    Checks the analytical gradient of a tensor against its numerical gradient.
    
    Args:
        tensor: The Tensor object to check.
        compute_loss: A function with no arguments that returns a scalar Tensor (the loss).
        eps: The epsilon value for finite differences.
        rtol: Relative tolerance for np.allclose.
        atol: Absolute tolerance for np.allclose.
        
    Returns:
        is_close (bool): True if gradients match within tolerance.
        max_diff (float): The maximum absolute difference between gradients.
        analytical_grad (np.ndarray): The gradients from backpropagation.
        numerical_grad (np.ndarray): The computed numerical gradients.
    """
    # 1. Compute analytical gradients via standard backpropagation
    loss = compute_loss()
    
    # Reset gradients to avoid accumulation from previous runs
    tensor.grad = np.zeros_like(tensor.grad)
    loss.backward()
    analytical_grad = tensor.grad.copy()

    # 2. Compute numerical gradients via central difference
    numerical_grad = np.zeros_like(tensor.arr, dtype=float)
    
    # Use nditer to iterate through the multi-dimensional array element by element
    it = np.nditer(tensor.arr, flags=['multi_index'], op_flags=['readwrite'])
    
    while not it.finished:
        idx = it.multi_index
        original_val = tensor.arr[idx]

        # f(x + eps)
        tensor.arr[idx] = original_val + eps
        loss_plus = compute_loss().arr
        
        # f(x - eps)
        tensor.arr[idx] = original_val - eps
        loss_minus = compute_loss().arr
        
        # Restore original value
        tensor.arr[idx] = original_val
        
        # Calculate central difference and force it to a scalar float
        numerical_grad[idx] = ((loss_plus - loss_minus) / (2 * eps)).item()  # safe for any shape
        it.iternext()

    # 3. Compare the two gradients
    is_close = np.allclose(analytical_grad, numerical_grad, rtol=rtol, atol=atol)
    max_diff = np.max(np.abs(analytical_grad - numerical_grad))
    
    return is_close, max_diff, analytical_grad, numerical_grad


def test_gradient_check():
    """
    Test the gradient_check function with a simple linear operation.
    """
    threshold = 1e-5

    # Initialize random tensors
    np.random.seed(42)
    X = Tensor(np.random.randn(3, 4), label='X')
    W = Tensor(np.random.randn(4, 2), label='W')

    # Define the computation graph as a closure
    def forward_pass():
        # Example: y = sum( (X @ W) * (X @ W) )
        out = X @ W
        squared = out * out
        loss = squared.sum(axis=1).sum(axis=0) # Reduce to scalar
        return loss

    # Run the gradient check on the weight matrix W
    is_correct, max_difference, analytic_grad, num_grad = gradient_check(W, forward_pass)

    assert is_correct, f"Gradients do not match! Max difference: {max_difference:.2e}"
    assert max_difference < threshold, f"Max difference {max_difference:.2e} exceeds threshold {threshold:.2e}"
