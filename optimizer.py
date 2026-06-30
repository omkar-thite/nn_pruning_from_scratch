import numpy as np 
from typing import List

class Adam:
    r"""
    Adam optimiser (Kingma & Ba, 2014).
 
    Maintains per-parameter running estimates of the first moment (mean of
    gradients, m) and the second moment (uncentred variance, v), then scales
    each update by the bias-corrected ratio m̂ / (√v̂ + ε).
 
    Update rule at step t:
        g_t  = ∂L/∂θ                               gradient
        m_t  = β₁ · m_{t-1} + (1 - β₁) · g_t     biased 1st moment
        v_t  = β₂ · v_{t-1} + (1 - β₂) · g_t²    biased 2nd moment
        m̂_t  = m_t / (1 - β₁ᵗ)                   bias-corrected 1st
        v̂_t  = v_t / (1 - β₂ᵗ)                   bias-corrected 2nd
        θ_t  = θ_{t-1} - alpha · m̂_t / (√v̂_t + ε)
 
    Args:
        parameters   : Tensor objects to optimise
        lr           : learning rate alpha            (default 1e-3)
        beta1        : 1st-moment EMA decay β₁   (default 0.9)
        beta2        : 2nd-moment EMA decay β₂   (default 0.999)
        eps          : numerical stability ε      (default 1e-8)
        weight_decay : L2 regularisation λ        (default 0.0)
    """
 
    def __init__(
        self,
        parameters,
        lr: float           = 1e-3,
        beta1: float        = 0.9,
        beta2: float        = 0.999,
        eps: float          = 1e-8,
        weight_decay: float = 0.0,
    ):
        self.parameters   = list(parameters)
        self.lr           = lr
        self.beta1        = beta1
        self.beta2        = beta2
        self.eps          = eps
        self.weight_decay = weight_decay
 
        # Allocate moment buffers once — same shape as each parameter
        self.m = [np.zeros_like(p.arr, dtype=float) for p in self.parameters]
        self.v = [np.zeros_like(p.arr, dtype=float) for p in self.parameters]
        self.t = [np.zeros_like(p.arr, dtype=int) for p in self.parameters]

    def step(self):
        """Apply one Adam update to every tracked parameter."""
        for idx, p in enumerate(self.parameters):
            g = p.grad.copy()

            if self.weight_decay:
                g += self.weight_decay * p.arr

            # Every active element advances its own counter by 1 each call.
            self.t[idx] += 1

            self.m[idx] = self.beta1 * self.m[idx] + (1.0 - self.beta1) * g
            self.v[idx] = self.beta2 * self.v[idx] + (1.0 - self.beta2) * g ** 2

            # Elementwise bias correction using each element's own t.
            bc1 = 1.0 - self.beta1 ** self.t[idx]
            bc2 = 1.0 - self.beta2 ** self.t[idx]

            m_hat = self.m[idx] / bc1
            v_hat = self.v[idx] / bc2

            p.arr -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)
 
    def zero_grad(self):
        """Set every tracked parameter's gradient to zero."""
        for p in self.parameters:
            p.grad = np.zeros_like(p.arr, dtype=float)
 
    def __repr__(self) -> str:
        t_max = max((t.max() for t in self.t), default=0)
        return (
            f"Adam(lr={self.lr}, β₁={self.beta1}, β₂={self.beta2}, "
            f"ε={self.eps}, weight_decay={self.weight_decay}, t_max={t_max})"
        )
 