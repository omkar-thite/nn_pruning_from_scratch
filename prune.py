import numpy as np
from engine import Tensor
from loss import cross_entropy

from nn import MLP
from optimizer import Adam

# ── Dynamic Pruning Implementation ───────────────────────────────────────────
class DynamicPruner:
    def __init__(self, model: MLP, optimizer: Adam, final_sparsity: float, prune_steps: int, prune_interval: int, criterion: str = 'saliency'):
        self.model = model
        self.optimizer = optimizer
        self.final_sparsity = final_sparsity
        self.prune_steps = prune_steps
        self.prune_interval = prune_interval
        self.criterion = criterion
        self.step = 0
        
        self.n_hidden = len(model.layers) - 1
        self.masks = []
        self.master_W, self.master_b = [], []
        self.master_m_W, self.master_m_b = [], []
        self.master_v_W, self.master_v_b = [], []
        self.master_t_W, self.master_t_b = [], []   # NEW: per-element step counters

        self.populate_initial_state()

    def populate_initial_state(self):
        """Initializes masks and stores the pristine state of the model and optimizer."""

        for i in range(self.n_hidden):
            n_neurons = self.model.layers[i].W.arr.shape[1]
            self.masks.append(np.ones(n_neurons, dtype=bool))

        # Store pristine initialization state as master framework
        for i, layer in enumerate(self.model.layers):
            self.master_W.append(layer.W.arr.copy())
            self.master_b.append(layer.b.arr.copy())
            self.master_m_W.append(self.optimizer.m[2*i].copy())
            self.master_m_b.append(self.optimizer.m[2*i+1].copy())
            self.master_v_W.append(self.optimizer.v[2*i].copy())
            self.master_v_b.append(self.optimizer.v[2*i+1].copy())
            self.master_t_W.append(self.optimizer.t[2*i].copy())
            self.master_t_b.append(self.optimizer.t[2*i+1].copy())


    def sync_to_master(self):
        """Copies active parameters and adam moments back into the master state."""

        for i, layer in enumerate(self.model.layers):
            row_mask = self.masks[i-1] if i > 0 else slice(None)
            col_mask = self.masks[i] if i < self.n_hidden else slice(None)

            if i == 0:
                self.master_W[i][:, col_mask] = layer.W.arr
                self.master_m_W[i][:, col_mask] = self.optimizer.m[2*i]
                self.master_v_W[i][:, col_mask] = self.optimizer.v[2*i]
                self.master_t_W[i][:, col_mask] = self.optimizer.t[2*i]
            elif i == self.n_hidden:
                self.master_W[i][row_mask, :] = layer.W.arr
                self.master_m_W[i][row_mask, :] = self.optimizer.m[2*i]
                self.master_v_W[i][row_mask, :] = self.optimizer.v[2*i]
                self.master_t_W[i][row_mask, :] = self.optimizer.t[2*i]
            else:
                idx = np.ix_(row_mask, col_mask)
                self.master_W[i][idx] = layer.W.arr
                self.master_m_W[i][idx] = self.optimizer.m[2*i]
                self.master_v_W[i][idx] = self.optimizer.v[2*i]
                self.master_t_W[i][idx] = self.optimizer.t[2*i]

            self.master_b[i][:, col_mask] = layer.b.arr
            self.master_m_b[i][:, col_mask] = self.optimizer.m[2*i+1]
            self.master_v_b[i][:, col_mask] = self.optimizer.v[2*i+1]
            self.master_t_b[i][:, col_mask] = self.optimizer.t[2*i+1]



    def sync_to_network(self):
        """Extracts dynamically sized submatrices corresponding to active neurons."""

        for i, layer in enumerate(self.model.layers):
            row_mask = self.masks[i-1] if i > 0 else slice(None)
            col_mask = self.masks[i] if i < self.n_hidden else slice(None)

            if i == 0:
                layer.W.arr = self.master_W[i][:, col_mask].copy()  # only keep columns corresponding to active neurons in the current layer
                self.optimizer.m[2*i] = self.master_m_W[i][:, col_mask].copy()
                self.optimizer.v[2*i] = self.master_v_W[i][:, col_mask].copy()
                self.optimizer.t[2*i] = self.master_t_W[i][:, col_mask].copy()
            elif i == self.n_hidden:
                layer.W.arr = self.master_W[i][row_mask, :].copy()   # only keep rows corresponding to active neurons in the previous layer
                self.optimizer.m[2*i] = self.master_m_W[i][row_mask, :].copy()
                self.optimizer.v[2*i] = self.master_v_W[i][row_mask, :].copy()
                self.optimizer.t[2*i] = self.master_t_W[i][row_mask, :].copy()
            else:
                idx = np.ix_(row_mask, col_mask)
                layer.W.arr = self.master_W[i][idx].copy()
                self.optimizer.m[2*i] = self.master_m_W[i][idx].copy()
                self.optimizer.v[2*i] = self.master_v_W[i][idx].copy()
                self.optimizer.t[2*i] = self.master_t_W[i][idx].copy()

            layer.b.arr = self.master_b[i][:, col_mask].copy()
            self.optimizer.m[2*i+1] = self.master_m_b[i][:, col_mask].copy()
            self.optimizer.v[2*i+1] = self.master_v_b[i][:, col_mask].copy()
            self.optimizer.t[2*i+1] = self.master_t_b[i][:, col_mask].copy()

        self.model.zero_grad()


    def step_prune(self, X_batch, Y_batch):
        self.step += 1
        
        if self.step >= self.prune_steps or self.step % self.prune_interval != 0:
            return
            
        # Force floating-point arithmetic to prevent integer division zeroing out the multiplier
        current_sparsity = self.final_sparsity * (float(self.step) / self.prune_steps)
        
        # Save structural adjustments back to dense representation
        self.sync_to_master()

        # reinstate the full dense network state for gradient probing 
        for i in range(self.n_hidden):
            self.masks[i] = np.ones_like(self.masks[i])
        self.sync_to_network()

        # Isolate the gradients for the full topology using the current batch
        self.model.zero_grad()
        out = self.model(Tensor(X_batch))
        loss = cross_entropy(out, Y_batch)
        loss.backward()

        # Taylor criteria importance masking and structure reshaping
        for i in range(self.n_hidden):
            n_neurons = len(self.masks[i])
            n_keep = max(1, int(n_neurons * (1.0 - current_sparsity)))

            W_grad = self.model.layers[i].W.grad
            W_arr = self.model.layers[i].W.arr

            if self.criterion == 'saliency':
                # Taylor importance: Σ |W_ij * ∇W_ij|
                importance = np.sum(np.abs(W_arr * W_grad), axis=0)
            elif self.criterion == 'magnitude':
                # Pure magnitude: Σ |W_ij|, ignores the gradient entirely
                importance = np.sum(np.abs(W_arr), axis=0)
            else:
                raise ValueError(f"unknown criterion: {self.criterion}")

            mask = np.zeros(n_neurons, dtype=bool)
            if n_keep < n_neurons:
                # Use argsort to strictly enforce exactly n_keep neurons, avoiding threshold ties
                top_indices = np.argsort(importance)[-n_keep:]
                mask[top_indices] = True
            else:
                mask[:] = True
                
            self.masks[i] = mask

            pruned = ~mask
            self.master_t_b[i][:, pruned] = 0
            self.master_t_W[i][:, pruned] = 0      # output side (columns of layer i)
            self.master_t_W[i+1][pruned, :] = 0    # input side (rows of layer i+1)


        # Slice sub-matrices and inject entirely reconfigured dynamic network
        self.sync_to_network()
