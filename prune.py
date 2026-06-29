# ── Dynamic Pruning Implementation ───────────────────────────────────────────
class DynamicPruner:
    def __init__(self, model: MLP, optimizer: Adam, final_sparsity: float, prune_steps: int):
        self.model = model
        self.optimizer = optimizer
        self.final_sparsity = final_sparsity
        self.prune_steps = prune_steps
        self.step = 0
        
        self.n_hidden = len(model.layers) - 1
        self.masks = []
        self.master_W, self.master_b = [], []
        self.master_m_W, self.master_m_b = [], []
        self.master_v_W, self.master_v_b = [], []

        self.master = {}

    def populate_initial_state(self):
        """Initializes masks and stores the pristine state of the model and optimizer."""

        for i in range(self.n_hidden):
            n_neurons = self.model.layers[i].W.arr.shape[1]
            self.masks.append(np.ones(n_neurons, dtype=bool))

        # Store pristine initialization state as master framework
        for i, layer in enumerate(model.layers):
            self.master_W.append(layer.W.arr.copy())
            self.master_b.append(layer.b.arr.copy())
            self.master_m_W.append(optimizer.m[2*i].copy())
            self.master_m_b.append(optimizer.m[2*i+1].copy())
            self.master_v_W.append(optimizer.v[2*i].copy())
            self.master_v_b.append(optimizer.v[2*i+1].copy())


    def sync_to_master(self):
        """Copies active parameters and adam moments back into the master state."""
        for i, layer in enumerate(self.model.layers):
            row_mask = self.masks[i-1] if i > 0 else slice(None)
            col_mask = self.masks[i] if i < self.n_hidden else slice(None)

            if i == 0:
                self.master_W[i][:, col_mask] = layer.W.arr
                self.master_m_W[i][:, col_mask] = self.optimizer.m[2*i]
                self.master_v_W[i][:, col_mask] = self.optimizer.v[2*i]
            elif i == self.n_hidden:
                self.master_W[i][row_mask, :] = layer.W.arr
                self.master_m_W[i][row_mask, :] = self.optimizer.m[2*i]
                self.master_v_W[i][row_mask, :] = self.optimizer.v[2*i]
            else:
                idx = np.ix_(row_mask, col_mask)
                self.master_W[i][idx] = layer.W.arr
                self.master_m_W[i][idx] = self.optimizer.m[2*i]
                self.master_v_W[i][idx] = self.optimizer.v[2*i]

            self.master_b[i][:, col_mask] = layer.b.arr
            self.master_m_b[i][:, col_mask] = self.optimizer.m[2*i+1]
            self.master_v_b[i][:, col_mask] = self.optimizer.v[2*i+1]

    def sync_to_network(self):
        """Extracts dynamically sized submatrices corresponding to active neurons."""
        for i, layer in enumerate(self.model.layers):
            row_mask = self.masks[i-1] if i > 0 else slice(None)
            col_mask = self.masks[i] if i < self.n_hidden else slice(None)

            if i == 0:
                layer.W.arr = self.master_W[i][:, col_mask].copy()
                self.optimizer.m[2*i] = self.master_m_W[i][:, col_mask].copy()
                self.optimizer.v[2*i] = self.master_v_W[i][:, col_mask].copy()
            elif i == self.n_hidden:
                layer.W.arr = self.master_W[i][row_mask, :].copy()
                self.optimizer.m[2*i] = self.master_m_W[i][row_mask, :].copy()
                self.optimizer.v[2*i] = self.master_v_W[i][row_mask, :].copy()
            else:
                idx = np.ix_(row_mask, col_mask)
                layer.W.arr = self.master_W[i][idx].copy()
                self.optimizer.m[2*i] = self.master_m_W[i][idx].copy()
                self.optimizer.v[2*i] = self.master_v_W[i][idx].copy()

            layer.b.arr = self.master_b[i][:, col_mask].copy()
            self.optimizer.m[2*i+1] = self.master_m_b[i][:, col_mask].copy()
            self.optimizer.v[2*i+1] = self.master_v_b[i][:, col_mask].copy()

        self.model.zero_grad()

    def step_prune(self, X_batch, Y_batch):
        self.step += 1
        # Periodically prune the model based on gradients to re-evaluate inactive features
        if self.step > self.prune_steps or self.step % 20 != 0:
            return

        current_sparsity = self.final_sparsity * (self.step / self.prune_steps)
        
        # Save structural adjustments back to dense representation
        self.sync_to_master()

        # Temporarily reinstate the full dense network state for gradient probing 
        original_masks = [m.copy() for m in self.masks]
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
            
            # Score: Σ |W_ij * ∇W_ij| 
            importance = np.sum(np.abs(W_arr * W_grad), axis=0)

            if n_keep < n_neurons:
                threshold = np.sort(importance)[-n_keep]
                self.masks[i] = importance >= threshold
            else:
                self.masks[i] = np.ones_like(self.masks[i])

        # Slice sub-matrices and inject entirely reconfigured dynamic network
        self.sync_to_network()
