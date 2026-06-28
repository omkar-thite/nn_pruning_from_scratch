class Module:
    def zero_grad(self):
        for p in self.parameters():
            p.grad = np.zeros_like(p.arr)

    def parameters(self):
        return []
    

class Layer(Module):
    def __init__(self, n_in: int, n_out: int, activ: str = 'linear'):
        # He-uniform init: keeps gradient variance stable through ReLU layers.
        limit  = np.sqrt(6.0 / n_in)
        self.W = Tensor(np.random.uniform(-limit, limit, (n_in, n_out)))
        self.b = Tensor(np.zeros((1, n_out)))
        self.activ = activ
        
    def __call__(self, x):
        out = x @ self.W + self.b    # (batch, n_in) @ (n_in, n_out) → (batch, n_out)

        if self.activ == 'relu':
            return out.relu()
        elif self.activ == 'softmax':
            return out.softmax()
        elif self.activ == 'tanh':
            return out.tanh()
        return out
        
    def parameters(self):
        return [self.W, self.b]

    def __repr__(self):
        return f"Layer({self.W.arr.shape}, activ={self.activ})"


class MLP(Module):
    def __init__(self, nin, nouts):
        sz = [nin] + nouts
        self.layers = [
            Layer(sz[i], sz[i+1],
                  activ='relu' if i != len(nouts)-1 else 'softmax')   
            for i in range(len(nouts))
        ]

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

    def parameters(self):
        return [p for layer in self.layers for p in layer.parameters()]

    def __repr__(self):
        return f"MLP of [{', '.join(str(l) for l in self.layers)}]"