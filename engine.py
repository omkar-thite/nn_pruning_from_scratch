import numpy as np

class Tensor:
    def __init__(self, arr, _parents=(), _op='', label=''):
        self.arr = np.array(arr) if not isinstance(arr, np.ndarray) else arr
        self._parents = _parents
        self._op = _op
        self.grad = np.zeros_like(self.arr, dtype=float)
        self._backward = lambda: None
        self.label = label

    def __repr__(self):
        return f'Tensor(data={self.arr}, label={self.label})'

    def __add__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(self.arr + other.arr, _parents=(self, other), _op='+')

        def _backward():
            self.grad += out.grad
            other.grad += out.grad

        out._backward = _backward
        return out

    def __radd__(self, other):
        return self + other

    def __neg__(self):
        return self * -1.0

    def __sub__(self, other):
        return self + (-other)
        
    def __rsub__(self, other):
        return other + (-self)

    def __mul__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(self.arr * other.arr, _parents=(self, other), _op='*')

        def _backward():
            self.grad += out.grad * other.arr
            other.grad += out.grad * self.arr

        out._backward = _backward
        return out
        
    def __rmul__(self, other):
        return self * other

    def __pow__(self, other):
        assert isinstance(other, (int, float)), "only supporting int/float powers"
        out = Tensor(self.arr ** other, _parents=(self,), _op=f'**{other}')

        def _backward():
            self.grad += (other * (self.arr ** (other - 1))) * out.grad

        out._backward = _backward
        return out

    def __truediv__(self, other): 
        return self * (other**-1)
        
    def __rtruediv__(self, other):
        return other * (self**-1)

    def __matmul__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(self.arr @ other.arr, _parents=(self, other), _op='@')

        def _backward():
            self.grad += out.grad @ other.arr.T
            other.grad += self.arr.T @ out.grad

        out._backward = _backward
        return out

    def tanh(self):
        x = self.arr
        t = np.tanh(x)
        out = Tensor(t, _parents=(self,), _op='tanh')

        def _backward():
            self.grad += (1.0 - t**2) * out.grad
            
        out._backward = _backward
        return out

    def relu(self):
        out = Tensor(np.maximum(self.arr, 0), _parents=(self,), _op='relu')

        def _backward():
            self.grad += (self.arr > 0) * out.grad

        out._backward = _backward
        return out
        
    def softmax(self):
        z = self.arr - np.max(self.arr, axis=1, keepdims=True)
        exp_z = np.exp(z)
        out_arr = exp_z / np.sum(exp_z, axis=1, keepdims=True)
        out = Tensor(out_arr, _parents=(self,), _op='softmax')

        def _backward():
            dot = np.sum(out.grad * out.arr, axis=1, keepdims=True) 
            self.grad += out.arr * (out.grad - dot)

        out._backward = _backward
        return out
        
    def sum(self, axis=1, keepdims=True):
        out = Tensor(np.sum(self.arr, axis=axis, keepdims=keepdims), _parents=(self,), _op='sum')

        def _backward():
            self.grad += np.ones_like(self.arr) * out.grad
        
        out._backward = _backward
        return out
    
    def mean(self, axis=1, keepdims=True):
        out = Tensor(np.mean(self.arr, axis=axis, keepdims=keepdims), _parents=(self,), _op='mean')

        def _backward():
            n = self.arr.shape[axis]
            self.grad += (1.0 / n) * np.ones_like(self.arr) * out.grad 

        out._backward = _backward
        return out
            
    def build_graph(self):
        topo = []
        visited = set()
        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for parent in v._parents:
                    build_topo(parent)
                topo.append(v)
        build_topo(self)
        return topo

    def backward(self): 
        topo = self.build_graph()
        self.grad = np.ones_like(self.arr, dtype=float)
        for node in reversed(topo):
            node._backward()