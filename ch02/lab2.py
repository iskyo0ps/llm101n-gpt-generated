class Scalar:
    def __init__(self, value, _children=(), _op=''):
        self.value = value
        self.grad = 0.0
        self._backward = lambda: None
        self._prev = set(_children)
        self._op = _op

    def __add__(self, other):
        out = Scalar(self.value + other.value, (self, other), '+')
        
        def _backward():
            self.grad += out.grad
            other.grad += out.grad
        out._backward = _backward
        
        return out

    def __mul__(self, other):
        out = Scalar(self.value * other.value, (self, other), '*')
        
        def _backward():
            self.grad += other.value * out.grad
            other.grad += self.value * out.grad
        out._backward = _backward
        
        return out

    def backward(self):
        topo = []
        visited = set()
        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)
        build_topo(self)
        
        self.grad = 1.0
        for v in reversed(topo):
            v._backward()
            
    def tanh(self):
        x = self.value
        t = (math.exp(2*x) - 1) / (math.exp(2*x) + 1)
        out = Scalar(t, (self,), 'tanh')
        
        def _backward():
            self.grad += (1 - t**2) * out.grad
        out._backward = _backward
        
        return out

    def __pow__(self, other):
        assert isinstance(other, (int, float)), "only supporting int/float powers for now"
        out = Scalar(self.value ** other, (self,), f'**{other}')
        
        def _backward():
            self.grad += (other * self.value**(other-1)) * out.grad
        out._backward = _backward
        
        return out

import random
import math

class Neuron:
    def __init__(self, nin):
        self.w = [Scalar(random.uniform(-1, 1)) for _ in range(nin)]
        self.b = Scalar(0)
    
    def __call__(self, x):
        act = sum((wi * xi for wi, xi in zip(self.w, x)), self.b)
        out = act.tanh()
        return out

class Layer:
    def __init__(self, nin, nout):
        self.neurons = [Neuron(nin) for _ in range(nout)]
    
    def __call__(self, x):
        outs = [neuron(x) for neuron in self.neurons]
        return outs if len(outs) > 1 else outs[0]

class MLP:
    def __init__(self, nin, nouts):
        sz = [nin] + nouts
        self.layers = [Layer(sz[i], sz[i+1]) for i in range(len(nouts))]
    
    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x
    
    def parameters(self):
        return [p for layer in self.layers for neuron in layer.neurons for p in neuron.w] + [neuron.b for layer in self.layers for neuron in layer.neurons]

# Sample dataset
data = [
    ([Scalar(2.0), Scalar(3.0)], Scalar(1.0)),
    ([Scalar(1.0), Scalar(2.0)], Scalar(0.0)),
    ([Scalar(2.0), Scalar(2.0)], Scalar(1.0)),
    ([Scalar(3.0), Scalar(3.0)], Scalar(0.0))
]

# Define the model
model = MLP(2, [4, 4, 1])

# Training loop
learning_rate = 0.01
for epoch in range(100):
    # Forward pass
    loss = Scalar(0)
    for x, y in data:
        pred = model(x)[0]
        loss += (pred - y) ** 2

    # Backward pass
    loss.backward()
    
    # Update weights
    for p in model.parameters():
        p.value -= learning_rate * p.grad

    # Reset gradients
    for p in model.parameters():
        p.grad = 0.0

    print(f"Epoch {epoch}: Loss = {loss.value}")