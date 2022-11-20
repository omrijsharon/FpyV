import numpy as np


class Module:
    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def forward(self, *args, **kwargs):
        raise NotImplementedError

    def backward(self, *args, **kwargs):
        raise NotImplementedError

    def parameters(self):
        if hasattr(self, 'weight') and hasattr(self, 'bias'):
            return [self.weight, self.bias]
        else:
            return []

    def zero_grad(self):
        for p in self.parameters():
            if p.grad is not None:
                p.grad.fill(0)

    def __repr__(self):
        return self.__class__.__name__ + '()'

    def __str__(self):
        return self.__class__.__name__ + '()'


class Parameter:
    def __init__(self, data):
        self.data = data
        self.grad = np.zeros_like(data)

    def __repr__(self):
        return 'Parameter(data={}, grad={})'.format(self.data, self.grad)

    def __str__(self):
        return 'Parameter(data={}, grad={})'.format(self.data, self.grad)


class Linear(Module):
    """
    An MLP layer without an activation function.
    """
    def __init__(self, in_features, out_features):
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(np.random.randn(in_features, out_features))
        self.bias = Parameter(np.random.randn(out_features))

    def forward(self, x):
        return x @ self.weight.data + self.bias.data

    def __call__(self, x):
        return self.forward(x)

    def backward(self, x, grad_output):
        grad_input = grad_output @ self.weight.data.T
        self.weight.grad = x.T @ grad_output
        self.bias.grad = np.sum(grad_output, axis=0)
        return grad_input

    def __repr__(self):
        return 'Linear(in_features={}, out_features={})'.format(self.in_features, self.out_features)


class ReLU(Module):
    """
    A ReLU activation function.
    """
    def forward(self, x):
        return np.maximum(0, x)

    def backward(self, x, grad_output):
        grad_input = grad_output.copy()
        grad_input[x < 0] = 0
        return grad_input

    def __repr__(self):
        return 'ReLU()'


class Sin(Module):
    """
    A Sin activation function.
    """
    def forward(self, x):
        return np.sin(x)

    def backward(self, x, grad_output):
        grad_input = grad_output.copy()
        grad_input = np.cos(x)
        return grad_input

    def __repr__(self):
        return 'Sin()'


class Cos:
    """
    A Cos activation function.
    """
    def forward(self, x):
        return np.cos(x)

    def backward(self, x, grad_output):
        grad_input = grad_output.copy()
        grad_input = -np.sin(x)
        return grad_input

    def __repr__(self):
        return 'Cos()'


class Tanh(Module):
    """
    A Tanh activation function.
    """
    def forward(self, x):
        return np.tanh(x)

    def backward(self, x, grad_output):
        grad_input = grad_output.copy()
        grad_input = 1 - np.tanh(x) ** 2
        return grad_input

    def __repr__(self):
        return 'Tanh()'


class Sigmoid(Module):
    """
    A Sigmoid activation function.
    """
    def forward(self, x):
        return 1/(1+np.exp(-x))

    def backward(self, x, grad_output):
        grad_input = grad_output.copy()
        grad_input = np.exp(-x)/((1+np.exp(-x))**2)
        return grad_input

    def __repr__(self):
        return 'Sigmoide()'


class Attention(Module):
    def forward(self, q, k, v):
        # q: (batch_size, q_len, d_model)
        # k: (batch_size, k_len, d_model)
        # v: (batch_size, v_len, d_model)
        # output: (batch_size, q_len, d_model)
        # attention: (batch_size, q_len, k_len)
        batch_size, q_len, d_model = q.shape
        k_len = k.shape[1]
        attention = np.matmul(q, k.transpose(0, 2, 1)) / np.sqrt(d_model)
        attention = np.exp(attention - np.max(attention, axis=-1, keepdims=True))
        attention /= np.sum(attention, axis=-1, keepdims=True)
        output = np.matmul(attention, v)
        return output, attention

    # def forward(self, x):
    #     self.x = x
    #     self.x_norm = np.linalg.norm(x, axis=1, keepdims=True)
    #     self.x_norm[self.x_norm == 0] = 1
    #     self.x_normed = self.x / self.x_norm
    #     self.weight = np.einsum('ij,ik->jk', self.x_normed, self.x_normed)
    #     self.weight = self.weight / np.sum(self.weight, axis=0, keepdims=True)
    #     self.y = np.einsum('ij,jk->ik', self.x, self.weight)
    #     return self.y


class Sequential(Module):
    """
    A sequential container.
    Modules will be added to it in the order they are passed in the constructor.
    """
    def __init__(self, *layers):
        self.layers = layers

    def forward(self, x):
        for layer in self.layers:
            x = layer.forward(x)
        return x

    def backward(self, x, grad_output):
        for layer in reversed(self.layers):
            x, grad_output = layer.backward(x, grad_output)
        return x

    def update(self, learning_rate):
        for layer in self.layers:
            if hasattr(layer, 'update'):
                layer.update(learning_rate)

    def __repr__(self):
        return 'Sequential({})'.format(', '.join(map(str, self.layers)))

