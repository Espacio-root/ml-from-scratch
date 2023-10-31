import numpy as np


class ActLinear:
    def __call__(self, X):
        return X

    def backward(self, dZ):
        return dZ


class ActRelu:
    def __call__(self, X):
        self.X = X
        return np.maximum(0, X)

    def backward(self, dZ):
        dZ[self.X <= 0] = 0
        return dZ


class ActSoftmax:
    def __call__(self, X):
        e_x = np.exp(X - np.max(X, axis=1, keepdims=True))
        self.output = e_x / np.sum(e_x, axis=1, keepdims=True)
        return self.output

    def backward(self, dZ):
        self.dinputs = np.empty_like(dZ)
        for i, (output, dZ_row) in enumerate(zip(self.output, dZ)):
            output = output.reshape(-1, 1)
            jacobian = np.diagflat(output) - np.dot(output, output.T)
            self.dinputs[i] = np.dot(jacobian, dZ_row)
        return self.dinputs

class CategoricalCrossEntropy:
    def __init__(self, net=None, epsilon=1e-7):
        self.epsilon = epsilon
        self.net = net

    def __call__(self, y_pred, y_true):
        y_pred = np.clip(y_pred, self.epsilon, 1-self.epsilon)
        if len(y_true.shape) == 1:
            y_true = np.eye(y_pred.shape[1])[y_true]
        self.y_pred, self.y_true = y_pred, y_true
        loss = np.sum(y_pred * y_true, axis=1)

        return -np.log(loss).mean()


    def backward(self):
        samples = len(self.y_pred)
        dZ = -self.y_true / self.y_pred
        dZ /= samples

        if self.net == None:
            return dZ
        self.net.backward(dZ)


class CategoricalCrossEntropyWithSoftmax:

    def __init__(self, net=None, epsilon=1e-7):
        self.epsilon = epsilon
        self.net = net
        self.activation = ActSoftmax()
        self.lossfn = CategoricalCrossEntropy()

    def __call__(self, sig_inputs, y_true):
        self.outputs, self.y_true  = self.activation(sig_inputs), y_true
        return self.lossfn(self.outputs, y_true)

    def backward(self):
        samples = len(self.outputs)
        if len(self.y_true.shape) == 2:
            self.y_true = np.argmax(self.y_true, axis=1)
        dZ = self.outputs.copy()
        dZ[range(samples), self.y_true] -= 1
        dZ /= samples

        if self.net == None:
            return dZ
        self.net.backward(dZ)


class Layer:
    def __init__(self, inp, out, step_size, act):
        self.w = 0.01 * np.random.randn(inp, out)
        self.b = np.zeros((1, out))
        self.act = act
        self.step_size = step_size

    def forward(self, X):
        self.X = X
        return self.act(X @ self.w + self.b)

    def backward(self, dZ):
        dA = self.act.backward(dZ)
        dw = self.X.T @ dA
        db = np.sum(dA, axis=0, keepdims=True)

        self.w -= self.step_size * dw
        self.b -= self.step_size * db

        return dA @ self.w.T

class Net:
    def __init__(self, config, step_size=0.01, softmax=False):
        self.step_size = step_size
        self.layers = [Layer(*config[i : i + 2], step_size=step_size, act=ActRelu()) for i in range(len(config) - 2)]
        last_layer_act = ActSigmoid() if softmax else ActLinear()
        self.layers.append(Layer(*config[-2:], step_size=step_size, act=last_layer_act))


    def __call__(self, X):
        for layer in self.layers:
            X = layer.forward(X)
        return X

    def update_step_size(self, new_step_size):
        self.step_size = new_step_size
        for layer in self.layers:
            layer.step_size = new_step_size

    def backward(self, dZ):
        for layer in reversed(self.layers):
            dZ = layer.backward(dZ)


