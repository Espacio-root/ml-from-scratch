import numpy as np


class Layer:
    def __init__(self, inp, out, act="relu"):
        self.w = 0.01 * np.random.randn(inp, out)
        self.b = np.zeros((1, out))
        self.act = getattr(self, act)

    def forward(self, X):
        return self.act(X @ self.w + self.b)

    def backward(self, y):
        pass

    def relu(self, X):
        return np.maximum(0, X)

    def sigmoid(self, X):
        e_x = np.exp(X - np.max(X))
        return e_x / np.sum(e_x, axis=1, keepdims=True)


class Net:
    def __init__(self, config):
        self.layers = [Layer(*config[i : i + 2]) for i in range(len(config) - 2)]
        self.layers.append(Layer(*config[-2:], act="sigmoid"))

    def forward(self, X):
        for layer in self.layers:
            X = layer.forward(X)
        return X


class BinaryCrossEntropy:
    def __init__(self, epsilon=1e-7):
        self.epsilon = epsilon

    def __call__(self, y_pred, y):
        samples = len(y_pred)
        y_pred_clipped = np.clip(y_pred, self.epsilon, 1 - self.epsilon)

        if len(y.shape) == 1:
            confidences = y_pred_clipped[range(samples), y]
        elif len(y.shape) == 2:
            confidences = np.sum(y_pred_clipped * y, axis=1)
        else:
            raise Exception("y is invalid.")

        return np.mean(-np.log(confidences))
