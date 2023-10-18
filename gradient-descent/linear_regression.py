import numpy as np

class LinearRegression:

    def __init__(self, lr=0.002):
        self.lr = lr

        self.slope = 0.0
        self.intercept = 0.0

    def forward(self, x):
        return self.slope * x + self.intercept

    def loss(self, X, Y):
        return sum((self.forward(X) - Y) ** 2)

    def step(self, X,  Y):
        grad_slope = sum(2 * (self.forward(X) - Y) * X)
        grad_intercept = sum(2 * (self.forward(X) - Y))

        self.slope -= self.lr * grad_slope
        self.intercept -= self.lr * grad_intercept

        return self.loss(X, Y)
        
    def fit(self, X, Y, epochs=100):
        X, Y = np.array(X), np.array(Y)
        for epoch in range(epochs):
            loss = self.step(X, Y)
            print("Epoch: {}, Loss: {}".format(epoch, loss))

        return loss
