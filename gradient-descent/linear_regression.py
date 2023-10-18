import numpy as np

class LinearRegression:

    def __init__(self, lr=0.001):
        self.lr = lr

        self.w = 0.0
        self.b = 0.0

        self.w_history = []
        self.b_history = []

    def forward(self, x):
        return self.w * x + self.b

    def loss(self, X, Y):
        return np.mean((Y - self.forward(X)) ** 2)

    def step(self, X,  Y):
        n = len(X)
        grad_w = (-2/n) * np.sum(X * (Y - self.forward(X)))
        grad_b = (-2/n) * np.sum(Y - self.forward(X))
        
        # Update the parameters using the gradients
        self.w -= self.lr * grad_w
        self.b -= self.lr * grad_b

        return self.loss(X, Y)
        
    def fit(self, X, Y, epochs=1000, test_size=0.2, early_stopping=100):
        X, Y = np.array(X), np.array(Y)
        s = int(len(X) * (1 - test_size))
        X_train, Y_train, X_test, Y_test = X[:s], Y[:s], X[s:], Y[s:]
        test_losses = []
        get_loss_idx = lambda: test_losses.index(min(test_losses))

        for epoch in range(epochs):
            loss = self.step(X_train, Y_train)
            test_loss = self.loss(X_test, Y_test)
            test_losses.append(test_loss)
            self.w_history.append(self.w)
            self.b_history.append(self.b)
            if get_loss_idx() < len(test_losses) - early_stopping:
                print('Early stopping')
                break
            if (epoch+1) % (epochs // 10) == 0:
                print("Epoch: {}, Loss: {}".format(epoch+1, loss))

        loss_idx = get_loss_idx()
        return self.w_history[loss_idx], self.b_history[loss_idx]

    def predict(self, X):
        return np.array([self.forward(x) for x in X])
