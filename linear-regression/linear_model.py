import numpy as np


class LinearRegression:
    
    def __init__(self):
        self.coef_ = None
        self.intercept_ = None
    
    def fit(self, X, Y):
        X_mean, Y_mean = np.mean(X), np.mean(Y)
        SPxy = np.sum(np.dot(X-X_mean, Y-Y_mean))
        SSxx = np.sum((X-X_mean)**2)

        self.coef_ = SPxy / SSxx
        self.intercept_ = Y_mean - self.coef_ * X_mean

    def predict(self, X):
        X = X.copy()
        return self.coef_ * X + self.intercept_
