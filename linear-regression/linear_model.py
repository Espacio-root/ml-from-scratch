import numpy as np


class LinearRegression:
    
    def __init__(self):
        self.coef_ = None
        self.intercept_ = None

    def fit(self, X, Y):
        X = np.column_stack((X, np.ones(X.shape[0])))
        b = (np.linalg.inv(X.T @ X)) @ (X.T @ Y)
        self.coef_, self.intercept_ = b[:-1, :].reshape(1, -1), b[-1, :]

    def predict(self, X):
        X = X.copy()
        return self.coef_ * X + self.intercept_
