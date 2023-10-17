import math
import numpy as np
import os
import sys
sys.path.append(os.path.join(os.path.dirname(sys.path[0]), 'decision-tree'))
from decision_tree import DecisionTreeClassifier # type: ignore

class Tree:

    def __init__(self, tree: DecisionTreeClassifier, feature_indices: list[int]):
        self.tree = tree
        self.feature_indices = feature_indices

    def forward(self, x):
        x = [v for i, v in enumerate(x) if i in self.feature_indices]
        return self.tree._make_prediction(x, self.tree.root)


class RandomForestClassifier():

    def __init__(self, n_estimators=100, min_sample_split=2, max_depth=7):
        self.n_estimators = n_estimators
        self.min_sample_split = min_sample_split
        self.max_depth = max_depth
        self.trees = []

    def _build_forest_tree(self, X, y):
        num_samples, num_features = X.shape
        res_features = int(math.sqrt(num_features))

        random_columns = np.random.choice(list(range(num_features)), res_features, replace=False)
        random_rows = np.random.choice(list(range(num_samples)), num_samples, replace=True)
        X, y = X[random_rows, :][:, random_columns], y[random_rows]

        decision_tree = DecisionTreeClassifier(self.min_sample_split, self.max_depth)
        decision_tree.fit(X, y)
        return decision_tree, random_columns

    def _build_forest(self, X, y):
        for _ in range(self.n_estimators):
            tree, feature_indices = self._build_forest_tree(X, y)
            self.trees.append(Tree(tree, feature_indices))

    def fit(self, X, y):
        self._build_forest(np.array(X), np.array(y))

    def predict(self, X):
        if not isinstance(X, np.ndarray): X = np.array(X)
        return np.array([self._make_prediction(x) for x in X])

    def _make_prediction(self, x):
        ys = []
        for tree in self.trees:
            ys.append(tree.forward(x))
        return max(ys, key=ys.count)
            
        

        


