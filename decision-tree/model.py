import numpy as np

# Node class
class Node:

    # default arguments set to none since the node can be decision or leaf
    def __init__(self, feature_index=None, threshold=None, left=None, right=None, value=None):

        # decision node
        self.feature_index = feature_index
        self.threshold = threshold
        self.left = left
        self.right = right

        # leaf node
        self.value = value

class DecisionTreeClassifier:

    def __init__(self, min_samples_split=2, max_depth=float('inf'), criterion='entropy'):
        # will be set as the tree when fit function is run
        self.root = None

        self.min_samples_split = min_samples_split
        self.max_depth = max_depth
        self.criterion = criterion

    def _build_tree(self, X, y, depth=1):
        num_samples = np.shape(X)[0]
        
        # if the current node is a decision node
        if num_samples >= self.min_samples_split and self.max_depth >= depth:
            # get the best split with the maximum information gain
            best_split = self._get_best_split(X, y)

            if best_split['information_gain'] > 0:
                # recursively generate the the subtrees
                left_subtree = self._build_tree(depth=depth+1,**best_split['left'])
                right_subtree = self._build_tree(depth=depth+1,**best_split['right'])

                return Node(best_split['feature_index'], best_split['threshold'], left_subtree, right_subtree)

        Y = list(y)
        value = max(Y, key=Y.count)
        return Node(value=value)

    def _get_best_split(self, X, y):
        num_features = np.shape(X)[1]
        max_gain = -float('inf')
        best_split = {}

        # iterates through each possible feature and threshold combination
        for feature_index in range(num_features):
            for threshold in np.unique(X.iloc[:, feature_index]):
                # condition for dividing the samples
                left_c = X.iloc[:, feature_index] <= threshold
                right_c = X.iloc[:, feature_index] > threshold

                # divides all the samples into two nodes based on the feature_index and threshold
                left_X, right_X = X[left_c], X[right_c]
                left_y, right_y = y[left_c], y[right_c]

                # calculate the information gained by a unique split
                information_gain = self._get_information_gain(y, left_y, right_y)
                if information_gain > max_gain:
                    best_split['feature_index'] = feature_index
                    best_split['threshold'] = threshold
                    best_split['left'] = {'X': left_X, 'y': left_y}
                    best_split['right'] = {'X': right_X, 'y': right_y}
                    best_split['information_gain'] = information_gain
                    max_gain = information_gain

        return best_split
    
    def _get_information_gain(self, y, left, right):
        l_weight = len(left) / (len(left) + len(right))
        r_weight = 1 - l_weight

        if self.criterion == 'entropy':
            return self._calculate_entropy(y) - (l_weight * self._calculate_entropy(left) + r_weight * self._calculate_entropy(right))
        elif self.criterion == 'gini':
            return self._calculate_gini(y) - (l_weight * self._calculate_gini(left) + r_weight * self._calculate_gini(right))
        else:
            raise ValueError('criterion can only be "entropy" or "gini"')

    def _calculate_entropy(self, y):
        classes = y.unique()
        entropy = 0
        for cls in classes:
            p_cls = len(y[y == cls]) / len(y)
            entropy += -p_cls * np.log2(p_cls)
        return entropy

    def _calculate_gini(self, y):
        classes = y.unique()
        gini = 0
        for cls in classes:
            p_cls = len(y[y == cls]) / len(y)
            gini += p_cls ** 2
        return 1 - gini

    def fit(self, X, y):
        self.root = self._build_tree(X, y)

    def predict(self, X):
        return [self._make_prediction(x, self.root) for _, x in X.iterrows()]

    def _make_prediction(self, x, tree):
        if tree.value != None:
            return tree.value
        if x.iloc[tree.feature_index] <= tree.threshold:
            return self._make_prediction(x, tree.left)
        else:
            return self._make_prediction(x, tree.right)

