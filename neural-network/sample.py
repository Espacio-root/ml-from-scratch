from neuralnet import Net, CategoricalCrossEntropyWithSoftmax
from nnfs.datasets import spiral_data
import nnfs
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import time

nnfs.init()
X, y = spiral_data(samples=100, classes=3)
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=True)

# input layer: 2 neurons
# 1st hidden layer: 32 neurons
# 2nd hidden layer: 64 neurons
# output layer: 3 neurons
layer_config = [2, 64, 32, 3] 
net = Net(layer_config, step_size=1)
loss = CategoricalCrossEntropyWithSoftmax(net)

for epoch in range(10001):
    outputs = net(X)
    cur_loss = loss(outputs, y)
    loss.backward()
    if epoch % 100 == 0:
        y_pred = np.argmax(outputs, axis=1)
        acc = np.mean(y==y_pred)
        print(f'epoch: {epoch}, loss: {cur_loss}, accuracy: {acc}')
