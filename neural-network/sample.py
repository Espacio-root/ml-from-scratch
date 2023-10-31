from neuralnet import Net, CategoricalCrossEntropyWithSoftmax
from nnfs.datasets import spiral_data
import nnfs
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import time

nnfs.init()
X, y = spiral_data(samples=100, classes=3)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=True)

# input layer: 2 neurons
# 1st hidden layer: 32 neurons
# 2nd hidden layer: 64 neurons
# output layer: 3 neurons
layer_config = [2, 32, 64, 3] 
net = Net(layer_config, step_size=0.01)
loss = CategoricalCrossEntropyWithSoftmax(net)

for epoch in range(100000):
    outputs = net(X_train)
    cur_loss = loss(outputs, y_train)
    loss.backward()
    if epoch % 100 == 0:
        print(f'epoch: {epoch}, loss: {cur_loss}')
    # if epoch % 10 == 0:
    #     y_pred = net(X_test)
    #     test_loss = loss(y_pred, y_test)
    #     if test_loss < 1.029 and net.step_size == 0.1:
    #         net.update_step_size(net.step_size / 10)
    #     if test_loss < 0.89 and net.step_size == 0.01:
    #         net.update_step_size(net.step_size / 10)
    #     # print([f'{e}: {c}' for e,c in zip(*np.unique(y_test, return_counts=True))])
    #     acc = accuracy_score(y_pred.argmax(axis=1), y_test)
    #     print(f"epoch: {epoch}, loss: {test_loss}, accuracy: {acc}")
    # # if (epoch+1) % 1000 == 0:
    # #     net.update_step_size(net.step_size / 10)
    # #     print(f'New step size: {net.step_size}')
