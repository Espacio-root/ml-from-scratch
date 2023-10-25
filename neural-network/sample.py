from neuralnet import Net, BinaryCrossEntropy
from nnfs.datasets import spiral_data
import nnfs
import numpy as np

nnfs.init()
X, y = spiral_data(samples=100, classes=3)

# input layer: 2 neurons
# 1st hidden layer: 32 neurons
# 2nd hidden layer: 64 neurons
# output layer: 3 neurons
layer_config = [2, 32, 64, 3] 
net = Net(layer_config)
loss = BinaryCrossEntropy()

outputs = net.forward(X)
loss = loss(outputs, y)
print(loss)

