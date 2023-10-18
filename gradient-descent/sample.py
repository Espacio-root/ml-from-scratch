import numpy as np
from linear_regression import LinearRegression
import matplotlib.pyplot as plt
plt.style.use('ggplot')

X = np.linspace(0, 10, 10)
y = np.linspace(0, 20, 10)

reg = LinearRegression()
loss = reg.fit(X, y)
slope, intercept = reg.slope, reg.intercept

plt.scatter(X, y)
plt.plot(X, slope * X + intercept, color='red')
plt.show()
