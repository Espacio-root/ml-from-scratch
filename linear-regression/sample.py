import matplotlib.pyplot as plt
import numpy as np
import random
from linear_model import LinearRegression

x = np.linspace(0, 10, 200)
y = np.linspace(0, 10, 200)

noise_factor = 1.0
def noise(k):
    return k + (random.random()*2-1)*noise_factor

# vectorize does the same thing as map but for numpy arrays and is faster
Xs = np.vectorize(noise)(x)
Ys = np.vectorize(noise)(y)

model = LinearRegression()
model.fit(Xs, Ys)

print(model.coef_, model.intercept_)

plt.scatter(Xs,Ys, s=10, c='b')
plt.plot(Xs, model.predict(Xs), c='r')
plt.show()
