import matplotlib.pyplot as plt
import numpy as np
import random
from linear_model import LinearRegression as LocalLinearRegression
from sklearn.linear_model import LinearRegression as SklearnLinearRegression

x = np.linspace(0, 10, 100)
y = np.linspace(0, 10, 100)
z = np.linspace(0, 10, 100)

def noise(k, n_factor=1.0):
    return k + (random.random()*2-1)*n_factor

# vectorize does the same thing as map but for numpy arrays and is faster
Xs = np.vectorize(noise)(x, 3.0)
Ys = np.vectorize(noise)(y)
Zs = np.vectorize(noise)(z, 2.0)

Xs_m = np.column_stack((Xs, Zs))
Ys = Ys.reshape(-1, 1)
X, Z = np.meshgrid(Xs, Zs)

def generate_model_plane(model):
    model.fit(Xs_m, Ys)
    return model, model.coef_[0][0] * X + model.coef_[0][1] * Z + model.intercept_

model1, Plane1 = generate_model_plane(LocalLinearRegression())
model2, Plane2 = generate_model_plane(SklearnLinearRegression())

print(f'Local Linear Regression Model: {model1.coef_}, {model1.intercept_}')
print(f'Sklearn Linear Regression Model: {model2.coef_}, {model2.intercept_}')

fig = plt.figure(figsize=(10, 10))
ax = plt.axes(projection='3d')
ax.grid()

ax.scatter(Xs, Zs, Ys, c='r', s=50)
ax.plot_surface(X, Z, Plane1, color='blue', alpha=0.5, label='Local Model')
ax.plot_surface(X, Z, Plane2, color='red', alpha=0.5, label='Sklearn Model')

ax.set_xlabel('x', labelpad=20)
ax.set_ylabel('z', labelpad=20)
ax.set_zlabel('y', labelpad=20)
ax.legend()

plt.show()
