import numpy as np
from sklearn.model_selection import train_test_split
import pandas as pd
from linear_regression import LinearRegression
import matplotlib.pyplot as plt
plt.style.use('ggplot')

df = pd.read_csv('../data/experience-salary.csv')
X, y = df.iloc[:, -1], df.iloc[:, :-1]
y = np.squeeze(y)
print(X.shape, y.shape)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

reg = LinearRegression()
slope, intercept = reg.fit(X_train, y_train, epochs=10000, test_size=0.2, early_stopping=100)

y_pred = reg.predict(X_test)
print("Loss: {}".format(reg.loss(X_test, y_test)))
print("Slope: {}, Intercept: {}".format(slope, intercept))

plt.scatter(X, y)
plt.plot(X, slope * X + intercept, color='red')
plt.show()
