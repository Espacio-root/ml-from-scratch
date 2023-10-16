from main import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pandas as pd

df = pd.read_csv('iris.csv')
X, y = df.iloc[:, :-1], df.iloc[:, -1]
for i, v in enumerate(y.unique()):
    y = y.replace(v, i)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2, random_state=42)

clf = DecisionTreeClassifier(max_depth=7)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

print(f'{accuracy_score(y_pred, y_test):.2f}')
