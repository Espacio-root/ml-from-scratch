from decision_tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pandas as pd

def process_df(df):
    df['Sex'] = df['Sex'].replace('male', 0).replace('female', 1)
    df.dropna(subset=['Embarked', 'Fare'], inplace=True)
    df['Age'].fillna(-1, inplace=True)
    for i, v in enumerate(df['Embarked'].unique()):
        df['Embarked'] = df['Embarked'].replace(v, i)

    columns = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']
    return df[columns], df['Survived']

df = pd.read_csv('../data/titanic.csv')
X, y = process_df(df)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2, random_state=42)

clf = DecisionTreeClassifier(max_depth=7)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

print(f'{accuracy_score(y_pred, y_test):.2f}')

