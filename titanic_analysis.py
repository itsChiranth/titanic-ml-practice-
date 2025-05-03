import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

df = pd.read_csv('train.csv')
df['Age'] = df['Age'].fillna(df['Age'].median())
df = df.drop(columns=['Cabin'])
df['Fare'] = df['Fare'].fillna(df['Fare'].median())
df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode()[0])
df['Sex'] = df['Sex'].map({'male': 0, 'female': 1})
df['Embarked'] = df['Embarked'].map({'C': 0, 'Q': 1, 'S': 2})
df = pd.get_dummies(df, columns=['Pclass'], drop_first=True)
X = df.drop(columns=['Survived', 'PassengerId', 'Name', 'Ticket'])
Y = df['Survived']
X_train, X_test, y_train, y_test = train_test_split(X, df['Survived'], test_size=0.2, random_state=42)


print(df.head())
print(df.info())
print(df.isnull().sum())
print(X.head())
print(Y.head())
print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)
