import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Load the data
df = pd.read_csv('train.csv')

# Drop irrelevant or highly missing columns
df.drop(columns=['Cabin'], inplace=True)

# Handle missing values
df['Age'] = df['Age'].fillna(df['Age'].median())
df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode()[0])

df['Embarked'].fillna(df['Embarked'].mode()[0], inplace=True)

# Convert categorical variables
df['Sex'] = df['Sex'].map({'male': 0, 'female': 1})
df['Embarked'] = df['Embarked'].map({'C': 0, 'Q': 1, 'S': 2})

# One-hot encoding for Pclass (drop first to avoid dummy variable trap)
df = pd.concat([df, pd.get_dummies(df['Pclass'], prefix='Pclass', drop_first=True)], axis=1)

# Drop columns we won’t use
df.drop(columns=['Name', 'Ticket', 'Pclass'], inplace=True)

# Define feature matrix (X) and target vector (y)
X = df[['Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked', 'Pclass_2', 'Pclass_3']]
y = df['Survived']

# Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train the logistic regression model
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
