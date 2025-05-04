## Titanic - Machine leanring 
This project is classic machine leanring task based on th Titanic dataset. The goal is to predict whether a passeneger is survied te Titanic shipwreck such as sex, age, class, and fare.

 
## 📊 Dataset

Dataset used: [`train.csv`](https://www.kaggle.com/c/titanic/data) from Kaggle.

### Features Used:
- `Pclass`: Ticket class (1 = 1st, 2 = 2nd, 3 = 3rd)
- `Sex`: Gender of the passenger
- `Age`: Age in years (missing values filled with median)
- `SibSp`: # of siblings / spouses aboard
- `Parch`: # of parents / children aboard
- `Fare`: Ticket fare
- `Embarked`: Port of embarkation (encoded as 0 = C, 1 = Q, 2 = S)
- `Pclass_2`, `Pclass_3`: One-hot encoded columns for class 2 and 3

## 🧼 Data Preprocessing

- Handled missing values in `Age` and `Embarked`
- Dropped irrelevant columns like `Cabin`, `Name`, and `PassengerId`
- Converted categorical variables into numerical using:
  - Label Encoding (`Sex`, `Embarked`)
  - One-hot encoding (`Pclass`)
- Feature-target split (`X`, `y`)
- Train-test split (80%-20%)

## 🤖 Model Used

**Logistic Regression**:
- A linear classification model used to estimate survival probabilities.

### Model Evaluation:

- **Accuracy**: ~79.9%
- **Confusion Matrix**:
- **Classification Report**:
- Precision: Proportion of positive identifications that were actually correct
- Recall: Proportion of actual positives that were identified correctly
- F1-score: Harmonic mean of precision and recall
- Support: Number of actual occurrences for each class

## 📈 Metrics Breakdown
[[90 15]
[21 53]]

| Metric        | Description |
|---------------|-------------|
| Accuracy      | Overall correctness of the model |
| Precision     | True Positives / (True Positives + False Positives) |
| Recall        | True Positives / (True Positives + False Negatives) |
| F1 Score      | 2 * (Precision * Recall) / (Precision + Recall) |
| Macro Avg     | Average of precision, recall, f1-score over all classes (unweighted) |
| Weighted Avg  | Average of metrics weighted by class support |

