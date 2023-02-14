import os
import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, \
    f1_score

path = 'Titanic'
train_path = os.path.join(path, 'train.csv')
test_path = os.path.join(path, 'test.csv')
train_df = pd.read_csv(train_path)


def score_model(X, y, kf, model):
    accuracy_scores = []
    precision_scores = []
    recall_scores = []
    f1_scores = []
    scores = []
    for train_index, test_index in kf.split(X):
        X_train, X_eval = X[train_index], X[test_index]
        y_train, y_eval = y[train_index], y[test_index]
        model.fit(X_train, y_train)
        y_pred = model.predict(X_eval)
        accuracy_scores.append(accuracy_score(y_pred, y_eval))
        precision_scores.append(precision_score(y_pred, y_eval))
        recall_scores.append(recall_score(y_pred, y_eval))
        f1_scores.append(f1_score(y_pred, y_eval))
        scores.append(model.score(X_train, y_train))
        print('#', end='')
    print('\nAccuracy score: ', np.mean(accuracy_scores))
    print('Precision score: ', np.mean(precision_scores))
    print('Recall score: ', np.mean(recall_scores))
    print('f1 score: ', np.mean(f1_scores))
    print('score:', np.mean(scores))


# Remove unnecessary columns
train_df.drop('Name', axis=1, inplace=True)
train_df.drop('Ticket', axis=1, inplace=True)
train_df.drop('Cabin', axis=1, inplace=True)

embark_dic = {'S': 1, 'C': 2, 'Q': 3}
train_df["Embarked"] = train_df["Embarked"].map(embark_dic)

mean_age = int(train_df['Age'].mean())
train_df["Age"].fillna(mean_age, inplace=True)

sex_dic = {'male': 0, 'female': 1}
train_df["Sex"] = train_df["Sex"].map(sex_dic)

X = train_df[['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']].values
y = train_df['Survived'].values

# train model
model = HistGradientBoostingClassifier()
kf = KFold(n_splits=5, shuffle=True)
score_model(X, y, kf, model)

# test data
test_df = pd.read_csv(test_path)
test_df.drop('Name', axis=1, inplace=True)
test_df.drop('Ticket', axis=1, inplace=True)
test_df.drop('Cabin', axis=1, inplace=True)
test_df["Embarked"] = test_df["Embarked"].map(embark_dic)
test_df["Age"].fillna(mean_age, inplace=True)
test_df["Sex"] = test_df["Sex"].map(sex_dic)
X1 = test_df[['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']].values

y_test_pred = model.predict(X1)
print(y_test_pred)

with open('answer.csv', 'a') as file:
    start = 892
    file.write('PassengerId,Survived\n')
    for i in y_test_pred:
        text = f'{start},{i}\n'
        file.write(text)
        start += 1
