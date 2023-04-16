import os
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
import pickle as pk

folder = r'datasets/Titanic'

'''Preprocessing'''


# Fill missing values for age
def preprocessing(df):
    df['Age'] = df["Age"].fillna(value=int(df['Age'].mean()))
    df['Embarked'] = df["Embarked"].fillna(value='S')
    df['Fare'] = df["Fare"].fillna(value=int(df['Fare'].mean()))
    # Drop columns
    df = df.drop('Cabin', axis=1)
    df = df.drop('Ticket', axis=1)
    return df


# Sorting out names by individuals title
def categories(df, col):
    for idx, name in enumerate(df["Name"]):
        if ('Mrs.' in name) or ('Mme.' in name) or ('Mlle.' in name) or \
                ('Mr.' in name) or ('Miss.' in name) or ('Capt.' in name):
            df.iloc[idx, col] = 'Low'
        elif ('Rev.' in name) or ('Dr.' in name) or ('Master.' in name):
            df.iloc[idx, col] = 'Mid'
        elif ('Major.' in name) or ('Col.' in name) or ('Don.' in name) or ('Lady.' in name):
            df.iloc[idx, col] = 'High'
        else:
            df.iloc[idx, col] = 'Low'
    return df


# Mapping titles and embarked
def mapping(df):
    titles = {'Low': 0, 'Mid': 1, "High": 2}
    df["Name"] = df["Name"].map(titles)
    embark = {'C': 1, 'Q': 2, "S": 0}
    df["Embarked"] = df["Embarked"].map(embark)
    sex = {'female': 1, "male": 0}
    df["Sex"] = df["Sex"].map(sex)
    return df


def extract_df(path, file, col_num):
    path = os.path.join(path, file)
    df = pd.read_csv(path)
    df = preprocessing(df)
    df = categories(df, col_num)
    df = mapping(df)
    return df


train_df = extract_df(folder, 'train.csv', 3)
test_df = extract_df(folder, 'test.csv', 2)

# Scoring model
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


# Training model
X = train_df[["Pclass", "Name", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked"]].values
X1 = test_df[["Pclass", "Name", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked"]].values
y = train_df['Survived'].values
kf = KFold(n_splits=5, shuffle=True)
logR = LogisticRegression(max_iter=1000)

score_model(X, y, kf, logR)
logR_pred = logR.predict(X1)

# save model
pk.dump(logR, open(r'titanic_models/LogisticRegression.p', 'wb'))

# loading model
logR = pk.load(open(r'titanic_models/LogisticRegression.p', 'rb'))
logR_pred = logR.predict(X1)

print('Writing to csv file')
with open('titanic_answers.csv', 'w') as file:
    file.write('PassengerId,Survived\n')
    for index, prediction in enumerate(logR_pred):
        file.write(f'{index + 892},{prediction}\n')
