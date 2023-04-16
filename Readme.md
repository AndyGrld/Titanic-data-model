# First kaggle challenge

All datasets located in Titanic directory.

## Getting started
pip install -r requirements.txt

## Preprocessing
1. Filled in missing data with the average value in specified columns
2. Dropped cabin and ticket columns.
3. For names, sorted them according to titles, eg.Mr. and Mrs. had 'low' survival rates and  Major. and Col. had higher survial chances.

## Training
1. Used Logistic Regression model and a KFold alogrithm to split the model for training.
2. The score_model function trains the model and displays the scores.
3. After training model is saved in titanic_models directory using the pickle library.

## Prediction
1 for Survived, 0 for did not Survive
