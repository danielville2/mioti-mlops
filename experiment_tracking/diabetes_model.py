import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt #data visualization
import seaborn as sns #data visualization
from sklearn.preprocessing import LabelEncoder
from imblearn.combine import SMOTETomek
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn import metrics
from hyperopt import tpe, hp, fmin, STATUS_OK, Trials
from hyperopt.pyll.base import scope
import joblib
import mlflow

mlflow.set_tracking_uri('http://127.0.0.1:5000')
mlflow.set_experiment('RandomForest-exp')

df = pd.read_csv('diabetes_prediction_dataset.csv')
print("##### Data Preprocessing #####\n")
print(f'Numero de datos que tenemos: {len(df)}\n')

def preprocess_data(df):
    """Preprocess the dataset by handling missing values, duplicates, and encoding categorical variables."""
    # Drop duplicates
    df.drop_duplicates(inplace=True)

    # Convert categorical variables to numeric using Label Encoding
    label = LabelEncoder()
    df['gender'] = label.fit_transform(df['gender'])
    df['smoking_history'] = label.fit_transform(df['smoking_history'])

    return df

df = preprocess_data(df)

print("\n##### Model Training #####\n")

X = df.drop('diabetes', axis=1)
y = df['diabetes']
        
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Define hyperparameters for tuning
param_distributions = {
    'n_estimators': hp.choice("n_estimators", [100, 200, 300, 400, 500, 600]),
    'max_depth': hp.choice("max_depth", [1, 2, 3, 5, 8, 10, 15, 20]),
    "criterion": hp.choice("criterion", ["gini", "entropy"]),
}

def train_model(params):

    with mlflow.start_run():
        mlflow.set_tag('model', 'Random Forest')
        mlflow.set_tag('Author', 'Daniel Villegas')
        mlflow.log_params(params)
        
        # Train the model
        clf = RandomForestClassifier(**params, n_jobs=-1)
        clf.fit(X_train, y_train)

        y_predicted = clf.predict(X_test)
        accuracy = metrics.accuracy_score(y_test, y_predicted)
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric('precision', metrics.precision_score(y_test, y_predicted))
        mlflow.log_metric('recall', metrics.recall_score(y_test, y_predicted))
        
        print("Accuracy Score:", accuracy)

    return {'loss': 1 - metrics.recall_score(y_test, y_predicted), 'status': STATUS_OK}

best_result = fmin(
        fn=train_model,
        space=param_distributions,
        algo=tpe.suggest,
        max_evals=50,
        trials=Trials()
    )
print("Best hyperparameters:", best_result)