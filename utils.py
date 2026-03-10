import pickle
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score


def load_data(data_path):
    data = pd.read_csv(data_path)
    X = data.drop("label", axis=1)
    y = data["label"]
    return X, y

def split_data(X, y, test_size=0.2, random_state=42):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    return X_train, X_test, y_train, y_test

def train_model(X_train, y_train):
    pipe = Pipeline(
        [("scaler", StandardScaler()),
         ("logistic_regression", LogisticRegression())
         ])
    pipe.fit(X_train, y_train)
    return pipe


def evaluate_model(pipe, X_test, y_test):
    # Predict classes
    y_pred = pipe.predict(X_test)
    # Predict probabilities (needed for ROC-AUC score) - take those for positive class only
    y_pred_probas = pipe.predict_proba(X_test)[:, 1]

    accuracy = accuracy_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_pred_probas)

    print(f"Accuracy: {accuracy:.2f}")
    print(f"ROC-AUC score: {roc_auc:.2f}")

def save_model(pipe, model_path):

    with open(model_path, "wb") as file:
        pickle.dump(pipe, file)
        print(f"Done")