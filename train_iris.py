import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics
import joblib
import argparse


def import_data(file_path):
    data = pd.read_csv(file_path)
    return data

def split_data(data):
    train, test = train_test_split(data, test_size=0.4, stratify=data['species'], random_state=42)
    X_train = train[['sepal_length', 'sepal_width', 'petal_length', 'petal_width']]
    y_train = train.species
    X_test = test[['sepal_length', 'sepal_width', 'petal_length', 'petal_width']]
    y_test = test.species
    return X_train, y_train, X_test, y_test

def train_model(X_train, y_train):
    mod_dt = DecisionTreeClassifier(max_depth=3, random_state=1)
    mod_dt.fit(X_train, y_train)
    return mod_dt

def predict(model, X_test, y_test):
    prediction = model.predict(X_test)
    accuracy = metrics.accuracy_score(prediction, y_test)
    return accuracy

def export_model(model, file_path):
    joblib.dump(model, file_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a Decision Tree on the Iris dataset.")
    parser.add_argument("--dataset_path", required=True, help="Path to the CSV file containing the dataset.")
    parser.add_argument("--model_path", default="artifacts/model.joblib", help="Path to save the trained model.")
    args = parser.parse_args()

    data = import_data(args.dataset_path)
    X_train, y_train, X_test, y_test = split_data(data)

    model = train_model(X_train, y_train)
    accuracy = predict(model, X_test, y_test)

    print('The accuracy of the Decision Tree is', "{:.3f}".format(accuracy))

    export_model(model, args.model_path)
