import argparse
import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

parser = argparse.ArgumentParser()
parser.add_argument("--data_path", type=str, required=True)
args = parser.parse_args()

df = pd.read_csv(args.data_path)
X = df.drop(columns=["target"])
y = df["target"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

with mlflow.start_run():
    model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)

    mlflow.log_param("n_estimators", 100)
    mlflow.log_param("max_depth", 5)
    mlflow.log_param("data_path", args.data_path)
    mlflow.log_metric("accuracy", acc)

    mlflow.sklearn.log_model(model, "model")

    print("Akurasi:", acc)