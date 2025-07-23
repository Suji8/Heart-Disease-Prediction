# initial_train.py
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

def train_initial_model():
    df = pd.read_csv("heart_split_1.csv")

    X = df.drop("target", axis=1)
    y = df["target"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = RandomForestClassifier(n_estimators=5, max_depth=2, random_state=42)
    model.fit(X_train, y_train)

    acc = accuracy_score(y_test, model.predict(X_test))
    print(f"Initial weak model trained. Accuracy: {acc:.2%}")

    joblib.dump(model, "initial_model.pkl")

if __name__ == "__main__":
    train_initial_model()
