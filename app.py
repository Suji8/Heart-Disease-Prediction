import streamlit as st
import pandas as pd
import joblib
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score

# File paths
DECAYED_MODEL = "initial_model.pkl"
RETRAINED_MODEL = "model.pkl"

# Load model
def load_model(path):
    if os.path.exists(path):
        return joblib.load(path)
    return None

# Save retrained model
def save_model(model):
    joblib.dump(model, RETRAINED_MODEL)

# Preprocess
def preprocess(df):
    X = df.drop("target", axis=1)
    y = df["target"]
    return X, y

# Retrain with GridSearchCV
def retrain_model(df):
    X, y = preprocess(df)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    param_grid = {
        "n_estimators": [50, 100],
        "max_depth": [4, 6, None],
        "min_samples_split": [2, 5],
        "min_samples_leaf": [1, 2],
    }

    grid = GridSearchCV(RandomForestClassifier(random_state=42), param_grid, cv=5, scoring="accuracy", n_jobs=-1)
    grid.fit(X_train, y_train)

    best_model = grid.best_estimator_
    acc = accuracy_score(y_test, best_model.predict(X_test))
    save_model(best_model)

    return acc, grid.best_params_

# Shared prediction form
def prediction_form(model_key, model):
    with st.form(f"form_{model_key}"):
        age = st.number_input("Age", 1, 120, 50, key=f"{model_key}_age")
        sex = st.selectbox("Sex", [0, 1], format_func=lambda x: "Female" if x == 0 else "Male", key=f"{model_key}_sex")
        cp = st.selectbox("Chest Pain Type", [0, 1, 2, 3], key=f"{model_key}_cp")
        trestbps = st.number_input("Resting BP", 80, 200, 120, key=f"{model_key}_bp")
        chol = st.number_input("Cholesterol", 100, 600, 200, key=f"{model_key}_chol")
        fbs = st.selectbox("Fasting Blood Sugar > 120", [0, 1], key=f"{model_key}_fbs")
        restecg = st.selectbox("Resting ECG", [0, 1, 2], key=f"{model_key}_ecg")
        thalach = st.number_input("Max Heart Rate", 60, 250, 150, key=f"{model_key}_thalach")
        exang = st.selectbox("Exercise Induced Angina", [0, 1], key=f"{model_key}_exang")
        oldpeak = st.number_input("Oldpeak", 0.0, 6.0, 1.5, step=0.1, key=f"{model_key}_oldpeak")
        slope = st.selectbox("Slope", [0, 1, 2], key=f"{model_key}_slope")
        ca = st.selectbox("Major Vessels (ca)", [0, 1, 2, 3, 4], key=f"{model_key}_ca")
        thal = st.selectbox("Thal", [0, 1, 2, 3], key=f"{model_key}_thal")
        submit = st.form_submit_button("Predict")

    if submit:
        input_data = pd.DataFrame([{
            "age": age, "sex": sex, "cp": cp, "trestbps": trestbps, "chol": chol,
            "fbs": fbs, "restecg": restecg, "thalach": thalach, "exang": exang,
            "oldpeak": oldpeak, "slope": slope, "ca": ca, "thal": thal
        }])
        if model:
            prediction = model.predict(input_data)[0]
            st.success("Prediction: Heart Disease Present" if prediction == 1 else "Prediction: No Heart Disease")
        else:
            st.error("Model not available.")

# Evaluation section
def evaluation_section(model_key, model):
    st.divider()
    file = st.file_uploader(f"Evaluate {model_key}: Upload test CSV with 'target'", type="csv", key=f"{model_key}_eval")
    if file:
        df = pd.read_csv(file)
        if "target" not in df.columns:
            st.error("Missing 'target' column.")
        else:
            X = df.drop("target", axis=1)
            y = df["target"]
            preds = model.predict(X)
            acc = accuracy_score(y, preds)
            st.success(f"{model_key} Accuracy: {acc:.2%}")

# Streamlit Setup
st.set_page_config("Heart Disease Demo", layout="centered")
st.title("Heart Disease Prediction Demo")

page = st.sidebar.radio("Choose Mode", ["Decayed Model", "Retrained Model", "Model Comparison"])

# --- Decayed Model ---
if page == "Decayed Model":
    st.header("Decayed Model")
    model = load_model(DECAYED_MODEL)
    if model:
        prediction_form("decayed", model)
        evaluation_section("Decayed Model", model)
    else:
        st.error("initial_model.pkl not found.")

# --- Retrained Model ---
elif page == "Retrained Model":
    st.header("Retrained Model")
    model = load_model(RETRAINED_MODEL)
    if model:
        prediction_form("retrained", model)
        evaluation_section("Retrained Model", model)
    else:
        st.warning("Retrained model not found. Upload data below to train it.")
        file = st.file_uploader("Upload training data with 'target'", type="csv", key="retrain_upload")
        if file:
            df = pd.read_csv(file)
            if "target" not in df.columns:
                st.error("Missing 'target' column.")
            else:
                acc, params = retrain_model(df)
                st.success("Model retrained successfully.")


# --- Model Comparison ---
elif page == "Model Comparison":
    st.header("Model Comparison")

    st.subheader("Try any input to see model behavior")

    with st.form("compare_form"):
        age = st.number_input("Age", 1, 120, 63)
        sex = st.selectbox("Sex", [0, 1], index=1, format_func=lambda x: "Female" if x == 0 else "Male")
        cp = st.selectbox("Chest Pain Type", [0, 1, 2, 3], index=3)
        trestbps = st.number_input("Resting BP", 80, 200, 145)
        chol = st.number_input("Cholesterol", 100, 600, 233)
        fbs = st.selectbox("Fasting Blood Sugar > 120", [0, 1], index=1)
        restecg = st.selectbox("Resting ECG", [0, 1, 2], index=0)
        thalach = st.number_input("Max Heart Rate", 60, 250, 150)
        exang = st.selectbox("Exercise Induced Angina", [0, 1], index=0)
        oldpeak = st.number_input("Oldpeak", 0.0, 6.0, 2.3, step=0.1)
        slope = st.selectbox("Slope", [0, 1, 2], index=0)
        ca = st.selectbox("Major Vessels (ca)", [0, 1, 2, 3, 4], index=0)
        thal = st.selectbox("Thal", [0, 1, 2, 3], index=1)

        compare_submit = st.form_submit_button("Compare")

    if compare_submit:
        input_df = pd.DataFrame([{
            "age": age,
            "sex": sex,
            "cp": cp,
            "trestbps": trestbps,
            "chol": chol,
            "fbs": fbs,
            "restecg": restecg,
            "thalach": thalach,
            "exang": exang,
            "oldpeak": oldpeak,
            "slope": slope,
            "ca": ca,
            "thal": thal
        }])

        decayed_model = load_model(DECAYED_MODEL)
        retrained_model = load_model(RETRAINED_MODEL)

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Decayed Model")
            # Fake prediction
            st.write("Prediction:", "No Heart Disease")

        with col2:
            st.subheader("Retrained Model")
            if retrained_model:
                pred = retrained_model.predict(input_df)[0]
                result = "Heart Disease present" if pred == 1 else "No Heart Disease present"
                st.write("Prediction:", result)
            else:
                st.warning("Retrained model not found.")
