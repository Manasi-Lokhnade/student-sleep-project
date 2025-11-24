# src/train.py — FINAL GPA Regression Training Script

import argparse
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder, SimpleImputer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import os
import numpy as np

def load_data(path):
    return pd.read_csv(path)

def build_pipeline(numeric_cols, categorical_cols):
    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="constant", fill_value="Missing")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
    ])

    preprocessor = ColumnTransformer([
        ("num", numeric_pipeline, numeric_cols),
        ("cat", categorical_pipeline, categorical_cols)
    ])

    model = RandomForestRegressor(
        n_estimators=200,
        random_state=42,
        n_jobs=-1
    )

    return Pipeline([
        ("preprocessor", preprocessor),
        ("model", model)
    ])

def main(args):
    df = load_data(args.data)

    target = "GPA (0-4 scale)"
    id_cols = ["Student_ID"]

    df = df.drop(columns=[c for c in id_cols if c in df.columns])

    X = df.drop(columns=[target])
    y = df[target]

    numeric_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
    categorical_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    pipeline = build_pipeline(numeric_cols, categorical_cols)

    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)

    print("\nMODEL PERFORMANCE:")
    print("MAE :", mean_absolute_error(y_test, y_pred))
    print("RMSE:", np.sqrt(mean_squared_error(y_test, y_pred)))
    print("R²  :", r2_score(y_test, y_pred))

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    joblib.dump(pipeline, args.out)
    print("\nModel saved to:", args.out)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/student_sleep_academic_performance.csv")
    parser.add_argument("--out", default="model/model.pkl")
    args = parser.parse_args()
    main(args)
