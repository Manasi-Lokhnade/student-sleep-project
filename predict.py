# src/predict.py — GPA Prediction Wrapper

import joblib
import pandas as pd

class Predictor:
    def __init__(self, model_path="model/model.pkl"):
        self.model = joblib.load(model_path)

    def predict(self, X: pd.DataFrame):
        return self.model.predict(X)
