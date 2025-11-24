import joblib
import pandas as pd

class Predictor:
    def __init__(self, model_path):
        self.model = joblib.load(model_path)

    def predict(self, df: pd.DataFrame):
        return self.model.predict(df)
