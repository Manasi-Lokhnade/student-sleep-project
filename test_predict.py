# tests/test_predict.py

import pandas as pd
from src.predict import Predictor
import os

def test_prediction_output():
    assert os.path.exists("model/model.pkl"), "model.pkl missing! Run train.py first."

    predictor = Predictor("model/model.pkl")

    sample = pd.DataFrame([{
        "Age": 20,
        "Study_Hours_Per_Week": 10,
        "Sleep_Hours_Per_Night": 7,
        "Sleep_Quality (1-10)": 8,
        "Caffeine_Intake (cups/day)": 1,
        "Physical_Activity (mins/day)": 60,
        "Attendance (%)": 90,
        "Gender": "Male",
        "Academic_Year": "2nd"
    }])

    pred = predictor.predict(sample)
    assert len(pred) == 1, "Prediction output must contain exactly one value"
