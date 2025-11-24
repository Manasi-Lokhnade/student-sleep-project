import streamlit as st
import pandas as pd
from predict import Predictor   # no src. prefix!

st.set_page_config(page_title="GPA Predictor", layout="centered")
st.title("Student GPA Prediction App")

@st.cache_resource
def load_model():
    return Predictor("../model/model.pkl")   # model is outside src/

predictor = load_model()

st.sidebar.header("Enter Student Details")

inputs = {}

numeric_fields = {
    "Age": (10, 30),
    "Study_Hours_Per_Week": (0, 80),
    "Sleep_Hours_Per_Night": (0, 10),
    "Sleep_Quality (1-10)": (1, 10),
    "Caffeine_Intake (cups/day)": (0, 10),
    "Physical_Activity (mins/day)": (0, 200),
    "Attendance (%)": (0, 100)
}

categorical_fields = {
    "Gender": ["Male", "Female", "Other"],
    "Academic_Year": ["1st", "2nd", "3rd", "4th"]
}

for col, (min_v, max_v) in numeric_fields.items():
    inputs[col] = st.sidebar.slider(col, min_v, max_v)

for col, options in categorical_fields.items():
    inputs[col] = st.sidebar.selectbox(col, options)

if st.sidebar.button("Predict GPA"):
    df_input = pd.DataFrame([inputs])
    prediction = float(predictor.predict(df_input)[0])
    st.success(f"Predicted GPA: {prediction:.2f}")

