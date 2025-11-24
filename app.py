import streamlit as st
import pandas as pd
from predict import Predictor

st.set_page_config(page_title="GPA Predictor", layout="centered")

# ---------------------------
# App Title
# ---------------------------
st.title("🎓 Student GPA Prediction App")
st.write("Use the sidebar to enter student details and click **Predict GPA**.")

# ---------------------------
# Load Model
# ---------------------------
@st.cache_resource
def load_model():
    return Predictor("model.pkl")

predictor = load_model()

# ---------------------------
# Sidebar inputs
# ---------------------------
st.sidebar.title("📌 Enter Student Details")

inputs = {}

numeric_fields = {
    "Age": (10, 30),
    "Study_Hours_Per_Week": (0, 80),
    "Sleep_Hours_Per_Night": (0, 12),
    "Sleep_Quality (1-10)": (1, 10),
    "Caffeine_Intake (cups/day)": (0, 10),
    "Physical_Activity (mins/day)": (0, 200),
    "Attendance (%)": (0, 100)
}

for col, (min_v, max_v) in numeric_fields.items():
    inputs[col] = st.sidebar.slider(col, min_v, max_v)

categorical_fields = {
    "Gender": ["Male", "Female", "Other"],
    "Academic_Year": ["1st", "2nd", "3rd", "4th"]
}

for col, options in categorical_fields.items():
    inputs[col] = st.sidebar.selectbox(col, options)

# ---------------------------
# Prediction Button
# ---------------------------
if st.sidebar.button("Predict GPA"):
    input_df = pd.DataFrame([inputs])
    prediction = float(predictor.predict(input_df)[0])

    st.subheader("📘 Prediction Result")
    st.success(f"Predicted GPA: **{prediction:.2f}**")
