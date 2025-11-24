import streamlit as st
import pandas as pd
import joblib
import os

st.set_page_config(page_title="GPA Predictor", layout="centered")
st.title("🎓 Student GPA Prediction App")

# ----------------------------
# Load model
# ----------------------------
@st.cache_resource
def load_model():
    if not os.path.exists("model.pkl"):
        st.error("❌ model.pkl not found. Upload correct model.")
        st.stop()
    return joblib.load("model.pkl")

model = load_model()

# ----------------------------
# Inputs (raw columns as model expects)
# ----------------------------
st.sidebar.header("Enter Student Details")

age = st.sidebar.slider("Age", 10, 40, 20)
study_hours = st.sidebar.slider("Study Hours Per Week", 0, 100, 10)
sleep_hours = st.sidebar.slider("Sleep Hours Per Night", 0.0, 12.0, 7.0)
sleep_quality = st.sidebar.slider("Sleep Quality (1-10)", 1, 10, 6)
caffeine = st.sidebar.slider("Caffeine Intake (cups/day)", 0, 10, 1)
physical = st.sidebar.slider("Physical Activity (mins/day)", 0, 240, 30)
attendance = st.sidebar.slider("Attendance (%)", 0, 100, 85)
stress = st.sidebar.slider("Stress Level (1-10)", 1, 10, 5)

gender = st.sidebar.selectbox("Gender", ["Male", "Female", "Other"])
academic_year = st.sidebar.selectbox("Academic Year", ["1st", "2nd", "3rd", "4th"])

# Build dataframe for model
input_df = pd.DataFrame([{
    "Age": age,
    "Gender": gender,
    "Academic_Year": academic_year,
    "Study_Hours_Per_Week": study_hours,
    "Sleep_Hours_Per_Night": sleep_hours,
    "Sleep_Quality (1-10)": sleep_quality,
    "Caffeine_Intake (cups/day)": caffeine,
    "Physical_Activity (mins/day)": physical,
    "Attendance (%)": attendance,
    "Stress_Level (1-10)": stress
}])

st.subheader("Input Preview")
st.table(input_df)

# ----------------------------
# Predict
# ----------------------------
if st.button("Predict GPA"):
    try:
        pred = model.predict(input_df)[0]
        st.success(f"🎯 Predicted GPA: **{pred:.2f}**")
    except Exception as e:
        st.error("Prediction error: " + str(e))
