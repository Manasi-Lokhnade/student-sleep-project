import streamlit as st
import pandas as pd
import joblib
import os

st.set_page_config(page_title="Student GPA Predictor", layout="centered")
st.title("🎓 Student Academic Predictor (Streamlit)")

# -------------------------
# Config / Paths
# -------------------------
MODEL_PATH = "model.pkl"
EXAMPLE_CSV = r"C:\Users\DELL\OneDrive\Desktop\sleep-pattern\student_sleep_academic_performance.csv"  # your uploaded CSV path (optional)

# -------------------------
# Load model (cached)
# -------------------------
@st.cache_resource
def load_model(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Model file not found at: {path}. Make sure model.pkl is in the app folder.")
    return joblib.load(path)

try:
    model = load_model(MODEL_PATH)
except Exception as e:
    st.error(f"Could not load model: {e}")
    st.stop()

# -------------------------
# Exact feature names your model expects (confirmed)
# -------------------------
MODEL_FEATURES = [
    "Age",
    "Gender",
    "Academic_Year",
    "Study_Hours_Per_Week",
    "Sleep_Hours_Per_Night",
    "Sleep_Quality (1-10)",
    "Caffeine_Intake (cups/day)",
    "Physical_Activity (mins/day)",
    "Attendance (%)",
    "Stress_Level (1-10)"
]

# -------------------------
# Helper: try to infer categories from uploaded CSV
# -------------------------
def infer_categories(csv_path):
    gender_opts = ["Male", "Female", "Other"]
    year_opts = ["1st", "2nd", "3rd", "4th"]
    try:
        if os.path.exists(csv_path):
            df_example = pd.read_csv(csv_path)
            if "Gender" in df_example.columns:
                g = df_example["Gender"].dropna().unique().tolist()
                if len(g) > 0:
                    gender_opts = g
            if "Academic_Year" in df_example.columns:
                y = df_example["Academic_Year"].dropna().unique().tolist()
                if len(y) > 0:
                    year_opts = y
    except Exception:
        pass
    return gender_opts, year_opts

gender_options, year_options = infer_categories(EXAMPLE_CSV)

# -------------------------
# Sidebar: Inputs
# -------------------------
st.sidebar.header("Enter student details")

# Numeric inputs with reasonable defaults / ranges
age = st.sidebar.slider("Age", 10, 40, value=20)
study_hours = st.sidebar.slider("Study_Hours_Per_Week", 0, 100, value=10)
sleep_hours = st.sidebar.slider("Sleep_Hours_Per_Night", 0.0, 12.0, value=7.0, step=0.5)
sleep_quality = st.sidebar.slider("Sleep_Quality (1-10)", 1, 10, value=6)
caffeine = st.sidebar.slider("Caffeine_Intake (cups/day)", 0, 10, value=1)
physical = st.sidebar.slider("Physical_Activity (mins/day)", 0, 300, value=30)
attendance = st.sidebar.slider("Attendance (%)", 0, 100, value=85)
stress = st.sidebar.slider("Stress_Level (1-10)", 1, 10, value=4)

# Categorical inputs
gender = st.sidebar.selectbox("Gender", gender_options)
academic_year = st.sidebar.selectbox("Academic_Year", year_options)

# Optional Student_ID input (not used by model) and quick CSV upload to override defaults
st.sidebar.markdown("---")
uploaded = st.sidebar.file_uploader("Optional: upload CSV to infer categories/verify", type=["csv"])

if uploaded is not None:
    try:
        uploaded_df = pd.read_csv(uploaded)
        st.sidebar.success("CSV loaded — form options updated where possible.")
        # update gender/year options if available
        if "Gender" in uploaded_df.columns:
            opts = uploaded_df["Gender"].dropna().unique().tolist()
            if opts:
                gender = st.sidebar.selectbox("Gender (from uploaded file)", opts)
        if "Academic_Year" in uploaded_df.columns:
            opts = uploaded_df["Academic_Year"].dropna().unique().tolist()
            if opts:
                academic_year = st.sidebar.selectbox("Academic_Year (from uploaded file)", opts)
    except Exception as e:
        st.sidebar.error("Failed to read uploaded CSV: " + str(e))

# -------------------------
# Build input DataFrame (exact column names)
# -------------------------
input_dict = {
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
}

input_df = pd.DataFrame([input_dict], columns=MODEL_FEATURES)

st.subheader("Input preview")
st.table(input_df.T.rename(columns={0: "value"}))

# -------------------------
# Predict button
# -------------------------
if st.button("Predict GPA"):
    try:
        # If model is a pipeline it should accept the DataFrame directly.
        preds = model.predict(input_df)
        # If model predicts an array-like with a single value per sample
        pred = preds[0]
        # If your model predicts probabilities, you might want to handle that separately.
        st.success(f"🎯 Predicted GPA (model output): **{pred:.4f}**")
        st.info("If this number is not within 0-4, ensure your model outputs GPA on 0-4 scale.")
    except Exception as e:
        st.error("Prediction failed: " + str(e))
        # helpful debugging hints
        st.markdown("**Debug help**:")
        st.write("- Confirm `model.pkl` is a scikit-learn estimator or pipeline that accepts a DataFrame with these exact column names.")
        st.write("- If your pipeline expects one-hot columns (e.g. `Gender_Male`), make sure the pipeline and `model.pkl` include preprocessing.")
        st.write("- You can print `model.feature_names_in_` in a local script/notebook to confirm expected names.")
