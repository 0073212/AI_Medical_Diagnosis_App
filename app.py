import joblib
import streamlit as st
import numpy as np
import pandas as pd
import os

st.set_page_config(page_title="Disease Prediction", layout="wide")

# Function to load models safely
def load_model(model_path):
    if os.path.exists(model_path):
        return joblib.load(model_path)
    else:
        st.warning(f"Model file missing: {model_path}")
        return None

# Load models
model_paths = {
    "Lung Cancer": "Models/lung_cancer_model.pkl",
    "Kidney Disease": "Models/kidney_model.pkl",
    "Diabetes": "Models/diabetes_model.pkl",
    "Heart Disease": "Models/heart_disease_model.pkl",
    "Fetal Health": "Models/fetal_health_rf_model.pkl",
    "Breast Cancer": "Models/breast_cancer_model.pkl"
}

models = {disease: load_model(path) for disease, path in model_paths.items()}

st.sidebar.title("Navigation")
disease = st.sidebar.selectbox("Choose a disease for prediction", list(models.keys()), key="disease_select")

st.sidebar.write("""
### Instructions:
- Select a disease from the dropdown.
- Enter required input values.
- Click **Predict** to see the result.
""")

# Define input fields for each disease
disease_inputs = {
    "Lung Cancer": [
        "AGE", "SMOKING", "YELLOW_FINGERS", "ANXIETY", "PEER_PRESSURE", "CHRONIC DISEASE", "FATIGUE", "ALLERGY", "WHEEZING", "ALCOHOL CONSUMING", "COUGHING", "SHORTNESS OF BREATH", "SWALLOWING DIFFICULTY", "CHEST PAIN"], 
    "Kidney Disease": [
        "AGE", "BP", "SG", "AL", "SU", "RBC", "PC", "PCC", "BA", "BGR", "BU", "SC", "SOD", "POT", "HEMO", "PCV", "WC", "RC", "HTN", "DM", "CAD", "APPET", "PE", "ANE"],
    "Diabetes": [
        "PREGNANCIES", "GLUCOSE", "BLOOD PRESSURE", "SKIN THICKNESS", "INSULIN", "BMI", "DIABETES PEDIGREE FUNCTION", "AGE"],
    "Heart Disease": {
        "AGE": "number",
        "SEX": {"Female": 0, "Male": 1},
        "CP": {"Typical Angina": 0, "Atypical Angina": 1, "Non-Anginal Pain": 2, "Asymptomatic": 3},
        "TRESTBPS": "number",
        "CHOL": "number",
        "FBS": {"No": 0, "Yes": 1},
        "RESTECG": {"Normal": 0, "ST-T Wave Abnormality": 1, "Left Ventricular Hypertrophy": 2},
        "THALACH": "number",
        "EXANG": {"No": 0, "Yes": 1},
        "OLDPEAK": "number",
        "SLOPE": {"Upsloping": 0, "Flat": 1, "Downsloping": 2},
        "CA": {"0": 0, "1": 1, "2": 2, "3": 3},
        "THAL": {"Normal": 0, "Fixed Defect": 1, "Reversible Defect": 2}},  
    "Fetal Health": [
        "baseline value", "accelerations", "fetal_movement", "uterine_contractions", "light_decelerations", "severe_decelerations", "prolongued_decelerations", "abnormal_short_term_variability", "mean_value_of_short_term_variability", "percentage_of_time_with_abnormal_long_term_variability", "mean_value_of_long_term_variability", "histogram_width", "histogram_min", "histogram_max", "histogram_number_of_peaks", "histogram_number_of_zeroes", "histogram_mode", "histogram_mean", "histogram_median", "histogram_variance", "histogram_tendency"],
    "Breast Cancer": [
        "radius_mean", "texture_mean", "perimeter_mean", "area_mean", "smoothness_mean", "compactness_mean", "concavity_mean", "concave points_mean", "symmetry_mean", "fractal_dimension_mean", "radius_se", "texture_se", "perimeter_se", "area_se", "smoothness_se", "compactness_se", "concavity_se", "concave points_se", "symmetry_se", "fractal_dimension_se", "radius_worst", "texture_worst", "perimeter_worst", "area_worst", "smoothness_worst", "compactness_worst", "concavity_worst", "concave points_worst", "symmetry_worst", "fractal_dimension_worst"]
}

# Streamlit UI

st.title(f"{disease} Diagnosis")

with st.form("input_form"):
    inputs = []
    
    if disease in disease_inputs:
        fields = disease_inputs[disease]
        
        if isinstance(fields, dict):
            for key, val in fields.items():
                if isinstance(val, dict):
                    user_input = st.selectbox(key, list(val.keys()), key=key)
                    inputs.append(val[user_input])
                else:
                    inputs.append(st.number_input(key, value=0.0, key=key))
        else:
            for feature in fields:
                if disease == "Breast Cancer":
                    inputs.append(st.number_input(feature, value=0.0, format="%.6f", key=feature))
                else:
                    inputs.append(st.number_input(feature, value=0.0, key=feature))

    submitted = st.form_submit_button("Predict")

    if submitted:
        model = models[disease]
        input_array = np.array([inputs])
        prediction = model.predict(input_array)

        prediction_label = "Yes" if prediction[0] == 1 else "No"
    
        st.success(f"Prediction: {prediction_label}")





