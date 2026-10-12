import streamlit as st
import pandas as pd
import numpy as np
import joblib


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="SugarCheck",
    page_icon="🩺",
    layout="centered"
)


# ============================================================
# LOAD TRAINED MODEL
# ============================================================

model_path = "models/sugarcheck_model.pkl"
model = joblib.load(model_path)

# Validated threshold from OOF predictions
FINAL_THRESHOLD = 0.39


# ============================================================
# HEADER
# ============================================================

st.title("🩺 SugarCheck")
st.subheader("Early Diabetes Risk Screening")

st.write(
    "Enter the patient's clinical information below to estimate "
    "the risk of diabetes."
)

st.divider()


# ============================================================
# PATIENT INPUTS
# ============================================================

st.subheader("👤 Patient Information")

col1, col2 = st.columns(2)

with col1:

    pregnancies = st.number_input(
        "Pregnancies",
        min_value=0,
        max_value=20,
        value=0
    )

    glucose = st.number_input(
        "Glucose",
        min_value=0.0,
        max_value=250.0,
        value=120.0
    )

    blood_pressure = st.number_input(
        "Blood Pressure",
        min_value=0.0,
        max_value=150.0,
        value=70.0
    )

    skin_thickness = st.number_input(
        "Skin Thickness",
        min_value=0.0,
        max_value=100.0,
        value=20.0
    )


with col2:

    insulin = st.number_input(
        "Insulin",
        min_value=0.0,
        max_value=900.0,
        value=80.0
    )

    bmi = st.number_input(
        "BMI",
        min_value=0.0,
        max_value=70.0,
        value=32.0
    )

    diabetes_pedigree = st.number_input(
        "Diabetes Pedigree Function",
        min_value=0.0,
        max_value=3.0,
        value=0.47
    )

    age = st.number_input(
        "Age",
        min_value=1,
        max_value=120,
        value=33
    )


st.divider()


# ============================================================
# PREDICTION BUTTON
# ============================================================

if st.button(
    "🔍 Check Diabetes Risk",
    use_container_width=True
):

    # --------------------------------------------------------
    # Convert problematic zero measurements to NaN
    # --------------------------------------------------------

    glucose_value = np.nan if glucose == 0 else glucose

    blood_pressure_value = (
        np.nan if blood_pressure == 0 else blood_pressure
    )

    skin_thickness_value = (
        np.nan if skin_thickness == 0 else skin_thickness
    )

    insulin_value = (
        np.nan if insulin == 0 else insulin
    )

    bmi_value = np.nan if bmi == 0 else bmi


    # --------------------------------------------------------
    # Feature Engineering
    # --------------------------------------------------------

    bmi_obese = (
        np.nan
        if pd.isna(bmi_value)
        else int(bmi_value >= 30)
    )

    glucose_high = (
        np.nan
        if pd.isna(glucose_value)
        else int(glucose_value >= 140)
    )

    missing_count = sum([
        pd.isna(glucose_value),
        pd.isna(blood_pressure_value),
        pd.isna(skin_thickness_value),
        pd.isna(insulin_value),
        pd.isna(bmi_value)
    ])


    # --------------------------------------------------------
    # Create Input DataFrame
    # --------------------------------------------------------

    input_data = pd.DataFrame([{

        "Pregnancies": pregnancies,

        "Glucose": glucose_value,

        "BloodPressure": blood_pressure_value,

        "SkinThickness": skin_thickness_value,

        "Insulin": insulin_value,

        "BMI": bmi_value,

        "DiabetesPedigreeFunction": diabetes_pedigree,

        "Age": age,

        "BMI_Obese": bmi_obese,

        "Glucose_High": glucose_high,

        "Missing_Measurement_Count": missing_count

    }])


    # --------------------------------------------------------
    # Predict Probability
    # --------------------------------------------------------

    probability = model.predict_proba(
        input_data
    )[0, 1]

    prediction = int(
        probability >= FINAL_THRESHOLD
    )


    # ========================================================
    # DISPLAY RESULT
    # ========================================================

    st.divider()

    st.subheader("📊 Screening Result")

    st.metric(
        "Estimated Diabetes Risk",
        f"{probability * 100:.2f}%"
    )


    # --------------------------------------------------------
    # Risk Levels
    # --------------------------------------------------------

    if probability < 0.20:

        st.success("😊 Low Risk")

        st.info(
            "The model estimates a lower probability of diabetes. "
            "This does not rule out diabetes."
        )

    elif probability < FINAL_THRESHOLD:

        st.warning("😐 Medium Risk")

        st.info(
            "The model estimates an intermediate probability of "
            "diabetes. Consider discussing your risk with a "
            "qualified healthcare professional."
        )

    else:

        st.error("😟 High Risk")

        st.warning(
            "The screening result indicates elevated diabetes risk. "
            "Consider consulting a qualified healthcare professional "
            "for appropriate evaluation."
        )


    # --------------------------------------------------------
    # Probability Bar
    # --------------------------------------------------------

    st.write("Risk Probability")

    st.progress(
        min(int(probability * 100), 100)
    )


    # --------------------------------------------------------
    # Medical Disclaimer
    # --------------------------------------------------------

    st.caption(
        "⚠️ SugarCheck is a screening-support tool, "
        "not a medical diagnosis."
    )

    st.caption(
        "The prediction is based on a machine-learning model "
        "trained on the Pima Indians Diabetes dataset."
    )