import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
from fpdf import FPDF
import os

# Load the model
model = joblib.load("msd_model.pkl")

st.set_page_config(page_title="MSD Prediction", layout="centered")
st.title("🦴 MSD Risk Prediction for Sugarcane Industry Workers")

# Sidebar: User info
st.sidebar.header("Enter Participant Details")

age = st.sidebar.slider("Age", 18, 70, 35)
gender = st.sidebar.selectbox("Gender", ["Male", "Female"])
height = st.sidebar.number_input("Height (cm)", min_value=130, max_value=200, value=165)
weight = st.sidebar.number_input("Weight (kg)", min_value=30, max_value=120, value=65)
bmi = weight / ((height / 100) ** 2)

department = st.sidebar.selectbox("Department", [
    "Crystallizer Operator", "Water Treatment Plant Operator", "Clarifier Assistant",
    "Lubrication Technician", "Evaporator Section Worker", "Other"
])

shift = st.sidebar.radio("Work Shift", ["Day Shift", "Night Shift"])

# Pain Input (NMQ)
st.subheader("🩺 Reported Pain Areas (Nordic Questionnaire Scores)")
pain_inputs = {
    "NMQ_Neck_Pain": st.slider("Neck", 0, 10, 0),
    "NMQ_Shoulder_Pain": st.slider("Shoulder", 0, 10, 0),
    "NMQ_Elbow_Pain": st.slider("Elbow", 0, 10, 0),
    "NMQ_Wrist_Pain": st.slider("Wrist", 0, 10, 0),
    "NMQ_Upper_Back_Pain": st.slider("Upper Back", 0, 10, 0),
    "NMQ_Lower_Back_Pain": st.slider("Lower Back", 0, 10, 0),
    "NMQ_Hip_Thigh_Pain": st.slider("Hip/Thigh", 0, 10, 0),
    "NMQ_Knee_Pain": st.slider("Knee", 0, 10, 0),
    "NMQ_Ankle_Pain": st.slider("Ankle", 0, 10, 0)
}

# Ergonomic scores
st.subheader("🔧 Ergonomic Scores")
reba = st.slider("REBA Final Score", 1, 15, 5)
qec = st.slider("QEC Total Score", 0, 176, 50)

# Encode categorical manually
def one_hot_encoding(value, options):
    return [1 if value == opt else 0 for opt in options[1:]]

gender_enc = one_hot_encoding(gender, ["Male", "Female"])
shift_enc = one_hot_encoding(shift, ["Day Shift", "Night Shift"])
dept_enc = one_hot_encoding(department, [
    "Crystallizer Operator", "Water Treatment Plant Operator", "Clarifier Assistant",
    "Lubrication Technician", "Evaporator Section Worker", "Other"
])

# Feature vector
input_features = [age, bmi, reba, qec] + list(pain_inputs.values()) + gender_enc + dept_enc + shift_enc
columns = model.feature_names_in_

# Convert to DataFrame
input_df = pd.DataFrame([input_features], columns=columns)

# Predict button
if st.button("🔍 Predict MSD Risk"):
    prediction = model.predict(input_df)[0]
    result_text = "⚠ High Risk of Musculoskeletal Disorder" if prediction == 1 else "✅ Low Risk of MSD"
    st.subheader("🔮 Prediction Result:")
    st.write(result_text)

    # Plot bar chart
    st.subheader("📊 Pain Intensity by Region")
    fig, ax = plt.subplots()
    ax.bar(pain_inputs.keys(), pain_inputs.values(), color='orange')
    ax.set_ylabel("Pain Score (0–10)")
    ax.set_xticklabels(pain_inputs.keys(), rotation=45, ha='right')
    st.pyplot(fig)

    # Save chart for PDF
    chart_path = "pain_chart.png"
    fig.savefig(chart_path)

    # Generate PDF
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    pdf.cell(200, 10, "MSD Risk Prediction Report", ln=True, align="C")
    pdf.ln(10)
    pdf.cell(200, 10, f"Prediction: {'High Risk' if prediction == 1 else 'Low Risk'}", ln=True)
    pdf.cell(200, 10, f"Age: {age} | Gender: {gender} | BMI: {bmi:.2f}", ln=True)
    pdf.cell(200, 10, f"Department: {department} | Shift: {shift}", ln=True)
    pdf.cell(200, 10, f"REBA Score: {reba} | QEC Score: {qec}", ln=True)
    pdf.ln(5)
    pdf.cell(200, 10, "Pain Scores:", ln=True)
    for region, score in pain_inputs.items():
        pdf.cell(200, 10, f"{region.replace('NMQ_', '').replace('_', ' ')}: {score}/10", ln=True)

    pdf.image(chart_path, x=10, y=None, w=180)
    pdf_path = "MSD_Report.pdf"
    pdf.output(pdf_path)
    os.remove(chart_path)

    with open(pdf_path, "rb") as file:
        st.download_button(
            label="📥 Download PDF Report",
            data=file,
            file_name="MSD_Prediction_Report.pdf",
            mime="application/pdf"
        )
