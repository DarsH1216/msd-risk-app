import streamlit as st
import pandas as pd
import joblib
import os
from fpdf import FPDF
import base64
import matplotlib.pyplot as plt

st.set_page_config(page_title="MSD Risk Predictor", layout="centered")
st.title("🦴 MSD Risk Predictor with Charts and PDF")

# 🔍 Show files for debugging
with st.expander("🗂 Debug Info"):
    st.write("Files in app directory:", os.listdir("."))

# 📘 REBA & QEC Explanation
with st.expander("ℹ What are REBA and QEC?"):
    st.markdown("""
    - *REBA (Rapid Entire Body Assessment)*: Assesses physical stress from postures involving the neck, trunk, and limbs.
    - *QEC (Quick Exposure Check)*: Evaluates risk based on posture, repetition, force, vibration, and work stress.
    """)

# 📦 Load the trained model
model_path = "msd_risk_predictor.pkl"
if not os.path.exists(model_path):
    st.error("❌ Model file not found!")
    st.stop()
model = joblib.load(model_path)
st.success("✅ Model loaded successfully")

# 📝 Input Section
st.header("📋 Enter Worker Details")

age = st.number_input("Age (years)", min_value=18, max_value=65, step=1)
gender = st.selectbox("Gender", ["Male", "Female"])
height = st.number_input("Height (cm)", min_value=130, max_value=200)
weight = st.number_input("Weight (kg)", min_value=40, max_value=150)
reba_score = st.slider("REBA Final Score", 1, 15)
qec_score = st.slider("QEC Total Score", 50, 176)

gender_val = 0 if gender == "Male" else 1

# 📍 Reported Pain Areas
st.subheader("📍 Select Reported Pain Areas")
pain_areas = {
    'Neck': st.checkbox("Neck"),
    'Shoulder': st.checkbox("Shoulder"),
    'Elbow': st.checkbox("Elbow"),
    'Wrist': st.checkbox("Wrist"),
    'Upper Back': st.checkbox("Upper Back"),
    'Lower Back': st.checkbox("Lower Back"),
    'Hip/Thigh': st.checkbox("Hip/Thigh"),
    'Knee': st.checkbox("Knee"),
    'Ankle': st.checkbox("Ankle"),
}

# 📄 Create Input DataFrame
input_data = {
    'Age': age,
    'Gender': gender_val,
    'Height_cm': height,
    'Weight_kg': weight,
    'NMQ_Neck_Pain': int(pain_areas['Neck']),
    'NMQ_Shoulder_Pain': int(pain_areas['Shoulder']),
    'NMQ_Elbow_Pain': int(pain_areas['Elbow']),
    'NMQ_Wrist_Pain': int(pain_areas['Wrist']),
    'NMQ_Upper_Back_Pain': int(pain_areas['Upper Back']),
    'NMQ_Lower_Back_Pain': int(pain_areas['Lower Back']),
    'NMQ_Hip_Thigh_Pain': int(pain_areas['Hip/Thigh']),
    'NMQ_Knee_Pain': int(pain_areas['Knee']),
    'NMQ_Ankle_Pain': int(pain_areas['Ankle']),
    'QEC_Obs_Total_Score': qec_score,
    'REBA_Final_Score': reba_score
}

df = pd.DataFrame([input_data])
label_map = {0: 'Low', 1: 'Medium', 2: 'High', 3: 'Very High'}

# 🔍 Predict Button
if st.button("🔍 Predict MSD Risk"):
    prediction = model.predict(df)[0]
    risk_label = label_map[prediction]
    st.success(f"🎯 Predicted MSD Risk Level: *{risk_label}*")

    # 📊 Risk Level Bar Chart
    st.subheader("📊 MSD Risk Prediction Chart")
    chart_df = pd.DataFrame({
        "Risk Level": list(label_map.values()),
        "Score": [1 if label == risk_label else 0 for label in label_map.values()]
    })
    st.bar_chart(chart_df.set_index("Risk Level"))

    # 📍 Pain Area Chart
    st.subheader("📍 Reported Pain Areas")
    pain_df = pd.DataFrame({
        "Body Part": list(pain_areas.keys()),
        "Pain Reported": [int(val) for val in pain_areas.values()]
    })
    st.bar_chart(pain_df.set_index("Body Part"))

    # 📉 REBA vs QEC Chart
    st.subheader("🧮 REBA vs QEC Comparison")
    fig, ax = plt.subplots()
    ax.bar(["REBA Score", "QEC Score"], [reba_score, qec_score], color=["#e74c3c", "#3498db"])
    ax.set_ylabel("Score")
    st.pyplot(fig)

    # 📄 PDF Generation
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=14)
    pdf.cell(200, 10, txt="MSD Risk Prediction Report", ln=1, align='C')
    pdf.set_font("Arial", size=12)
    pdf.ln(10)
    pdf.cell(200, 10, txt=f"Age: {age}", ln=1)
    pdf.cell(200, 10, txt=f"Gender: {gender}", ln=1)
    pdf.cell(200, 10, txt=f"Height: {height} cm", ln=1)
    pdf.cell(200, 10, txt=f"Weight: {weight} kg", ln=1)
    pdf.cell(200, 10, txt=f"REBA Final Score: {reba_score}", ln=1)
    pdf.cell(200, 10, txt=f"QEC Total Score: {qec_score}", ln=1)
    pdf.cell(200, 10, txt=f"Predicted Risk Level: {risk_label}", ln=1)
    pdf.ln(5)
    pdf.cell(200, 10, txt="Reported Pain Areas:", ln=1)
    for area, val in pain_areas.items():
        if val:
            pdf.cell(200, 10, txt=f"- {area}", ln=1)

    pdf.output("report.pdf")
    with open("report.pdf", "rb") as f:
        base64_pdf = base64.b64encode(f.read()).decode('utf-8')
        href = f'<a href="data:application/octet-stream;base64,{base64_pdf}" download="MSD_Report.pdf">📄 Download PDF Report</a>'
        st.markdown(href, unsafe_allow_html=True)
