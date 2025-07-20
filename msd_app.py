import streamlit as st
import pandas as pd
import joblib
import os
from fpdf import FPDF
import base64

st.set_page_config(page_title="MSD Risk Predictor", layout="centered")
st.title("🦴 MSD Risk Predictor with PDF Report and Chart")

# 🔍 Debug Info
st.subheader("🗂 Debug Info")
st.write("Files in app directory:", os.listdir("."))

# Load the model
model_path = "msd_risk_predictor.pkl"

if not os.path.exists(model_path):
    st.error(f"❌ Model file not found: {model_path}")
    st.stop()
else:
    model = joblib.load(model_path)
    st.success("✅ Model loaded successfully!")

# 🧾 Input Form
st.header("📋 Enter Worker Details")

age = st.number_input("Age", 18, 60)
gender = st.selectbox("Gender", ["Male", "Female"])
reba = st.slider("REBA Final Score", 1, 15)
qec = st.slider("QEC Total Score", 50, 176)

gender_val = 0 if gender == "Male" else 1

# Prepare input for prediction
data = pd.DataFrame([{
    'Age': age,
    'Gender': gender_val,
    'Height_cm': 165,
    'Weight_kg': 70,
    'NMQ_Neck_Pain': 1,
    'NMQ_Shoulder_Pain': 0,
    'NMQ_Elbow_Pain': 0,
    'NMQ_Wrist_Pain': 0,
    'NMQ_Upper_Back_Pain': 0,
    'NMQ_Lower_Back_Pain': 1,
    'NMQ_Hip_Thigh_Pain': 0,
    'NMQ_Knee_Pain': 1,
    'NMQ_Ankle_Pain': 0,
    'QEC_Obs_Total_Score': qec,
    'REBA_Final_Score': reba
}])

label_map = {0: 'Low', 1: 'Medium', 2: 'High', 3: 'Very High'}

# 🔍 Predict and Show Result
if st.button("🔍 Predict MSD Risk"):
    prediction = model.predict(data)[0]
    risk_level = label_map[prediction]
    st.success(f"🎯 Predicted MSD Risk Level: *{risk_level}*")

    # 📊 Bar Chart
    chart_data = pd.DataFrame({
        "Risk Level": list(label_map.values()),
        "Score": [1 if label == risk_level else 0 for label in label_map.values()]
    })
    st.bar_chart(chart_data.set_index("Risk Level"))

    # 📄 Generate PDF
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=14)
    pdf.cell(200, 10, txt="MSD Risk Prediction Report", ln=1, align='C')
    pdf.set_font("Arial", size=12)
    pdf.ln(10)
    pdf.cell(200, 10, txt=f"Age: {age}", ln=1)
    pdf.cell(200, 10, txt=f"Gender: {gender}", ln=1)
    pdf.cell(200, 10, txt=f"REBA Final Score: {reba}", ln=1)
    pdf.cell(200, 10, txt=f"QEC Total Score: {qec}", ln=1)
    pdf.cell(200, 10, txt=f"Predicted Risk Level: {risk_level}", ln=1)

    # Save and show download
    pdf.output("report.pdf")
    with open("report.pdf", "rb") as f:
        base64_pdf = base64.b64encode(f.read()).decode('utf-8')
        href = f'<a href="data:application/octet-stream;base64,{base64_pdf}" download="MSD_Report.pdf">📄 Download PDF Report</a>'
        st.markdown(href, unsafe_allow_html=True)
