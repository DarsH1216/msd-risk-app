import streamlit as st
import pandas as pd
import joblib
import os
from fpdf import FPDF
import base64
import matplotlib.pyplot as plt

st.set_page_config(page_title="MSD Risk Predictor", layout="centered")
st.title("🦴 MSD Risk Predictor with Charts and PDF")

# Debug info
with st.expander("🗂 Debug Info"):
    st.write("Files in directory:", os.listdir("."))

# REBA & QEC explanation
with st.expander("ℹ What are REBA and QEC?"):
    st.markdown("""
    - *REBA (Rapid Entire Body Assessment)*: Evaluates physical load on body segments from postures.
    - *QEC (Quick Exposure Check)*: Measures exposure to musculoskeletal risk factors like force, repetition, and stress.
    """)

# Load model
model_path = "msd_risk_predictor.pkl"
if not os.path.exists(model_path):
    st.error("❌ Model file not found!")
    st.stop()

model = joblib.load(model_path)
st.success("✅ Model loaded successfully")

# Input fields
st.header("📋 Enter Worker Details")
age = st.number_input("Age (years)", 18, 65)
gender = st.selectbox("Gender", ["Male", "Female"])
height = st.number_input("Height (cm)", 130, 200)
weight = st.number_input("Weight (kg)", 40, 150)
reba_score = st.slider("REBA Final Score", 1, 15)
qec_score = st.slider("QEC Total Score", 50, 176)
gender_val = 0 if gender == "Male" else 1

# Pain area selections (must match model column names)
st.subheader("📍 Select Reported Pain Areas")
pain_areas = {
    'NMQ_Neck_Pain': st.checkbox("Neck"),
    'NMQ_Shoulder_Pain': st.checkbox("Shoulder"),
    'NMQ_Elbow_Pain': st.checkbox("Elbow"),
    'NMQ_Wrist_Pain': st.checkbox("Wrist"),
    'NMQ_Upper_Back_Pain': st.checkbox("Upper Back"),
    'NMQ_Lower_Back_Pain': st.checkbox("Lower Back"),
    'NMQ_Hip_Thigh_Pain': st.checkbox("Hip/Thigh"),
    'NMQ_Knee_Pain': st.checkbox("Knee"),
    'NMQ_Ankle_Pain': st.checkbox("Ankle"),
}

# Input data
input_data = {
    'Age': age,
    'Gender': gender_val,
    'Height_cm': height,
    'Weight_kg': weight,
    'QEC_Obs_Total_Score': qec_score,
    'REBA_Final_Score': reba_score,
}
input_data.update({k: int(v) for k, v in pain_areas.items()})

df = pd.DataFrame([input_data])
label_map = {0: 'Low', 1: 'Medium', 2: 'High', 3: 'Very High'}

# Prediction
if st.button("🔍 Predict MSD Risk"):
    try:
        prediction = model.predict(df)[0]
        risk_label = label_map[prediction]
        st.success(f"🎯 Predicted MSD Risk Level: *{risk_label}*")

        # Risk level bar chart
        st.subheader("📊 MSD Risk Level Chart")
        color_map = {"Low": "#2ecc71", "Medium": "#f1c40f", "High": "#e67e22", "Very High": "#e74c3c"}
        colors = [color_map.get(label, "#bdc3c7") for label in label_map.values()]
        risk_chart = pd.DataFrame({
            "Risk Level": list(label_map.values()),
            "Score": [1 if label == risk_label else 0 for label in label_map.values()]
        })
        fig1, ax1 = plt.subplots()
        ax1.bar(risk_chart["Risk Level"], risk_chart["Score"], color=colors)
        ax1.set_ylabel("Predicted")
        st.pyplot(fig1)

        # Pain area chart
        st.subheader("📍 Reported Pain Areas")
        selected_pains = {k.replace("NMQ_", "").replace("Pain", "").replace("", " "): v for k, v in pain_areas.items() if v}
        if selected_pains:
            pain_df = pd.DataFrame({
                "Body Part": list(selected_pains.keys()),
                "Pain Reported": [1] * len(selected_pains)
            })
            fig2, ax2 = plt.subplots()
            ax2.bar(pain_df["Body Part"], pain_df["Pain Reported"], color="#9b59b6")
            ax2.set_ylim(0, 1.2)
            st.pyplot(fig2)
        else:
            st.info("No pain areas selected.")

        # REBA vs QEC comparison
        st.subheader("🧮 REBA vs QEC Comparison")
        fig3, ax3 = plt.subplots()
        ax3.bar(["REBA", "QEC"], [reba_score, qec_score], color=["#e74c3c", "#3498db"])
        ax3.set_ylabel("Score")
        st.pyplot(fig3)

        # PDF report
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
        pdf.cell(200, 10, txt="Reported Pain Areas:", ln=1)
        for part in selected_pains:
            pdf.cell(200, 10, txt=f"- {part}", ln=1)

        pdf.output("report.pdf")
        with open("report.pdf", "rb") as f:
            base64_pdf = base64.b64encode(f.read()).decode("utf-8")
            href = f'<a href="data:application/octet-stream;base64,{base64_pdf}" download="MSD_Report.pdf">📄 Download PDF Report</a>'
            st.markdown(href, unsafe_allow_html=True)

    except Exception as e:
        st.error(f"🚨 Error during prediction: {e}")
