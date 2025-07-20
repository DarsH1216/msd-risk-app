import streamlit as st
import pandas as pd
import joblib
import os
from fpdf import FPDF
import base64
import matplotlib.pyplot as plt

# Page settings
st.set_page_config(page_title="MSD Risk Predictor", layout="centered")
st.title("🦴 MSD Risk Predictor with Charts and PDF")

# Debugging: Show available files
with st.expander("🗂 Debug Info"):
    st.write("Files in directory:", os.listdir("."))

# Explanation of REBA and QEC
with st.expander("ℹ What are REBA and QEC?"):
    st.markdown("""
    - *REBA (Rapid Entire Body Assessment)*: Assesses posture-related risk based on neck, trunk, and limb angles.
    - *QEC (Quick Exposure Check)*: Assesses physical and psychosocial risk exposure such as force, vibration, and posture.
    """)

# Load the trained model
model_path = "msd_risk_predictor.pkl"
if not os.path.exists(model_path):
    st.error("❌ Model file not found!")
    st.stop()
model = joblib.load(model_path)
st.success("✅ Model loaded successfully")

# Input section
st.header("📋 Enter Worker Details")
age = st.number_input("Age (years)", min_value=18, max_value=65)
gender = st.selectbox("Gender", ["Male", "Female"])
height = st.number_input("Height (cm)", min_value=130, max_value=210)
weight = st.number_input("Weight (kg)", min_value=35, max_value=150)
reba_score = st.slider("REBA Final Score", 1, 15)
qec_score = st.slider("QEC Total Score", 50, 176)
gender_val = 0 if gender == "Male" else 1

# Pain area selection
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

# Construct input dataframe
input_data = {
    'Age': age,
    'Gender': gender_val,
    'Height_cm': height,
    'Weight_kg': weight,
    'QEC_Obs_Total_Score': qec_score,
    'REBA_Final_Score': reba_score
}
# Add dynamic pain fields
for key, value in pain_areas.items():
    col_name = f'NMQ_{key.replace("/", "").replace(" ", "")}_Pain'
    input_data[col_name] = int(value)

df = pd.DataFrame([input_data])

# Label decoder
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
        chart_df = pd.DataFrame({
            "Risk Level": list(label_map.values()),
            "Score": [1 if label == risk_label else 0 for label in label_map.values()],
            "Color": [color_map[label] for label in label_map.values()]
        })
        fig1, ax1 = plt.subplots()
        ax1.bar(chart_df["Risk Level"], chart_df["Score"], color=chart_df["Color"])
        ax1.set_ylabel("Selected")
        st.pyplot(fig1)

        # Pain area chart
        st.subheader("📍 Reported Pain Areas Chart")
        pain_selected = {k: v for k, v in pain_areas.items() if v}
        if pain_selected:
            pain_df = pd.DataFrame({
                "Body Part": list(pain_selected.keys()),
                "Pain Reported": [1]*len(pain_selected)
            })
            fig2, ax2 = plt.subplots()
            ax2.bar(pain_df["Body Part"], pain_df["Pain Reported"], color="#9b59b6")
            ax2.set_ylim(0, 1.2)
            st.pyplot(fig2)
        else:
            st.info("No pain areas selected.")

        # REBA vs QEC
        st.subheader("🧮 REBA vs QEC Comparison")
        fig3, ax3 = plt.subplots()
        ax3.bar(["REBA", "QEC"], [reba_score, qec_score], color=["#ff6b6b", "#3498db"])
        ax3.set_ylabel("Score")
        st.pyplot(fig3)

        # PDF Report
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
        pdf.cell(200, 10, txt=f"REBA Score: {reba_score}", ln=1)
        pdf.cell(200, 10, txt=f"QEC Score: {qec_score}", ln=1)
        pdf.cell(200, 10, txt=f"Predicted Risk Level: {risk_label}", ln=1)
        pdf.cell(200, 10, txt="Pain Areas Reported:", ln=1)
        for area in pain_selected.keys():
            pdf.cell(200, 10, txt=f"- {area}", ln=1)

        pdf.output("report.pdf")
        with open("report.pdf", "rb") as f:
            base64_pdf = base64.b64encode(f.read()).decode("utf-8")
            href = f'<a href="data:application/octet-stream;base64,{base64_pdf}" download="MSD_Report.pdf">📄 Download PDF Report</a>'
            st.markdown(href, unsafe_allow_html=True)
    except Exception as e:
        st.error(f"🚨 Error during prediction: {e}")
