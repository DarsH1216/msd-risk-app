import streamlit as st
import pandas as pd
import joblib
import os
from fpdf import FPDF
import base64
import matplotlib.pyplot as plt

st.set_page_config(page_title="MSD Risk Predictor", layout="centered")
st.title("🦴 MSD Risk Predictor with Charts and PDF")

# Debug
with st.expander("🗂 Debug Info"):
    st.write("Files in directory:", os.listdir("."))

# 📘 Explanation
with st.expander("ℹ What are REBA and QEC?"):
    st.markdown("""
    *REBA (Rapid Entire Body Assessment)* evaluates posture-related risks from tasks involving the neck, trunk, and limbs.  
    *QEC (Quick Exposure Check)* checks ergonomic and psychosocial risk exposure—like posture, repetition, and force.
    """)

# Load Model
model_path = "msd_risk_predictor.pkl"
if not os.path.exists(model_path):
    st.error("❌ Model file not found!")
    st.stop()
model = joblib.load(model_path)
st.success("✅ Model loaded successfully")

# Inputs
st.header("📋 Enter Worker Details")

age = st.number_input("Age (years)", min_value=18, max_value=65, step=1)
gender = st.selectbox("Gender", ["Male", "Female"])
height = st.number_input("Height (cm)", min_value=130, max_value=200)
weight = st.number_input("Weight (kg)", min_value=40, max_value=150)
reba_score = st.slider("REBA Final Score", 1, 15)
qec_score = st.slider("QEC Total Score", 50, 176)
gender_val = 0 if gender == "Male" else 1

# Pain area checkboxes
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

# Prepare data
input_data = {
    'Age': age,
    'Gender': gender_val,
    'Height_cm': height,
    'Weight_kg': weight,
    'QEC_Obs_Total_Score': qec_score,
    'REBA_Final_Score': reba_score
}
# Add pain indicators
for key, checked in pain_areas.items():
    col_name = f'NMQ_{key.replace("/", "").replace(" ", "")}_Pain'
    input_data[col_name] = int(checked)

df = pd.DataFrame([input_data])

label_map = {0: 'Low', 1: 'Medium', 2: 'High', 3: 'Very High'}

# Predict
if st.button("🔍 Predict MSD Risk"):
    prediction = model.predict(df)[0]
    risk_label = label_map[prediction]
    st.success(f"🎯 Predicted MSD Risk Level: *{risk_label}*")

    # Save prediction to CSV
    save_df = df.copy()
    save_df["Predicted_Risk_Level"] = risk_label
    if not os.path.exists("msd_predictions.csv"):
        save_df.to_csv("msd_predictions.csv", index=False)
    else:
        save_df.to_csv("msd_predictions.csv", mode='a', header=False, index=False)

    # 📊 Risk Level Chart
    st.subheader("📊 MSD Risk Level Chart")
    color_map = {"Low": "#2ecc71", "Medium": "#f1c40f", "High": "#e67e22", "Very High": "#e74c3c"}
    bar_colors = [color_map.get(label, "#bdc3c7") for label in label_map.values()]
    chart_df = pd.DataFrame({
        "Risk Level": list(label_map.values()),
        "Score": [1 if label == risk_label else 0 for label in label_map.values()],
        "Color": bar_colors
    })
    fig1, ax1 = plt.subplots()
    ax1.bar(chart_df["Risk Level"], chart_df["Score"], color=chart_df["Color"])
    ax1.set_ylabel("Predicted")
    st.pyplot(fig1)

    # 📍 Pain Area Chart
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

    # 📉 REBA vs QEC Chart
    st.subheader("🧮 REBA vs QEC Comparison")
    fig3, ax3 = plt.subplots()
    ax3.bar(["REBA", "QEC"], [reba_score, qec_score], color=["#ff6b6b", "#3498db"])
    ax3.set_ylabel("Score")
    st.pyplot(fig3)

    # 📄 Generate PDF
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
    for area in pain_selected.keys():
        pdf.cell(200, 10, txt=f"- {area}", ln=1)

    pdf.output("report.pdf")
    with open("report.pdf", "rb") as f:
        base64_pdf = base64.b64encode(f.read()).decode("utf-8")
        href = f'<a href="data:application/octet-stream;base64,{base64_pdf}" download="MSD_Report.pdf">📄 Download PDF Report</a>'
        st.markdown(href, unsafe_allow_html=True)
