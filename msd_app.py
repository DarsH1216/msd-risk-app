import streamlit as st
import pandas as pd
import joblib
import datetime
import os
import matplotlib.pyplot as plt

# Load model
model = joblib.load("msd_risk_predictor.pkl")

# Set page configuration
st.set_page_config(page_title="MSD Risk Predictor", layout="centered")
st.markdown("<h1 style='text-align: center; color: #4A90E2;'>🦴 MSD Risk Predictor</h1>", unsafe_allow_html=True)
st.markdown("Predict musculoskeletal disorder (MSD) risk using ergonomic data.")

# 📘 REBA/QEC Explanation
with st.expander("ℹ What is REBA and QEC?"):
    st.markdown("""
    *🦴 REBA (Rapid Entire Body Assessment)*  
    Evaluates posture of neck, trunk, legs, arms, load handled, and repetition.
    - Score *1–15+*  
    - Higher score = Higher injury risk

    *📊 QEC (Quick Exposure Check)*  
    Assesses risk from posture, force, repetition, vibration, and job stress.
    - Score *50–176*  
    - Higher score = Greater overall risk
    """)

# ------------------- FORM -------------------
with st.form("risk_form"):
    st.subheader("📋 Enter Worker Details")

    age = st.number_input("Age", 18, 65)
    gender = st.selectbox("Gender", ["Male", "Female"])
    
    reba = st.slider(
        "REBA Final Score (1–15)",
        1, 15,
        help="REBA evaluates posture and load to assess MSD risk. Higher = higher risk."
    )
    
    qec = st.slider(
        "QEC Total Score (50–176)",
        50, 176,
        help="QEC evaluates posture, repetition, force, and job stress. Higher = higher risk."
    )

    st.markdown("### 😣 Reported Pain Areas")
    neck = st.checkbox("Neck Pain")
    shoulder = st.checkbox("Shoulder Pain")
    elbow = st.checkbox("Elbow Pain")
    wrist = st.checkbox("Wrist Pain")
    upper_back = st.checkbox("Upper Back Pain")
    lower_back = st.checkbox("Lower Back Pain")
    hip = st.checkbox("Hip/Thigh Pain")
    knee = st.checkbox("Knee Pain")
    ankle = st.checkbox("Ankle Pain")

    submitted = st.form_submit_button("🔍 Predict Risk")

# ------------------- PREDICTION -------------------
if submitted:
    # Prepare input
    input_df = pd.DataFrame([{
        'Age': age,
        'Gender': 0 if gender == "Male" else 1,
        'Height_cm': 165,
        'Weight_kg': 70,
        'NMQ_Neck_Pain': int(neck),
        'NMQ_Shoulder_Pain': int(shoulder),
        'NMQ_Elbow_Pain': int(elbow),
        'NMQ_Wrist_Pain': int(wrist),
        'NMQ_Upper_Back_Pain': int(upper_back),
        'NMQ_Lower_Back_Pain': int(lower_back),
        'NMQ_Hip_Thigh_Pain': int(hip),
        'NMQ_Knee_Pain': int(knee),
        'NMQ_Ankle_Pain': int(ankle),
        'QEC_Obs_Total_Score': qec,
        'REBA_Final_Score': reba
    }])

    # Predict
    prediction = model.predict(input_df)[0]
    label_map = {0: ('Low', '🟢'), 1: ('Medium', '🟡'), 2: ('High', '🟠'), 3: ('Very High', '🔴')}
    risk_label, emoji = label_map[prediction]

    st.markdown(f"<h3 style='text-align:center;'>Risk Level: {emoji} <span style='color:#444;'>{risk_label}</span></h3>", unsafe_allow_html=True)

    # Attach result
    input_df['Predicted_Risk_Level'] = risk_label
    input_df['Date'] = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    # Save to Excel
    file_name = "msd_predictions.xlsx"
    if os.path.exists(file_name):
        existing = pd.read_excel(file_name)
        final_df = pd.concat([existing, input_df], ignore_index=True)
    else:
        final_df = input_df

    final_df.to_excel(file_name, index=False)
    st.success("✅ Prediction saved to Excel file.")

    # Download CSV
    csv_data = input_df.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="📥 Download This Report as CSV",
        data=csv_data,
        file_name="msd_prediction_report.csv",
        mime="text/csv"
    )

    # ------------------- CHART -------------------
    st.markdown("### 📊 Risk Score Breakdown")

    fig, ax = plt.subplots(figsize=(5, 3))
    scores = [reba, qec]
    labels = ['REBA', 'QEC']
    colors = ['red' if val > 10 else 'orange' if val > 7 else 'green' for val in scores]

    bars = ax.bar(labels, scores, color=colors)
    ax.set_ylim(0, max(180, qec + 10))
    ax.set_ylabel("Score")
    ax.set_title("Ergonomic Risk Scores")

    for bar in bars:
        height = bar.get_height()
        ax.annotate(f'{height}', xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3), textcoords="offset points",
                    ha='center', va='bottom')

    st.pyplot(fig)