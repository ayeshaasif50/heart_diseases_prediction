import streamlit as st
import pandas as pd
import joblib

# Load saved model, scaler, and expected columns
model = joblib.load("KNN_heart.pkl")
scaler = joblib.load("scaler.pkl")
expected_columns = joblib.load("columns.pkl")

# ---------- Custom CSS Styling ----------
st.markdown("""
    <style>
    /* Main container - responsive width and padding */
    .main {
        background-color: #ffffff;
        padding: 20px 15px;
        border-radius: 12px;
        box-shadow: 0 4px 10px rgba(0,0,0,0.1);
        max-width: 90%;
        margin: auto;
    }
    /* Title styling */
    .title {
        text-align: center;
        font-size: 36px;
        font-weight: bold;
        color: #d90429;
        margin-top: 10px;
        margin-bottom: 5px;
    }
    /* Subtitle styling */
    .sub {
        text-align: center;
        font-size: 18px;
        color: #6c757d;
        margin-bottom: 30px;
    }
    /* Footer */
    .footer {
        margin-top: 40px;
        text-align: center;
        font-size: 14px;
        color: #aaa;
    }
    /* Button styling */
    .stButton>button {
        background-color: #d90429;
        color: white;
        padding: 0.6em 1.5em;
        border-radius: 8px;
        font-weight: bold;
    }
    .stButton>button:hover {
        background-color: #a6001a;
    }
    /* Header image container */
    .header-img-container {
        display: flex;
        justify-content: center;
        margin-bottom: 10px;
    }
    /* Responsive image with rounded corners & shadow */
    .header-img {
        max-width: 100%;
        width: auto;
        border-radius: 20px;
        box-shadow: 0 6px 18px rgba(0, 0, 0, 0.15);
        border: 4px solid #fff;
    }
    /* Caption below image */
    .header-caption {
        text-align: center;
        font-size: 16px;
        color: #666;
        margin-bottom: 30px;
    }

    /* Extra padding on very small screens */
    @media only screen and (max-width: 600px) {
        .main {
            padding: 15px 10px !important;
        }
        .title {
            font-size: 28px !important;
        }
        .sub {
            font-size: 16px !important;
        }
    }
    </style>
""", unsafe_allow_html=True)

# ---------- Styled Image Header ----------
st.markdown("""
    <div class="header-img-container">
        <img class="header-img" src="https://media.istockphoto.com/id/1145766620/photo/heart-disorder.jpg?s=612x612&w=0&k=20&c=S1dlcnKb1HUv2z1WOeKYXtZ5SL5InU2PjueXiEDMtBE=" />
    </div>
    <div class="header-caption">Know your heart health with AI 💡</div>
""", unsafe_allow_html=True)

# ---------- Main Content ----------
st.markdown('<div class="main">', unsafe_allow_html=True)

st.markdown('<div class="title">💓 Heart Stroke Risk Predictor 💓</div>', unsafe_allow_html=True)
st.markdown('<div class="sub">by Ayesha — Powered by Machine Learning</div>', unsafe_allow_html=True)

# ---------- Inputs ----------
age = st.slider("Age", 18, 100, 40)
sex = st.selectbox("Gender", ["M", "F"])
chest_pain = st.selectbox("Chest Pain Type", ["ATA", "NAP", "TA", "ASY"])
resting_bp = st.number_input("Resting Blood Pressure (mm Hg)", 80, 200, 120)
cholesterol = st.number_input("Cholesterol (mg/dL)", 100, 600, 200)
fasting_bs = st.selectbox("Fasting Blood Sugar > 120 mg/dL", [0, 1])
resting_ecg = st.selectbox("Resting ECG", ["Normal", "ST", "LVH"])
max_hr = st.slider("Max Heart Rate", 60, 220, 150)
exercise_angina = st.selectbox("Exercise-Induced Angina", ["Y", "N"])
oldpeak = st.slider("Oldpeak (ST Depression)", 0.0, 6.0, 1.0)
st_slope = st.selectbox("ST Slope", ["Up", "Flat", "Down"])

# ---------- Prediction ----------
if st.button("Predict"):

    # Create a raw input dictionary
    raw_input = {
        'Age': age,
        'RestingBP': resting_bp,
        'Cholesterol': cholesterol,
        'FastingBS': fasting_bs,
        'MaxHR': max_hr,
        'Oldpeak': oldpeak,
        'Sex_' + sex: 1,
        'ChestPainType_' + chest_pain: 1,
        'RestingECG_' + resting_ecg: 1,
        'ExerciseAngina_' + exercise_angina: 1,
        'ST_Slope_' + st_slope: 1
    }

    # Create input dataframe
    input_df = pd.DataFrame([raw_input])

    # Fill in missing columns with 0s
    for col in expected_columns:
        if col not in input_df.columns:
            input_df[col] = 0

    # Reorder columns
    input_df = input_df[expected_columns]

    # Scale the input
    scaled_input = scaler.transform(input_df)

    # Make prediction
    prediction = model.predict(scaled_input)[0]

    # Show result
    if prediction == 1:
        st.error("⚠️ High Risk of Heart Disease. Please consult a doctor.")
    else:
        st.success("✅ Low Risk of Heart Disease. Keep maintaining a healthy lifestyle!")

# ---------- Footer ----------
st.markdown('<div class="footer">Made with ❤️ by Ayesha | Powered by KNN Model</div>', unsafe_allow_html=True)
st.markdown('</div>', unsafe_allow_html=True)
