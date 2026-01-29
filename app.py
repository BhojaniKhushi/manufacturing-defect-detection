import streamlit as st
import pandas as pd
import pickle
import numpy as np

# ================= PAGE CONFIG =================
st.set_page_config(
    page_title="Manufacturing Defect Detection",
    page_icon="🏭",
    layout="centered"
)

st.title("🏭 Manufacturing Defect Detection System")

# ================= LOAD MODEL =================
@st.cache_resource
def load_model():
    with open("defect_model.pkl", "rb") as f:
        model = pickle.load(f)
    with open("scaler.pkl", "rb") as f:
        scaler = pickle.load(f)
    return model, scaler


try:
    model, scaler = load_model()
    st.success("✅ Model & Scaler loaded successfully")
except Exception as e:
    st.error("❌ Model load error")
    st.code(e)
    st.stop()

# ================= USER INPUT =================
st.subheader("🔢 Enter Machine Parameters")

temperature = st.number_input("🌡 Temperature (°C)", min_value=0.0, step=1.0)
pressure = st.number_input("🧪 Pressure", min_value=0.0, step=0.1)
humidity = st.number_input("💧 Humidity (%)", min_value=0.0, step=1.0)
machine_speed = st.number_input("⚙️ Machine Speed (RPM)", min_value=0.0, step=10.0)
operator_exp = st.number_input("👷 Operator Experience (Years)", min_value=0.0, step=1.0)
production_time = st.number_input("⏱ Production Time (Hours)", min_value=0.0, step=1.0)

# ================= THRESHOLD CONTROL =================
st.subheader("🎚 Defect Decision Sensitivity")
threshold = st.slider(
    "Defect Probability Threshold",
    min_value=0.50,
    max_value=0.90,
    value=0.70,
    step=0.05
)

# ================= PREDICTION =================
if st.button("🔮 Predict Defect"):
    try:
        # EXACT FEATURE ORDER FROM TRAINING
        feature_order = list(scaler.feature_names_in_)

        input_dict = {
            "Temperature": temperature,
            "Pressure": pressure,
            "Humidity": humidity,
            "Machine_Speed": machine_speed,
            "Operator_Experience_Years": operator_exp,
            "Production_Time": production_time
        }

        # Arrange values strictly in training order
        input_array = np.array([[input_dict[col] for col in feature_order]])

        # Scale
        input_scaled = scaler.transform(input_array)

        # Probability prediction
        if hasattr(model, "predict_proba"):
            prob = model.predict_proba(input_scaled)[0][1]
        else:
            prob = model.predict(input_scaled)[0]

        st.write(f"🔍 **Defect Probability:** `{prob*100:.2f}%`")

        # Final decision using threshold
        if prob >= threshold:
            st.error("⚠️ Defect Detected")
        else:
            st.success("✅ No Defect Detected")

    except Exception as e:
        st.error("❌ Prediction failed")
        st.code(e)

# ================= FOOTER =================
st.markdown("---")
st.markdown(
    "📌 **Data Science Internship Project**  \n"
    "🛠️ Streamlit + Machine Learning"
)
