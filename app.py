import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt

# ================= PAGE CONFIG =================
st.set_page_config(
    page_title="Manufacturing Defect Detection",
    page_icon="🏭",
    layout="wide"
)

# ================= LOAD MODEL =================
@st.cache_resource
def load_model():
    with open("model.pkl", "rb") as f:
        model = pickle.load(f)
    return model

model = load_model()

# ================= LOAD DATA =================
df = pd.read_csv("manufacturing_defect_detection_dataset.csv")

# ================= SIDEBAR =================
st.sidebar.title("📊 Navigation")
page = st.sidebar.radio(
    "Select Page",
    [
        "Project Overview",
        "Data Visualization",
        "Defect Prediction",
        "High Risk Analysis"
    ]
)

# ================= PROJECT OVERVIEW =================
if page == "Project Overview":
    st.title("🏭 Manufacturing Defect Detection & Quality Analytics")
    st.markdown("""
    This project applies **Machine Learning and Data Analytics**
    to identify manufacturing defects and analyze quality trends.

    ### 🎯 Objectives
    - Predict defect probability  
    - Analyze defect-prone conditions  
    - Improve quality control  

    ### 🛠 Tech Stack
    - Python  
    - Streamlit  
    - Scikit-learn  
    - Pandas, Matplotlib  
    """)

# ================= DATA VISUALIZATION =================
elif page == "Data Visualization":
    st.title("📊 Manufacturing Data Visualization")

    st.subheader("Dataset Preview")
    st.dataframe(df.head())

    st.markdown("---")
    fig, ax = plt.subplots()
    ax.hist(df["Temperature"], bins=20, color="#4caf50", edgecolor="black")
    ax.set_xlabel("Temperature")
    ax.set_ylabel("Frequency")
    st.pyplot(fig)

# ================= DEFECT PREDICTION =================
elif page == "Defect Prediction":
    st.markdown("<h2 style='text-align:center;'>🏭 Manufacturing Defect Prediction System</h2>", unsafe_allow_html=True)
    st.markdown("<p style='text-align:center;'>Predict defect probability using Machine Learning</p>", unsafe_allow_html=True)
    st.markdown("---")

    col1, col2 = st.columns(2)

    with col1:
        temperature = st.number_input("🌡 Temperature (°C)", min_value=0.0, step=1.0)
        pressure = st.number_input("🧪 Pressure", min_value=0.0, step=0.1)
        humidity = st.number_input("💧 Humidity (%)", min_value=0.0, step=1.0)
        shift = st.selectbox("🕑 Shift", df["Shift"].unique())

    with col2:
        machine_speed = st.number_input("⚙️ Machine Speed (RPM)", min_value=0.0, step=10.0)
        operator_exp = st.number_input("👷 Operator Experience (Years)", min_value=0.0, step=1.0)
        production_time = st.number_input("⏱ Production Time (Hours)", min_value=0.0, step=1.0)
        material_type = st.selectbox("🧱 Material Type", df["Material_Type"].unique())

    st.markdown("### 🎚 Defect Decision Sensitivity")
    threshold = st.slider(
        "Defect Probability Threshold",
        min_value=0.50,
        max_value=0.90,
        value=0.70,
        step=0.05
    )

    st.markdown("---")

    if st.button("🔮 Predict Defect", use_container_width=True):
        try:
            # ===== INPUT DATAFRAME WITH ALL REQUIRED FEATURES =====
            input_df = pd.DataFrame([{
                "Temperature": temperature,
                "Pressure": pressure,
                "Humidity": humidity,
                "Machine_Speed": machine_speed,
                "Production_Time": production_time,
                "Shift": shift,
                "Operator_Experience_Years": operator_exp,
                "Material_Type": material_type
            }])

            # ===== PREDICTION =====
            prob = model.predict_proba(input_df)[0][1]

            # ================= GAUGE CARD =================
            st.markdown(
                f"""
                <div style='
                    background: linear-gradient(135deg, #e0f7fa, #80deea);
                    padding:25px;
                    border-radius:15px;
                    text-align:center;
                    box-shadow:0px 4px 15px rgba(0,0,0,0.2);'>
                    <h2>🔍 Defect Probability</h2>
                    <div style='
                        width:150px;
                        height:150px;
                        border-radius:50%;
                        background: conic-gradient(#d32f2f {prob*100}%, #cfd8dc {prob*100}% 100%);
                        margin:auto;
                        display:flex;
                        align-items:center;
                        justify-content:center;
                        font-size:32px;
                        font-weight:bold;
                        color:#212121;'>
                        {prob*100:.1f}%
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )

            st.markdown("<br>", unsafe_allow_html=True)

            # ================= THRESHOLD ALERT =================
            if prob >= threshold:
                st.error("⚠️ High Risk of Defect", icon="🚨")
            else:
                st.success("✅ Low Risk of Defect", icon="✅")

            # ================= BAR CHART =================
            prob_df = pd.DataFrame({
                "Status": ["No Defect", "Defect"],
                "Probability": [1 - prob, prob]
            })
            st.subheader("📊 Prediction Probability")
            st.bar_chart(prob_df.set_index("Status"))

        except Exception as e:
            st.error("❌ Prediction Failed")
            st.code(e)

# ================= HIGH RISK ANALYSIS =================
elif page == "High Risk Analysis":
    st.title("🚨 High Risk Manufacturing Analysis")
    high_risk = df[df["Defect"] == 1]
    st.dataframe(high_risk)

    fig, ax = plt.subplots()
    ax.hist(high_risk["Temperature"], bins=15, color="#f44336", edgecolor="black")
    ax.set_xlabel("Temperature")
    ax.set_ylabel("Count")
    st.pyplot(fig)

# ================= FOOTER =================
st.markdown("---")
st.markdown("📌 **Manufacturing Defect Detection Dashboard** | Streamlit App")
