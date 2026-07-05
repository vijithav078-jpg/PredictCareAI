import streamlit as st
import requests

st.set_page_config(page_title="PredictCare AI", layout="wide")

st.title("🏭 PredictCare AI")
st.subheader("AI-Powered Predictive Maintenance & Intelligent Diagnosis System")

st.markdown("---")

air_temp = st.number_input("Air Temperature (K)", value=300.0)
process_temp = st.number_input("Process Temperature (K)", value=310.0)
rpm = st.number_input("Rotational Speed (RPM)", value=1500.0)
torque = st.number_input("Torque (Nm)", value=40.0)
tool_wear = st.number_input("Tool Wear (min)", value=20.0)

if st.button("Predict Machine Health"):

    payload = {
        "temperature": air_temp,
        "process_temperature": process_temp,
        "rpm": rpm,
        "torque": torque,
        "tool_wear": tool_wear
    }

    response = requests.post(
        "http://127.0.0.1:5000/predict",
        json=payload
    )

    result = response.json()

    if "error" in result:
        st.error(result["error"])

    else:

        st.success("Prediction Completed")

        st.markdown("---")
        st.subheader("🔍 Machine Health Report")

        col1, col2 = st.columns(2)

        with col1:
            st.metric("🖥 Machine Status", result["prediction"])
            st.metric("⚠ Risk Level", result["risk_level"])
            st.metric("📈 Confidence", result["confidence"])

        with col2:
            st.metric("🔧 Maintenance Priority", result["maintenance_priority"])
            st.metric("⏳ Estimated Life", result["estimated_life"])

        st.markdown("---")

        st.subheader("💡 AI Recommendation")
        st.info(result["recommendation"])

        st.markdown("---")

        st.subheader("🤖 AI Maintenance Assistant (RAG)")

        st.success(result["rag_response"])