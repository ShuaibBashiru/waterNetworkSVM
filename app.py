import streamlit as st
import numpy as np
import pandas as pd
import joblib
import tensorflow as tf
import os
from datetime import datetime
import plotly.express as px

# Load models and scaler
svm = joblib.load('svm_model.pkl')
scaler = joblib.load('scaler.pkl')
cnn = tf.keras.models.load_model('cnn_feature_extractor.keras')

# Log file
log_file = 'logs.csv'

# Load previous logs
if os.path.exists(log_file):
    logs = pd.read_csv(log_file)
else:
    logs = pd.DataFrame(columns=['Timestamp', 'Pressure_PSI', 'Flow_GPM', 'Velocity_FPS', 'Prediction'])

# Streamlit UI
st.set_page_config(page_title="Leakage Detection System", layout="centered")
st.title("💧 Smart Leakage Prediction System By Olayiwola O. PhD (In view)")
st.write("Upload multiple rows or enter single input for Leakage classification using CNN + SVM.")

# Input type
input_mode = st.radio("Choose input mode", ["Single Entry", "Batch Upload"])

# Input: Single Entry
if input_mode == "Single Entry":
    col1, col2, col3 = st.columns(3)
    with col1:
        pressure = st.number_input("Pressure (PSI)", min_value=0.0, step=0.1)
    with col2:
        flow = st.number_input("Flow (GPM)", min_value=0.0, step=0.1)
    with col3:
        velocity = st.number_input("Velocity (FPS)", min_value=0.0, step=0.1)

    if st.button("Predict"):
        input_data = np.array([[pressure, flow, velocity]])
        scaled_input = scaler.transform(input_data)
        reshaped_input = scaled_input.reshape((1, 3, 1))
        cnn_features = cnn.predict(reshaped_input)
        prediction = svm.predict(cnn_features)
        label = "Leakage 🚨" if prediction[0] == 1 else "No Leakage ✅"

        st.success(f"**Prediction:** {label}")

        # Save log
        new_log = pd.DataFrame([{
            'Timestamp': datetime.now(),
            'Pressure_PSI': pressure,
            'Flow_GPM': flow,
            'Velocity_FPS': velocity,
            'Prediction': label
        }])
        logs = pd.concat([logs, new_log], ignore_index=True)
        logs.to_csv(log_file, index=False)

# Input: Batch Upload
else:
    uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])
    if uploaded_file is not None:
        data = pd.read_csv(uploaded_file)
        st.write("📄 Uploaded Data", data)

        if all(col in data.columns for col in ['Pressure_PSI', 'Flow_GPM', 'Velocity_FPS']):
            input_data = data[['Pressure_PSI', 'Flow_GPM', 'Velocity_FPS']].values
            scaled_input = scaler.transform(input_data)
            reshaped_input = scaled_input.reshape((scaled_input.shape[0], 3, 1))
            cnn_features = cnn.predict(reshaped_input)
            predictions = svm.predict(cnn_features)
            labels = ["Leakage Detected 🚨" if pred == 1 else "No Leakage Detected ✅" for pred in predictions]

            data['Prediction'] = labels
            st.write("📊 Prediction Results", data)

            # Log all predictions
            data['Timestamp'] = datetime.now()
            logs = pd.concat([logs, data[['Timestamp', 'Pressure_PSI', 'Flow_GPM', 'Velocity_FPS', 'Prediction']]], ignore_index=True)
            logs.to_csv(log_file, index=False)
        else:
            st.error("CSV must contain columns: Pressure_PSI, Flow_GPM, Velocity_FPS")

# --- Charts Section ---
if not logs.empty:
    st.markdown("---")
    st.subheader("📈 Prediction Summary")

    # Pie Chart of Predictions
    pred_counts = logs['Prediction'].value_counts()
    st.write("### Prediction Distribution")
    fig_pie = px.pie(names=pred_counts.index, values=pred_counts.values, title='Leak vs No Leak')
    st.plotly_chart(fig_pie)

    # Bar Chart Over Time
    st.write("### Daily Prediction History")
    logs['Date'] = pd.to_datetime(logs['Timestamp']).dt.date
    daily_counts = logs.groupby(['Date', 'Prediction']).size().unstack(fill_value=0)
    fig_bar = px.bar(daily_counts.reset_index().melt(id_vars='Date', var_name='Prediction', value_name='Count'),
                     x='Date', y='Count', color='Prediction', barmode='stack',
                     title="Predictions Over Time")
    st.plotly_chart(fig_bar)
