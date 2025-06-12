import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import tensorflow as tf
from lime.lime_tabular import LimeTabularExplainer
from twilio.rest import Client
from twilio.twiml.voice_response import VoiceResponse
import pickle
import time
import matplotlib.pyplot as plt
import io
import base64

# Set Streamlit page config as the first command
st.set_page_config(page_title="Medical Anomaly Detection Dashboard", layout="wide")

# Twilio Configuration (use your actual credentials)
TWILIO_ACCOUNT_SID = "your_actual_account_sid"  # Replace with your Twilio SID
TWILIO_AUTH_TOKEN = "your_actual_auth_token"    # Replace with your Twilio Auth Token
TWILIO_PHONE_NUMBER = "+15075015667"            # Your Twilio phone number
DOCTOR_PHONE = "+918074920408"                  # Valid number
FAMILY_PHONE = "+918074920408"                  # Valid number

# Initialize Twilio client with error handling
try:
    twilio_client = Client(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)
    account = twilio_client.api.accounts(TWILIO_ACCOUNT_SID).fetch()
    st.success("Twilio client initialized successfully!")
except Exception as e:
    st.error(f"Failed to initialize Twilio client: {e}")
    twilio_client = None

# Mock patient data
patient_data = {
    "patient_id": 101,
    "name": "John Doe",
    "age": 45,
    "disease": "Hypertension",
    "photo": "https://via.placeholder.com/150"
}

# Generate synthetic training data for LIME explainer
def generate_synthetic_train_data(num_samples=1000):
    np.random.seed(42)
    data = {
        "patient_id": [101] * num_samples,
        "HR": np.random.normal(70, 10, num_samples),
        "PP": np.random.normal(40, 5, num_samples),
        "BP_S": np.random.normal(120, 15, num_samples),
        "BP_D": np.random.normal(80, 10, num_samples),
        "RESP": np.random.normal(16, 2, num_samples),
        "SpO2": np.random.normal(98, 1, num_samples)
    }
    return pd.DataFrame(data)

# Health Monitor class (unchanged from your original)
class HealthMonitor:
    def __init__(self, autoencoder, classifier, scaler, threshold, patient_id, train_data):
        self.autoencoder = autoencoder
        self.classifier = classifier
        self.scaler = scaler
        self.threshold = threshold
        self.window_size = 20
        self.vitals = ['HR', 'PP', 'BP_S', 'BP_D', 'RESP', 'SpO2']
        self.patient_id = patient_id
        try:
            self.explainer = LimeTabularExplainer(
                training_data=scaler.transform(train_data[self.vitals]),
                feature_names=self.vitals,
                mode='regression',
                discretize_continuous=False
            )
        except Exception as e:
            st.error(f"Failed to initialize LIME explainer: {e}")
            self.explainer = None
        self.last_alert_time = 0
        self.alert_cooldown = 0
        self.explanation_cache = {}

    def send_sms_alert(self, to_number, message):
        fallback_message = f"[Fallback Notification] {message} (Intended for {to_number})"
        if twilio_client is None:
            st.warning(fallback_message)
            return f"Failed to send SMS to {to_number}: Twilio client not initialized"
        try:
            message_response = twilio_client.messages.create(
                body=message,
                from_=TWILIO_PHONE_NUMBER,
                to=to_number
            )
            return f"SMS sent to {to_number}: {message} (SID: {message_response.sid})"
        except Exception as e:
            st.warning(fallback_message)
            return f"Failed to send SMS to {to_number}: {str(e)}"

    def make_call_alert(self, to_number, message):
        fallback_message = f"[Fallback Notification] Call - {message} (Intended for {to_number})"
        if twilio_client is None:
            st.warning(fallback_message)
            return f"Failed to initiate call to {to_number}: Twilio client not initialized"
        try:
            response = VoiceResponse()
            response.say(message, voice='Polly.Joanna')
            twiml = str(response)
            call = twilio_client.calls.create(
                twiml=twiml,
                to=to_number,
                from_=TWILIO_PHONE_NUMBER
            )
            return f"Call initiated to {to_number}: {message} (SID: {call.sid})"
        except Exception as e:
            st.warning(fallback_message)
            return f"Failed to initiate call to {to_number}: {str(e)}"

    def analyze_with_explanation(self, sample, context_window=None, sample_idx=None):
        try:
            scaled = self.scaler.transform([sample[self.vitals]])
            reconstruction = self.autoencoder.predict(scaled, verbose=0)
            error = np.mean((scaled - reconstruction) ** 2)
            explanation = None
            result = "Status Normal"
            alerts = []

            if error > self.threshold:
                feature_errors = np.abs(scaled[0] - reconstruction[0])
                if context_window is not None:
                    window_data = self.scaler.transform(context_window[self.vitals])
                    temporal_features = [
                        np.mean(window_data, axis=0),
                        np.std(window_data, axis=0),
                        np.median(window_data, axis=0)
                    ]
                else:
                    temporal_features = [np.zeros(len(self.vitals))] * 3

                if self.explainer is not None:
                    cache_key = f"{sample_idx}_{error:.4f}"
                    if cache_key in self.explanation_cache:
                        exp = self.explanation_cache[cache_key]
                    else:
                        try:
                            exp = self.explainer.explain_instance(
                                scaled[0],
                                self.autoencoder.predict,
                                num_features=len(self.vitals)
                            )
                            self.explanation_cache[cache_key] = exp
                        except Exception as e:
                            st.error(f"LIME explanation failed: {e}")
                            exp = None
                else:
                    exp = None

                lime_vals = np.array([x[1] for x in exp.local_exp[0]]) if exp is not None else np.zeros(len(self.vitals))
                features = np.concatenate([
                    feature_errors,
                    *temporal_features,
                    lime_vals,
                    [error]
                ]).reshape(1, -1)

                prediction = self.classifier.predict(features, verbose=0)[0][0]
                current_time = time.time()

                if current_time - self.last_alert_time >= self.alert_cooldown:
                    if prediction > 0.2:
                        result = "ALERT: Medical Event Detected!"
                        message = f"URGENT: Medical Event detected for Patient {self.patient_id}. Vital signs anomaly detected (MSE: {error:.4f}). Please check immediately."
                        sms1 = self.send_sms_alert(DOCTOR_PHONE, message)
                        sms2 = self.send_sms_alert(FAMILY_PHONE, message)
                        call = self.make_call_alert(DOCTOR_PHONE, f"Urgent. Medical event detected for Patient {self.patient_id}. Please check vital signs immediately.")
                        alerts = [sms1, sms2, call]
                    else:
                        result = "Warning: Sensor Anomaly"
                        message = f"Warning: Sensor Anomaly detected for Patient {self.patient_id} (MSE: {error:.4f}). Please verify sensor data."
                        sms1 = self.send_sms_alert(DOCTOR_PHONE, message)
                        sms2 = self.send_sms_alert(FAMILY_PHONE, message)
                        call = self.make_call_alert(DOCTOR_PHONE, f"Warning. Sensor anomaly detected for Patient {self.patient_id}. Please verify sensor data.")
                        alerts = [sms1, sms2, call]
                    self.last_alert_time = current_time

                explanation = exp
                return result, explanation, error, self.vitals, alerts
            return result, None, error, self.vitals, []
        except Exception as e:
            st.error(f"Anomaly detection failed: {e}")
            return "Error: Anomaly Detection Failed", None, 0.0, self.vitals, [f"Error: {e}"]

# Load models and artifacts
try:
    autoencoder = tf.keras.models.load_model("autoencoder.h5", custom_objects={"mse": tf.keras.losses.MeanSquaredError()})
    classifier = tf.keras.models.load_model("classifier.h5")
    with open("scaler.pkl", "rb") as f:
        scaler = pickle.load(f)
    with open("threshold.pkl", "rb") as f:
        threshold = pickle.load(f)
    test_data = pd.read_csv("test_data.csv")
    with open("event_indices.pkl", "rb") as f:
        event_indices = pickle.load(f)
    with open("anomaly_indices.pkl", "rb") as f:
        anomaly_indices = pickle.load(f)
except Exception as e:
    st.error(f"Failed to load models or artifacts: {e}")
    st.stop()

# Generate synthetic training data
train_data = generate_synthetic_train_data()

# Initialize HealthMonitor
monitor = HealthMonitor(autoencoder, classifier, scaler, threshold, patient_id=101, train_data=train_data)

# Custom CSS
st.markdown("""
<style>
    .sidebar .sidebar-content {
        background-color: #f0f2f6;
        transition: all 0.3s ease;
    }
    .stButton>button {
        background-color: #007bff;
        color: white;
        border-radius: 8px;
        padding: 10px 20px;
        font-weight: bold;
    }
    .stButton>button:hover {
        background-color: #0056b3;
    }
    .metric-card {
        background-color: #ffffff;
        border-radius: 10px;
        padding: 15px;
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
        margin-bottom: 20px;
        text-align: center;
    }
    .metric-card h3 {
        margin: 0;
        color: #333;
    }
    .metric-card p {
        font-size: 24px;
        font-weight: bold;
        color: #007bff;
        margin: 10px 0 0 0;
    }
    .alert-box {
        background-color: #ff4d4d;
        color: white;
        padding: 15px;
        border-radius: 8px;
        margin-bottom: 20px;
        font-weight: bold;
    }
    .alarm-box {
        background-color: #ff0000;
        color: white;
        padding: 20px;
        border-radius: 10px;
        font-size: 20px;
        font-weight: bold;
        text-align: center;
        margin-bottom: 20px;
    }
    .solved-button-container {
        display: flex;
        justify-content: center;
        margin-top: 20px;
        margin-bottom: 20px;
    }
</style>
""", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    if 'sidebar_collapsed' not in st.session_state:
        st.session_state.sidebar_collapsed = False

    if st.button("Toggle Sidebar"):
        st.session_state.sidebar_collapsed = not st.session_state.sidebar_collapsed

    if not st.session_state.sidebar_collapsed:
        st.image(patient_data["photo"], width=150)
        st.write(f"**Patient ID**: {patient_data['patient_id']}")
        st.write(f"**Name**: {patient_data['name']}")
        st.write(f"**Age**: {patient_data['age']}")
        st.write(f"**Disease**: {patient_data['disease']}")

    st.subheader("Twilio Debug")
    if st.button("Test Twilio SMS and Call"):
        test_message = "Test message from Medical Anomaly Detection Dashboard."
        sms_result = monitor.send_sms_alert(DOCTOR_PHONE, test_message)
        call_result = monitor.make_call_alert(DOCTOR_PHONE, "This is a test call from the Medical Anomaly Detection Dashboard.")
        st.write("**Test SMS Result:**")
        st.write(sms_result)
        st.write("**Test Call Result:**")
        st.write(call_result)

# Initialize session state
if 'sample_idx' not in st.session_state:
    st.session_state.sample_idx = 0
if 'last_update_time' not in st.session_state:
    st.session_state.last_update_time = time.time()
if 'anomaly_detected' not in st.session_state:
    st.session_state.anomaly_detected = False
if 'anomaly_time' not in st.session_state:
    st.session_state.anomaly_time = None
if 'last_result' not in st.session_state:
    st.session_state.last_result = "Status Normal"
if 'last_explanation' not in st.session_state:
    st.session_state.last_explanation = None
if 'last_error' not in st.session_state:
    st.session_state.last_error = 0.0
if 'last_alerts' not in st.session_state:
    st.session_state.last_alerts = []
if 'last_sample' not in st.session_state:
    st.session_state.last_sample = None
if 'last_context' not in st.session_state:
    st.session_state.last_context = None
if 'action_taken' not in st.session_state:
    st.session_state.action_taken = False

# Main Dashboard
st.title("Medical Anomaly Detection Dashboard")

# Pause duration for anomaly (2 minutes)
pause_duration = 120  # seconds

# Create placeholders
sample_info_placeholder = st.empty()
vitals_placeholder = st.empty()
trends_placeholder = st.empty()
anomaly_placeholder = st.empty()
alerts_placeholder = st.empty()
lime_placeholder = st.empty()
solved_placeholder = st.empty()

# Main loop
while True:
    current_time = time.time()
    
    if st.session_state.anomaly_detected:
        time_elapsed = current_time - st.session_state.anomaly_time
        if time_elapsed < pause_duration and not st.session_state.action_taken:
            with sample_info_placeholder.container():
                st.subheader("Anomaly Detected - Monitoring Paused")
                st.write(f"Paused for {int(pause_duration - time_elapsed)} seconds remaining. Please take action.")
                st.write(f"Sample Index: {st.session_state.sample_idx}")
                st.write(f"Last Update: {time.strftime('%H:%M:%S', time.localtime(st.session_state.last_update_time))} IST")

            with vitals_placeholder.container():
                st.subheader("Last Known Vital Signs")
                if st.session_state.last_sample is not None:
                    cols = st.columns(3)
                    vitals = ['HR', 'PP', 'BP_S', 'BP_D', 'RESP', 'SpO2']
                    for i, vital in enumerate(vitals):
                        with cols[i % 3]:
                            st.markdown(f"""
                            <div class="metric-card">
                                <h3>{vital}</h3>
                                <p>{st.session_state.last_sample[vital]:.2f}</p>
                            </div>
                            """, unsafe_allow_html=True)

            with trends_placeholder.container():
                st.subheader("Last Known Vital Signs Trends")
                if st.session_state.last_context is not None:
                    fig = make_subplots(rows=3, cols=2, subplot_titles=vitals, vertical_spacing=0.15)
                    for i, vital in enumerate(vitals):
                        row = (i // 2) + 1
                        col = (i % 2) + 1
                        trace = go.Scatter(x=st.session_state.last_context.index, y=st.session_state.last_context[vital], mode='lines+markers', name=vital, line=dict(color='#007bff'))
                        fig.add_trace(trace, row=row, col=col)
                    fig.update_layout(height=800, showlegend=False, title_text="Vital Signs Over Time", template="plotly_white")
                    st.plotly_chart(fig, use_container_width=True)

            with anomaly_placeholder.container():
                st.subheader("Anomaly Detection")
                ground_truth = "Normal"
                if st.session_state.sample_idx in event_indices:
                    ground_truth = "Medical Event"
                elif st.session_state.sample_idx in anomaly_indices:
                    ground_truth = "Sensor Anomaly"
                if "ALERT" in st.session_state.last_result:
                    st.markdown(f'<div class="alert-box">{st.session_state.last_result} (MSE: {st.session_state.last_error:.4f})</div>', unsafe_allow_html=True)
                else:
                    st.write(f"{st.session_state.last_result} (MSE: {st.session_state.last_error:.4f})")
                st.write(f"**Ground Truth**: {ground_truth}")

            with alerts_placeholder.container():
                if st.session_state.last_alerts:
                    st.write("**Alerts Sent:**")
                    for alert in st.session_state.last_alerts:
                        st.write(f"- {alert}")
                else:
                    st.write("No alerts sent (normal status or cooldown active).")

            with lime_placeholder.container():
                if st.session_state.last_explanation is not None:
                    st.subheader("LIME Explanation")
                    try:
                        plt.figure(figsize=(10, 6))
                        st.session_state.last_explanation.as_pyplot_figure()
                        plt.title(f"LIME Explanation (MSE: {st.session_state.last_error:.4f})")
                        plt.tight_layout()
                        buf = io.BytesIO()
                        plt.savefig(buf, format="png")
                        plt.close()
                        buf.seek(0)
                        img_str = base64.b64encode(buf.getvalue()).decode()
                        st.image(f"data:image/png;base64,{img_str}", caption="Raw LIME Explanation")
                    except Exception as e:
                        st.error(f"Failed to generate LIME explanation plot: {e}")
                else:
                    st.write("No LIME explanation available (normal status or error occurred).")

            with solved_placeholder.container():
                st.markdown('<div class="solved-button-container">', unsafe_allow_html=True)
                if st.button("Solved"):
                    st.session_state.anomaly_detected = False
                    st.session_state.anomaly_time = None
                    st.session_state.action_taken = True
                    st.session_state.sample_idx = (st.session_state.sample_idx + 1) % len(test_data)
                    st.session_state.last_update_time = current_time
                    st.session_state.last_result = "Status Normal"
                    st.session_state.last_explanation = None
                    st.session_state.last_error = 0.0
                    st.session_state.last_alerts = []
                    st.session_state.last_sample = None
                    st.session_state.last_context = None
                    st.rerun()
                st.markdown('</div>', unsafe_allow_html=True)

        else:
            if not st.session_state.action_taken and time_elapsed >= pause_duration:
                with solved_placeholder.container():
                    st.markdown("""
                    <div class="alarm-box">
                        ALARM: No action taken within 2 minutes! Please respond immediately.
                    </div>
                    """, unsafe_allow_html=True)
            st.session_state.anomaly_detected = False
            st.session_state.anomaly_time = None
            st.session_state.action_taken = False
            st.session_state.sample_idx = (st.session_state.sample_idx + 1) % len(test_data)
            st.session_state.last_update_time = current_time
            solved_placeholder.empty()
    else:
        if current_time - st.session_state.last_update_time >= 2:
            st.session_state.last_update_time = current_time

            sample = test_data.iloc[st.session_state.sample_idx]
            context = test_data.iloc[max(0, st.session_state.sample_idx - 20):st.session_state.sample_idx + 1]

            with sample_info_placeholder.container():
                st.subheader("Current Data Sample")
                st.write(f"Sample Index: {st.session_state.sample_idx}")
                st.write(f"Last Update: {time.strftime('%H:%M:%S', time.localtime(st.session_state.last_update_time))} IST")

            with vitals_placeholder.container():
                st.subheader("Current Vital Signs")
                cols = st.columns(3)
                vitals = ['HR', 'PP', 'BP_S', 'BP_D', 'RESP', 'SpO2']
                for i, vital in enumerate(vitals):
                    with cols[i % 3]:
                        st.markdown(f"""
                        <div class="metric-card">
                            <h3>{vital}</h3>
                            <p>{sample[vital]:.2f}</p>
                        </div>
                        """, unsafe_allow_html=True)

            with trends_placeholder.container():
                st.subheader("Vital Signs Trends")
                fig = make_subplots(rows=3, cols=2, subplot_titles=vitals, vertical_spacing=0.15)
                for i, vital in enumerate(vitals):
                    row = (i // 2) + 1
                    col = (i % 2) + 1
                    trace = go.Scatter(x=context.index, y=context[vital], mode='lines+markers', name=vital, line=dict(color='#007bff'))
                    fig.add_trace(trace, row=row, col=col)
                fig.update_layout(height=800, showlegend=False, title_text="Vital Signs Over Time", template="plotly_white")
                st.plotly_chart(fig, use_container_width=True)

            result, explanation, error, feature_names, alerts = monitor.analyze_with_explanation(sample, context, st.session_state.sample_idx)

            with anomaly_placeholder.container():
                st.subheader("Anomaly Detection")
                ground_truth = "Normal"
                if st.session_state.sample_idx in event_indices:
                    ground_truth = "Medical Event"
                elif st.session_state.sample_idx in anomaly_indices:
                    ground_truth = "Sensor Anomaly"
                if "ALERT" in result:
                    st.markdown(f'<div class="alert-box">{result} (MSE: {error:.4f})</div>', unsafe_allow_html=True)
                else:
                    st.write(f"{result} (MSE: {error:.4f})")
                st.write(f"**Ground Truth**: {ground_truth}")

            with alerts_placeholder.container():
                if alerts:
                    st.write("**Alerts Sent:**")
                    for alert in alerts:
                        st.write(f"- {alert}")
                else:
                    st.write("No alerts sent (normal status or cooldown active).")

            with lime_placeholder.container():
                if explanation is not None:
                    st.subheader("LIME Explanation")
                    try:
                        plt.figure(figsize=(10, 6))
                        explanation.as_pyplot_figure()
                        plt.title(f"LIME Explanation (MSE: {error:.4f})")
                        plt.tight_layout()
                        buf = io.BytesIO()
                        plt.savefig(buf, format="png")
                        plt.close()
                        buf.seek(0)
                        img_str = base64.b64encode(buf.getvalue()).decode()
                        st.image(f"data:image/png;base64,{img_str}", caption="Raw LIME Explanation")
                    except Exception as e:
                        st.error(f"Failed to generate LIME explanation plot: {e}")
                else:
                    st.write("No LIME explanation available (normal status or error occurred).")

            if "ALERT" in result or "Warning" in result:
                st.session_state.anomaly_detected = True
                st.session_state.anomaly_time = current_time
                st.session_state.last_result = result
                st.session_state.last_explanation = explanation
                st.session_state.last_error = error
                st.session_state.last_alerts = alerts
                st.session_state.last_sample = sample
                st.session_state.last_context = context
                st.session_state.action_taken = False
                st.rerun()
            else:
                st.session_state.sample_idx = (st.session_state.sample_idx + 1) % len(test_data)

        remaining_time = max(0, 2 - (current_time - st.session_state.last_update_time))
        time.sleep(remaining_time)