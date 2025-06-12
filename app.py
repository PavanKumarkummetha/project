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

# Twilio Configuration
TWILIO_ACCOUNT_SID = "AC3ae95e66798dc1f27719b2096f448dd1"  # Your Account SID
TWILIO_AUTH_TOKEN = "a28e7d7b88c667a2a1514a6d85853c6e"    # Your Auth Token
TWILIO_PHONE_NUMBER = "+15075015667"  # Your Twilio phone number
DOCTOR_PHONE = "+918074920408"        # Valid number
FAMILY_PHONE = "+918074920408"        # Valid number

# Initialize Twilio client with error handling
try:
    twilio_client = Client(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)
    # Test Twilio client initialization by fetching account details
    account = twilio_client.api.accounts(TWILIO_ACCOUNT_SID).fetch()
    st.success("Twilio client initialized successfully!")
except Exception as e:
    st.error(f"Failed to initialize Twilio client: {e}")
    twilio_client = None

# Mock patient data
patient_data = {
    "patient_id": 101,
    "name": "Krishna",
    "age": 45,
    "disease": "Hypertension",
    "photo": "data:image/jpeg;base64,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"
}

# Generate synthetic training data for LIME explainer (since train_data.csv is not available)
def generate_synthetic_train_data(num_samples=1000):
    np.random.seed(42)
    data = {
        "patient_id": [101] * num_samples,
        "HR": np.random.normal(70, 10, num_samples),  # Heart rate
        "PP": np.random.normal(40, 5, num_samples),   # Pulse pressure
        "BP_S": np.random.normal(120, 15, num_samples),  # Systolic BP
        "BP_D": np.random.normal(80, 10, num_samples),   # Diastolic BP
        "RESP": np.random.normal(16, 2, num_samples),    # Respiratory rate
        "SpO2": np.random.normal(98, 1, num_samples)     # Oxygen saturation
    }
    return pd.DataFrame(data)

# Health Monitor with Alert Cooldown and LIME Explanations
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
        self.alert_cooldown = 0  # Set to 0 to ensure alerts for every anomaly
        self.explanation_cache = {}  # Cache for LIME explanations

    def send_sms_alert(self, to_number, message):
        # Fallback notification if Twilio fails
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
        # Fallback notification if Twilio fails
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
                # Prepare features for classification
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

                # Compute LIME explanation if explainer is available
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

                # Use LIME values if available, otherwise use zeros
                lime_vals = np.array([x[1] for x in exp.local_exp[0]]) if exp is not None else np.zeros(len(self.vitals))
                features = np.concatenate([
                    feature_errors,
                    *temporal_features,
                    lime_vals,
                    [error]
                ]).reshape(1, -1)

                # Classify the anomaly
                prediction = self.classifier.predict(features, verbose=0)[0][0]
                current_time = time.time()

                # Send alerts if cooldown period has passed
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

# Load models and artifacts with custom objects
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

# Generate synthetic training data for LIME explainer
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

# Sidebar with Twilio Test Button
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

# Check if we're in a paused state due to an anomaly
current_time = time.time()
if st.session_state.anomaly_detected:
    time_elapsed = current_time - st.session_state.anomaly_time
    if time_elapsed < pause_duration and not st.session_state.action_taken:
        # Display paused state
        st.subheader("Anomaly Detected - Monitoring Paused")
        st.write(f"Paused for {int(pause_duration - time_elapsed)} seconds remaining. Please take action.")
        
        # Display last known data
        st.subheader("Last Known Data Sample")
        st.write(f"Sample Index: {st.session_state.sample_idx}")
        st.write(f"Last Update: {time.strftime('%H:%M:%S', time.localtime(st.session_state.last_update_time))} IST")

        # Display last known vitals
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

        # Plot last known vital signs trends
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

        # Display last anomaly detection result
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

        # Display last alerts
        if st.session_state.last_alerts:
            st.write("**Alerts Sent:**")
            for alert in st.session_state.last_alerts:
                st.write(f"- {alert}")
        else:
            st.write("No alerts sent (normal status or cooldown active).")

        # Display last LIME explanation (raw plot only)
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

        # Solved button at the bottom
        st.markdown('<div class="solved-button-container">', unsafe_allow_html=True)
        if st.button("Solved"):
            st.session_state.action_taken = True
            st.session_state.anomaly_detected = False
            st.session_state.anomaly_time = None
            st.session_state.last_update_time = current_time
            st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)

    else:
        # Pause duration has ended or action was taken
        if not st.session_state.action_taken and time_elapsed >= pause_duration:
            st.markdown("""
            <div class="alarm-box">
                ALARM: No action taken within 2 minutes! Please respond immediately.
            </div>
            """, unsafe_allow_html=True)
        # Reset anomaly state and resume monitoring
        st.session_state.anomaly_detected = False
        st.session_state.anomaly_time = None
        st.session_state.action_taken = False
else:
    # Normal monitoring mode
    # Display current sample index and timestamp
    st.subheader("Current Data Sample")
    st.write(f"Sample Index: {st.session_state.sample_idx}")
    st.write(f"Last Update: {time.strftime('%H:%M:%S', time.localtime(st.session_state.last_update_time))} IST")

    # Get current sample and context
    sample = test_data.iloc[st.session_state.sample_idx]
    context = test_data.iloc[max(0, st.session_state.sample_idx - 20):st.session_state.sample_idx + 1]

    # Display current vitals
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

    # Plot vital signs
    st.subheader("Vital Signs Trends")
    fig = make_subplots(rows=3, cols=2, subplot_titles=vitals, vertical_spacing=0.15)
    for i, vital in enumerate(vitals):
        row = (i // 2) + 1
        col = (i % 2) + 1
        trace = go.Scatter(x=context.index, y=context[vital], mode='lines+markers', name=vital, line=dict(color='#007bff'))
        fig.add_trace(trace, row=row, col=col)
    fig.update_layout(height=800, showlegend=False, title_text="Vital Signs Over Time", template="plotly_white")
    st.plotly_chart(fig, use_container_width=True)

    # Anomaly Detection
    st.subheader("Anomaly Detection")
    result, explanation, error, feature_names, alerts = monitor.analyze_with_explanation(sample, context, st.session_state.sample_idx)

    # Display result and ground truth
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

    # Display alerts
    if alerts:
        st.write("**Alerts Sent:**")
        for alert in alerts:
            st.write(f"- {alert}")
    else:
        st.write("No alerts sent (normal status or cooldown active).")

    # Display LIME explanation (raw plot only)
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

    # Check for anomaly and pause if detected
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
    else:
        # Auto-update logic (continuous monitoring)
        if current_time - st.session_state.last_update_time >= 2:  # Update every 2 seconds
            st.session_state.last_update_time = current_time
            st.session_state.sample_idx = (st.session_state.sample_idx + 1) % len(test_data)  # Loop back to 0
            st.rerun()
        else:
            # Sleep for the remaining time to ensure consistent 2-second intervals
            remaining_time = max(0, 2 - (current_time - st.session_state.last_update_time))
            time.sleep(remaining_time)
            st.session_state.last_update_time = time.time()
            st.session_state.sample_idx = (st.session_state.sample_idx + 1) % len(test_data)  # Loop back to 0
            st.rerun()