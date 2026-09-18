"""
app.py
======
AI-Based Micro-Doppler Signature Classification
Border Surveillance Radar â€” Coordinator Dashboard
--------------------------------------------------
Entry point: `streamlit run app.py`

DASHBOARD OVERVIEW
-------------------
This Streamlit app ties together all modules into a live demo:

  1. Sidebar controls  â€” configure and start/stop the simulation
  2. Training panel    â€” train the CNN on first launch (cached after)
  3. Live feed tab     â€” auto-updating table of recent detections
  4. Alerts tab        â€” escalated alerts with Accept/Dismiss buttons
  5. Trend chart tab   â€” detection volume and threat score over time

SIMULATION LOOP
----------------
When simulation is running, each iteration:
  1. Randomly picks a target class (slightly weighted toward drones for drama)
  2. Generates a synthetic micro-Doppler signal via data_generator.py
  3. Converts it to a spectrogram via preprocessor.py
  4. Classifies it with the trained CNN â†’ (class_name, confidence)
  5. Scores the threat via threat_scorer.py â†’ (threat_score, level, direction)
  6. Routes it through triage_agent.py â†’ status + reason
  7. Appends to the shared detection log and rerenders the dashboard

All state is stored in st.session_state (Streamlit's built-in session store)
so it persists across reruns within the same browser session.
"""

import time
import datetime
import io
import os
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px

# Local modules
from data_generator import (generate_single_sample, load_csv,
                             CLASS_NAMES, SIGNAL_LENGTH, PRF)
from preprocessor import signal_to_spectrogram, estimate_peak_doppler_hz, doppler_to_velocity
from classifier import (MicroDopplerCNN, train_model, load_model,
                        predict, is_trained, DEFAULT_WEIGHTS_PATH)
from threat_scorer import score_detection, Detection
from triage_agent import TriageAgent

# â”€â”€ Page configuration â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
st.set_page_config(
    page_title="Micro-Doppler Border Surveillance",
    page_icon="ðŸ›¡ï¸",
    layout="wide",
    initial_sidebar_state="expanded",
)

# â”€â”€ Constants â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
MAX_DETECTIONS_DISPLAYED = 100   # keep last N detections in log
REFRESH_INTERVAL_SEC     = 1.5   # seconds between simulation steps
# Class sampling weights for live simulation (drone & human slightly more common)
CLASS_SAMPLING_WEIGHTS   = [0.20, 0.20, 0.15, 0.25, 0.20]

# Threat level â†’ color for UI
THREAT_COLORS = {
    "CRITICAL": "#FF0000",
    "HIGH":     "#FF6600",
    "MEDIUM":   "#FFB300",
    "LOW":      "#00AA44",
}
STATUS_COLORS = {
    "ESCALATED": "ðŸ”´",
    "WATCH":     "ðŸŸ¡",
    "LOGGED":    "ðŸŸ¢",
    "PENDING":   "âšª",
}

# â”€â”€ Session state initialisation â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

def _init_state():
    """Initialise all session state variables on first load."""
    defaults = {
        "running"        : False,      # simulation on/off
        "detections"     : [],         # list of Detection objects
        "triage_agent"   : None,       # TriageAgent instance
        "model"          : None,       # trained MicroDopplerCNN
        "model_trained"  : False,      # flag: has training happened?
        "training_log"   : [],         # per-epoch metrics from training
        "rng"            : np.random.default_rng(int(time.time())),
        "csv_signals"    : None,       # loaded CSV signals (np.ndarray or None)
        "csv_labels"     : None,       # loaded CSV labels
        "csv_idx"        : 0,          # current index into CSV stream
        "data_source"    : "synthetic",
        "snr_db"         : 15.0,
        "total_steps"    : 0,
        # â”€â”€ Radar Visualizer state â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
        "last_signal"    : None,       # most recent raw time-domain signal (512,)
        "last_spec"      : None,       # most recent 2D spectrogram (64, 64)
        "last_class"     : "",         # class name of last detection
        "last_conf"      : 0.0,        # confidence of last detection
        "last_peak_f"    : 0.0,        # peak Doppler freq of last detection (Hz)
        # Waterfall: deque of FFT magnitude rows (each shape 257,)
        # â€” 40 rows Ã— 257 freq bins â€” scrolls upward like a real radar waterfall
        "fft_waterfall"  : None,
        # PPI scope: list of dicts {azimuth, range, threat_score, class_name, status}
        "ppi_tracks"     : [],
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val

    if st.session_state.triage_agent is None:
        st.session_state.triage_agent = TriageAgent()


_init_state()

# â”€â”€ Helper: train or load model â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

@st.cache_resource(show_spinner=False)
def get_trained_model(snr_db: float = 15.0):
    """
    Load model from disk if weights exist, otherwise train from scratch.
    Cached by st.cache_resource â€” runs only once per Streamlit session.
    """
    from data_generator import generate_dataset
    from preprocessor import batch_to_spectrograms

    if is_trained(DEFAULT_WEIGHTS_PATH):
        model = load_model(DEFAULT_WEIGHTS_PATH)
        return model, []

    # Training from scratch
    X, y = generate_dataset(samples_per_class=600, snr_db=snr_db)
    specs = batch_to_spectrograms(X, fs=PRF)
    model, history = train_model(specs, y, epochs=15)
    return model, history


# â”€â”€ Helper: run one simulation step â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

def run_one_step():
    """
    Execute one detection cycle and append the result to session state.
    Called once per refresh while simulation is running.
    """
    model  = st.session_state.model
    agent  = st.session_state.triage_agent
    rng    = st.session_state.rng
    source = st.session_state.data_source
    snr    = st.session_state.snr_db

    # â”€â”€ 1. Get a signal â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    if source == "csv" and st.session_state.csv_signals is not None:
        signals = st.session_state.csv_signals
        labels  = st.session_state.csv_labels
        idx     = st.session_state.csv_idx % len(signals)
        raw_sig = signals[idx]
        true_label = int(labels[idx])
        st.session_state.csv_idx += 1
    else:
        # Pick a random class (weighted toward drone/human for better demos)
        true_label = int(rng.choice(len(CLASS_NAMES), p=CLASS_SAMPLING_WEIGHTS))
        raw_sig    = generate_single_sample(true_label, snr_db=snr)

    # â”€â”€ 2. Convert to spectrogram â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    spec = signal_to_spectrogram(raw_sig, fs=PRF)   # (1, 64, 64)

    # â”€â”€ 3. Classify â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    class_name, confidence, probs = predict(model, spec)

    # â”€â”€ 4. Estimate velocity from Doppler â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    peak_f   = estimate_peak_doppler_hz(raw_sig, fs=PRF)
    velocity = doppler_to_velocity(peak_f)

    # â”€â”€ 5. Compute cluster size (how many escalated in last 5 s) â”€â”€
    now = time.time()
    recent_escalated = [
        d for d in st.session_state.detections
        if d.status == "ESCALATED" and (now - d.timestamp) < 5.0
    ]
    cluster_size = len(recent_escalated) + 1  # +1 for current detection

    # â”€â”€ 6. Score threat â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    threat_score, threat_level, direction = score_detection(
        class_name, velocity, cluster_size, rng
    )

    # â”€â”€ 7. Build Detection object â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    detection = Detection(
        timestamp    = now,
        class_name   = class_name,
        class_idx    = CLASS_NAMES.index(class_name) if class_name in CLASS_NAMES else 0,
        confidence   = confidence,
        velocity_ms  = velocity,
        direction    = direction,
        threat_score = threat_score,
        threat_level = threat_level,
    )

    # â”€â”€ 8. Triage â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    agent.process(detection)

    # â”€â”€ 9. Append to log (keep last N) â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    st.session_state.detections.append(detection)
    if len(st.session_state.detections) > MAX_DETECTIONS_DISPLAYED:
        st.session_state.detections = st.session_state.detections[-MAX_DETECTIONS_DISPLAYED:]

    st.session_state.total_steps += 1

    # â”€â”€ 10. Update Radar Visualizer state â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    # Save raw signal and spectrogram for the visualizer tab
    st.session_state.last_signal = raw_sig
    st.session_state.last_spec   = spec[0]   # drop channel dim â†’ (64, 64)
    st.session_state.last_class  = class_name
    st.session_state.last_conf   = confidence
    st.session_state.last_peak_f = peak_f

    # Compute full FFT magnitude row and push to waterfall deque
    # rfft gives 257 bins for 512-sample signal (0 Hz â€¦ 500 Hz)
    from collections import deque
    if st.session_state.fft_waterfall is None:
        st.session_state.fft_waterfall = deque(maxlen=40)
    fft_mag = np.abs(np.fft.rfft(raw_sig))          # shape (257,)
    fft_log = 10 * np.log10(fft_mag ** 2 + 1e-9)   # log power
    st.session_state.fft_waterfall.append(fft_log)

    # Simulate a random azimuth bearing for the PPI scope
    # In a real system this comes from the antenna scan angle
    azimuth = float(rng.uniform(0, 360))
    rng_km  = float(rng.uniform(0.5, 8.0))          # simulated range 0.5â€“8 km
    track = {
        "azimuth"     : azimuth,
        "range_km"    : rng_km,
        "threat_score": threat_score,
        "class_name"  : class_name,
        "status"      : detection.status,
        "confidence"  : confidence,
        "threat_level": threat_level,
    }
    st.session_state.ppi_tracks.append(track)
    # Keep only the last 60 tracks on the PPI scope (older ones fade)
    if len(st.session_state.ppi_tracks) > 60:
        st.session_state.ppi_tracks = st.session_state.ppi_tracks[-60:]


# â”€â”€ Helper: build detections DataFrame â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

def detections_to_df(detections: list) -> pd.DataFrame:
    if not detections:
        return pd.DataFrame()
    rows = []
    for d in reversed(detections):  # newest first
        rows.append({
            "Time"        : datetime.datetime.fromtimestamp(d.timestamp).strftime("%H:%M:%S"),
            "Target"      : d.class_name,
            "Confidence"  : f"{d.confidence:.0%}",
            "Velocity"    : f"{d.velocity_ms:.1f} m/s",
            "Threat Score": f"{d.threat_score:.1f}/10",
            "Level"       : d.threat_level,
            "Direction"   : d.direction,
            "Status"      : STATUS_COLORS.get(d.status, "âšª") + " " + d.status,
        })
    return pd.DataFrame(rows)


# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
# SIDEBAR
# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•

with st.sidebar:
    st.markdown("## ðŸ“¡ ðŸ›¡ï¸")
    st.title("Border Radar Control")
    st.markdown("---")

    # â”€â”€ Data source â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    st.subheader("ðŸ“¡ Data Source")
    data_source = st.radio(
        "Signal input:",
        ["Synthetic (simulated)", "Upload CSV"],
        index=0,
    )
    st.session_state.data_source = "synthetic" if "Synthetic" in data_source else "csv"

    if st.session_state.data_source == "csv":
        uploaded = st.file_uploader(
            "Upload signal CSV",
            type=["csv"],
            help="CSV with columns: label, s0, s1, ..., s511"
        )
        if uploaded is not None:
            try:
                signals, labels = load_csv(io.StringIO(uploaded.read().decode("utf-8")))
                st.session_state.csv_signals = signals
                st.session_state.csv_labels  = labels
                st.session_state.csv_idx     = 0
                st.success(f"âœ“ Loaded {len(signals)} signals from CSV")
            except Exception as e:
                st.error(f"Error loading CSV: {e}")

    st.markdown("---")

    # â”€â”€ Noise / SNR â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    st.subheader("âš™ï¸ Signal Parameters")
    snr = st.slider(
        "Signal-to-Noise Ratio (dB)",
        min_value=5, max_value=30, value=15, step=1,
        help="Higher SNR = cleaner signals, easier to classify. Realistic: 10â€“20 dB"
    )
    st.session_state.snr_db = float(snr)

    st.markdown("---")

    # â”€â”€ Model training â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    st.subheader("ðŸ§  CNN Model")

    if not st.session_state.model_trained:
        if is_trained(DEFAULT_WEIGHTS_PATH):
            st.info("ðŸ’¾ Saved weights found. Loading...")
        else:
            st.warning("âš ï¸ No saved weights. Will train on first run (~3 min).")

        if st.button("ðŸ”§ Load / Train Model", width='stretch'):
            with st.spinner("Loading / training model (CPU only)..."):
                model, history = get_trained_model(st.session_state.snr_db)
                st.session_state.model         = model
                st.session_state.model_trained = True
                st.session_state.training_log  = history
            st.success("âœ… Model ready!")
            st.rerun()
    else:
        st.success("âœ… Model loaded")
        if st.session_state.training_log:
            final = st.session_state.training_log[-1]
            st.metric("Val Accuracy", f"{final['val_acc']:.1%}")

    if st.button("ðŸ—‘ï¸ Retrain (delete weights)", width='stretch',
                 disabled=not st.session_state.model_trained):
        if os.path.exists(DEFAULT_WEIGHTS_PATH):
            os.remove(DEFAULT_WEIGHTS_PATH)
        st.session_state.model_trained = False
        st.session_state.model         = None
        st.cache_resource.clear()
        st.rerun()

    st.markdown("---")

    # â”€â”€ Simulation control â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    st.subheader("â–¶ï¸ Simulation")
    if not st.session_state.model_trained:
        st.warning("Load the model first.")
    else:
        col1, col2 = st.columns(2)
        with col1:
            if st.button("â–¶ Start", width='stretch',
                         disabled=st.session_state.running):
                st.session_state.running = True
                st.rerun()
        with col2:
            if st.button("â¹ Stop", width='stretch',
                         disabled=not st.session_state.running):
                st.session_state.running = False
                st.rerun()

        if st.button("ðŸ”„ Reset", width='stretch'):
            st.session_state.running      = False
            st.session_state.detections   = []
            st.session_state.triage_agent = TriageAgent()
            st.session_state.total_steps  = 0
            st.session_state.csv_idx      = 0
            st.rerun()

    st.markdown("---")
    st.caption(
        "College prototype â€” not for operational use.\n"
        "Radar physics references: Chen (2011), *The Micro-Doppler Effect in Radar*"
    )


# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
# MAIN DASHBOARD
# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•

st.title("ðŸ›¡ï¸ Micro-Doppler Border Surveillance Radar")
st.caption(
    "AI-based classification of micro-Doppler signatures for multi-target "
    "detection and threat prioritisation in low-altitude border surveillance."
)

# â”€â”€ Status bar â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
detections = st.session_state.detections
agent = st.session_state.triage_agent

status_text = "ðŸŸ¢ RUNNING" if st.session_state.running else "â¹ STOPPED"
col_s1, col_s2, col_s3, col_s4, col_s5 = st.columns(5)
col_s1.metric("Status", status_text)
col_s2.metric("Total Detections", st.session_state.total_steps)
col_s3.metric("Escalated", sum(1 for d in detections if d.status == "ESCALATED"))
col_s4.metric("Pending Alerts", len(agent.get_pending_alerts()))
col_s5.metric(
    "Avg Threat Score",
    f"{np.mean([d.threat_score for d in detections]):.1f}/10" if detections else "â€”"
)

st.markdown("---")

# â”€â”€ Tabs â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
tab_feed, tab_alerts, tab_chart, tab_radar, tab_about = st.tabs(
    ["ðŸ“‹ Live Detection Feed", "ðŸš¨ Escalated Alerts", "ðŸ“ˆ Trend Chart",
     "ðŸ“¡ Radar Visualizer", "â„¹ï¸ About"]
)

# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
# TAB 1: LIVE DETECTION FEED  â€” powered by @st.fragment
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# @st.fragment(run_every=1.5) makes this function re-execute every 1.5 s
# INDEPENDENTLY of the rest of the page.  Only the DOM nodes owned by this
# fragment are patched in the browser â€” no full-page reload, no flicker.
#
# HOW IT WORKS (explain in viva):
#   â€¢ Streamlit 1.33+ supports "fragment mode": a decorated function renders
#     into a dedicated DOM subtree.  run_every=N tells the client to call
#     the fragment again every N seconds via a background websocket message.
#   â€¢ The fragment can read AND write st.session_state; writes are visible
#     to both the fragment itself on the next tick and to the rest of the app
#     on the next full rerun (e.g., when the user clicks a button).
#   â€¢ run_one_step() is called inside the fragment, so the simulation advances
#     on every tick even while the user is just watching â€” no button needed.
# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•

@st.fragment(run_every=1.5)
def live_feed_panel():
    """
    Self-contained live feed fragment.
    Advances the simulation by one step, then re-renders:
      â€¢ Status pill row
      â€¢ Detection table (color-coded)
      â€¢ STFT spectrogram heatmap  (Inferno, colour changes per target class)
      â€¢ Doppler power spectrum     (filled area, colour = class)
      â€¢ Radar PPI scope            (polar, green-on-black, blips fade with age)
    """
    # â”€â”€ Step the simulation â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    if st.session_state.running:
        if st.session_state.model_trained:
            run_one_step()
        else:
            st.warning("âš ï¸ Load the model first (sidebar â†’ Load / Train Model).")
            st.session_state.running = False
            return

    # Snapshot of current state for rendering
    detections  = st.session_state.detections
    agent       = st.session_state.triage_agent
    has_viz     = st.session_state.last_signal is not None

    # â”€â”€ Live status pill row (inside the fragment = updates every tick) â”€â”€â”€
    running_badge = "ðŸŸ¢ LIVE" if st.session_state.running else "â¹ PAUSED"
    p1, p2, p3, p4, p5 = st.columns(5)
    p1.metric("Simulation",    running_badge)
    p2.metric("Total Scans",   st.session_state.total_steps)
    p3.metric("Escalated",     sum(1 for d in detections if d.status == "ESCALATED"))
    p4.metric("Pending Alerts",len(agent.get_pending_alerts()))
    p5.metric("Avg Threat",
              f"{np.mean([d.threat_score for d in detections]):.1f}/10"
              if detections else "â€”")

    if not detections:
        st.info("â–¶ Press **Start** in the sidebar to begin the simulation.")
        return

    # â”€â”€ Two-column split: table left, charts right â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    col_table, col_viz = st.columns([55, 45])

    # â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
    # LEFT â€” Detection table
    # â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
    with col_table:
        st.markdown("**Detections (newest first):**")

        df = detections_to_df(detections)

        def _hl(row):
            s = row.get("Status", "")
            if "ESCALATED" in s:
                return ["background-color:#3d0000;color:white"] * len(row)
            if "WATCH" in s:
                return ["background-color:#3d2d00;color:white"] * len(row)
            return [""] * len(row)

        st.dataframe(df.style.apply(_hl, axis=1), width="stretch", height=370)

        # Class breakdown micro-metrics
        st.markdown("**Breakdown by class:**")
        cc = {}
        for d in detections:
            cc[d.class_name] = cc.get(d.class_name, 0) + 1
        mc = st.columns(len(cc))
        for i, (cls, cnt) in enumerate(cc.items()):
            mc[i].metric(cls.replace("_", " ").title(), cnt,
                         f"{cnt/len(detections)*100:.0f}%")

    # â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
    # RIGHT â€” Live mini radar stack
    # â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
    with col_viz:

        # â”€â”€ Chart 1: STFT Spectrogram heatmap â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
        st.markdown("##### ðŸ”¬ Micro-Doppler Spectrogram")
        if has_viz:
            spec_2d  = st.session_state.last_spec
            cls_name = st.session_state.last_class
            conf_val = st.session_state.last_conf
            t_ax = np.linspace(0, SIGNAL_LENGTH / PRF * 1000, 64)
            f_ax = np.linspace(0, PRF / 2, 64)

            fig_s = go.Figure(go.Heatmap(
                z=spec_2d, x=t_ax, y=f_ax,
                colorscale="Inferno", showscale=False, zsmooth="best",
            ))
            fig_s.add_annotation(
                x=0.02, y=0.97, xref="paper", yref="paper",
                text=f"<b>{cls_name.replace('_',' ').title()}</b> {conf_val:.0%}",
                showarrow=False, font=dict(size=12, color="white"),
                bgcolor="rgba(0,0,0,0.6)", bordercolor="cyan", borderwidth=1,
            )
            fig_s.update_layout(
                template="plotly_dark", height=185,
                margin=dict(l=40, r=6, t=6, b=34),
                xaxis=dict(title="Time (ms)", tickfont=dict(size=9)),
                yaxis=dict(title="Freq (Hz)", tickfont=dict(size=9)),
                plot_bgcolor="#060614", paper_bgcolor="#060614",
            )
            st.plotly_chart(fig_s, width="stretch")
        else:
            st.caption("_Spectrogram appears once simulation starts_")

        # â”€â”€ Chart 2: Doppler power spectrum â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
        st.markdown("##### ðŸ“Š Doppler Power Spectrum")
        if has_viz:
            raw_sig  = st.session_state.last_signal
            peak_f   = st.session_state.last_peak_f
            fft_mag  = np.abs(np.fft.rfft(raw_sig)) ** 2
            fft_db   = 10 * np.log10(fft_mag + 1e-9)
            fft_freq = np.fft.rfftfreq(len(raw_sig), d=1.0 / PRF)
            fft_sm   = np.convolve(fft_db, np.ones(5) / 5, mode="same")

            CCOL = {"human_walk":"#44aaff","human_run":"#00ffcc",
                    "animal":"#88ff44","drone":"#ff4444","vehicle":"#ffaa00"}
            lc = CCOL.get(st.session_state.last_class, "#00c8ff")
            # Build a semi-transparent fill colour from the hex line colour
            r = int(lc[1:3], 16); g = int(lc[3:5], 16); b = int(lc[5:7], 16)
            fc = f"rgba({r},{g},{b},0.12)"

            fig_f = go.Figure()
            fig_f.add_trace(go.Scatter(
                x=fft_freq, y=fft_sm, mode="lines",
                fill="tozeroy", fillcolor=fc,
                line=dict(color=lc, width=2),
            ))
            fig_f.add_vline(x=peak_f, line_color="#ffffff",
                            line_width=1.5, line_dash="dot",
                            annotation_text=f"{peak_f:.0f} Hz",
                            annotation_font_color="#ffffff",
                            annotation_font_size=10)
            fig_f.update_layout(
                template="plotly_dark", height=160,
                margin=dict(l=40, r=6, t=6, b=34),
                xaxis=dict(title="Freq (Hz)", range=[0, 500],
                           tickfont=dict(size=9)),
                yaxis=dict(title="dB", tickfont=dict(size=9)),
                plot_bgcolor="#060614", paper_bgcolor="#060614",
                showlegend=False,
            )
            st.plotly_chart(fig_f, width="stretch")
        else:
            st.caption("_Spectrum appears once simulation starts_")

        # â”€â”€ Chart 3: Mini PPI scope â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
        st.markdown("##### ðŸŽ¯ PPI Radar Scope")
        ppi = st.session_state.ppi_tracks
        if ppi:
            n_t    = len(ppi)
            alphas = np.linspace(0.1, 1.0, n_t)
            LRGB   = {"CRITICAL":(255,0,0),"HIGH":(255,100,0),
                      "MEDIUM":(255,180,0),"LOW":(0,180,80)}
            SSYM   = {"ESCALATED":"star","WATCH":"circle","LOGGED":"circle-open"}

            fig_p = go.Figure()
            # Decorative range rings
            for rr in [2, 4, 6, 8]:
                fig_p.add_trace(go.Scatterpolar(
                    r=[rr]*361, theta=list(np.linspace(0, 360, 361)),
                    mode="lines",
                    line=dict(color="rgba(0,255,100,0.1)", width=1),
                    showlegend=False, hoverinfo="skip",
                ))
            # Target blips
            for i, tk in enumerate(ppi):
                rc, gc, bc = LRGB.get(tk["threat_level"], (128,128,128))
                col_rgba   = f"rgba({rc},{gc},{bc},{alphas[i]:.2f})"
                fig_p.add_trace(go.Scatterpolar(
                    r=[tk["range_km"]], theta=[tk["azimuth"]],
                    mode="markers",
                    marker=dict(
                        color=col_rgba,
                        size=6 + tk["threat_score"] * 1.2,
                        symbol=SSYM.get(tk["status"], "circle"),
                        line=dict(color=col_rgba, width=1),
                    ),
                    hovertemplate=(
                        f"<b>{tk['class_name'].replace('_',' ').title()}</b><br>"
                        f"Range: {tk['range_km']:.1f} km | {tk['azimuth']:.0f}Â°<br>"
                        f"Threat: {tk['threat_score']:.1f}/10<br>"
                        f"Status: {tk['status']}<extra></extra>"
                    ),
                    showlegend=False,
                ))
            # Radar site
            fig_p.add_trace(go.Scatterpolar(
                r=[0], theta=[0], mode="markers",
                marker=dict(color="white", size=8, symbol="square"),
                showlegend=False,
                hovertemplate="<b>Radar Site</b><extra></extra>",
            ))
            fig_p.update_layout(
                polar=dict(
                    bgcolor="#020d14",
                    angularaxis=dict(
                        direction="clockwise", rotation=90,
                        tickfont=dict(color="#00ff88", size=8),
                        gridcolor="rgba(0,255,100,0.12)",
                        linecolor="rgba(0,255,100,0.2)",
                    ),
                    radialaxis=dict(
                        range=[0, 9],
                        tickvals=[2,4,6,8],
                        ticktext=["2km","4km","6km","8km"],
                        tickfont=dict(color="#00ff88", size=8),
                        gridcolor="rgba(0,255,100,0.1)",
                    ),
                ),
                template="plotly_dark", height=250,
                margin=dict(l=8, r=8, t=8, b=8),
                paper_bgcolor="#020d14",
            )
            st.plotly_chart(fig_p, width="stretch")
        else:
            st.caption("_PPI scope populates as detections accumulate_")


# â”€â”€ Call the fragment inside tab_feed â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
with tab_feed:
    live_feed_panel()

# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
# TAB 2: ESCALATED ALERTS PANEL
# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
with tab_alerts:
    st.subheader("ðŸš¨ Coordinator Alerts â€” Action Required")

    all_alerts = agent.get_alerts()
    pending    = [a for a in all_alerts if a.status == "PENDING"]
    closed     = [a for a in all_alerts if a.status != "PENDING"]

    if not all_alerts:
        st.info("No alerts yet. Escalated detections will appear here.")
    else:
        # â”€â”€ Pending alerts â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
        if pending:
            st.markdown(f"**{len(pending)} pending alert(s) â€” awaiting action:**")
            for alert in reversed(pending):
                ts_str = datetime.datetime.fromtimestamp(alert.timestamp).strftime(
                    "%H:%M:%S"
                )
                is_swarm = alert.alert_type == "SWARM"
                border_color = "#FF0000" if is_swarm else "#FF6600"

                with st.container(border=True):
                    cols_a = st.columns([1, 6, 1, 1])

                    with cols_a[0]:
                        st.markdown(
                            f"{'ðŸŒŠ' if is_swarm else 'âš ï¸'} **{alert.alert_type}**"
                        )
                        st.caption(f"ID: {alert.alert_id}")

                    with cols_a[1]:
                        st.markdown(f"**{ts_str}** â€” {alert.summary}")
                        if len(alert.detections) > 1:
                            st.caption(
                                " | ".join(
                                    f"{d.class_name} ({d.confidence:.0%} conf, "
                                    f"score={d.threat_score:.1f})"
                                    for d in alert.detections
                                )
                            )
                        else:
                            d = alert.detections[0]
                            st.caption(
                                f"Confidence: {d.confidence:.0%} | "
                                f"Threat: {d.threat_score:.1f}/10 | "
                                f"Velocity: {d.velocity_ms:.1f} m/s | "
                                f"Direction: {d.direction}"
                            )

                    with cols_a[2]:
                        if st.button("âœ… Accept", key=f"accept_{alert.alert_id}",
                                     width='stretch'):
                            agent.accept_alert(alert.alert_id)
                            st.rerun()

                    with cols_a[3]:
                        if st.button("âŒ Dismiss", key=f"dismiss_{alert.alert_id}",
                                     width='stretch'):
                            agent.dismiss_alert(alert.alert_id)
                            st.rerun()
        else:
            st.success("âœ… All alerts have been actioned.")

        # â”€â”€ Closed alerts history â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
        if closed:
            with st.expander(f"ðŸ“ Actioned alerts ({len(closed)} total)"):
                for alert in reversed(closed):
                    ts_str = datetime.datetime.fromtimestamp(alert.timestamp).strftime(
                        "%H:%M:%S"
                    )
                    icon   = "âœ…" if alert.status == "ACCEPTED" else "âŒ"
                    st.markdown(
                        f"{icon} **{alert.status}** â€” `{alert.alert_id}` @ {ts_str} â€” "
                        f"{alert.summary[:80]}"
                    )


# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
# TAB 3: TREND CHART
# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
with tab_chart:
    st.subheader("ðŸ“ˆ Detection Trend Over Session")

    if len(detections) < 2:
        st.info("Start the simulation to see trend charts.")
    else:
        # Build a time-series dataframe
        df_trend = pd.DataFrame([
            {
                "time"       : datetime.datetime.fromtimestamp(d.timestamp),
                "threat_score": d.threat_score,
                "status"     : d.status,
                "class"      : d.class_name,
                "confidence" : d.confidence,
            }
            for d in detections
        ])

        # â”€â”€ Threat score over time â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
        fig_threat = go.Figure()
        fig_threat.add_trace(go.Scatter(
            x=df_trend["time"],
            y=df_trend["threat_score"],
            mode="lines+markers",
            name="Threat Score",
            line=dict(color="#FF6600", width=2),
            marker=dict(
                size=6,
                color=df_trend["threat_score"],
                colorscale="Reds",
                showscale=False,
            ),
        ))
        # Rolling average (window=5)
        df_trend["threat_ma"] = df_trend["threat_score"].rolling(5, min_periods=1).mean()
        fig_threat.add_trace(go.Scatter(
            x=df_trend["time"],
            y=df_trend["threat_ma"],
            mode="lines",
            name="Moving Avg (5)",
            line=dict(color="#FFFF00", width=2, dash="dash"),
        ))
        fig_threat.add_hline(y=6.0, line_dash="dot", line_color="orange",
                             annotation_text="Escalation threshold (6.0)")
        fig_threat.add_hline(y=8.0, line_dash="dot", line_color="red",
                             annotation_text="Critical threshold (8.0)")
        fig_threat.update_layout(
            title="Threat Score Timeline",
            xaxis_title="Time",
            yaxis_title="Threat Score (0â€“10)",
            yaxis=dict(range=[0, 10]),
            template="plotly_dark",
            height=300,
            margin=dict(l=40, r=20, t=50, b=40),
        )
        st.plotly_chart(fig_threat, width='stretch')

        # â”€â”€ Detection class distribution pie â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
        col_pie, col_conf = st.columns(2)

        with col_pie:
            class_counts = df_trend["class"].value_counts()
            fig_pie = px.pie(
                names=class_counts.index,
                values=class_counts.values,
                title="Target Class Distribution",
                template="plotly_dark",
                color_discrete_sequence=px.colors.qualitative.Bold,
            )
            fig_pie.update_layout(height=300, margin=dict(l=20, r=20, t=50, b=20))
            st.plotly_chart(fig_pie, width='stretch')

        with col_conf:
            # Confidence histogram
            fig_conf = px.histogram(
                df_trend,
                x="confidence",
                nbins=20,
                color="status",
                title="Confidence Distribution by Status",
                template="plotly_dark",
                color_discrete_map={
                    "ESCALATED": "#FF0000",
                    "WATCH"    : "#FFB300",
                    "LOGGED"   : "#00AA44",
                    "PENDING"  : "#888888",
                },
                barmode="overlay",
            )
            fig_conf.update_layout(
                height=300,
                margin=dict(l=20, r=20, t=50, b=20),
                xaxis_title="Classifier Confidence",
                yaxis_title="Count",
            )
            st.plotly_chart(fig_conf, width='stretch')

        # â”€â”€ Status counts over time â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
        st.markdown("**Status count over session:**")
        status_counts = df_trend["status"].value_counts()
        cols_sc = st.columns(4)
        for i, status in enumerate(["ESCALATED", "WATCH", "LOGGED"]):
            cnt = status_counts.get(status, 0)
            pct = cnt / len(df_trend) * 100 if len(df_trend) > 0 else 0
            cols_sc[i].metric(
                f"{STATUS_COLORS.get(status, '')} {status}",
                cnt,
                f"{pct:.0f}% of all",
            )



# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
# TAB 4: RADAR VISUALIZER
# Live signal charts: waterfall, spectrogram, Doppler FFT, PPI
# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
with tab_radar:
    st.subheader("ðŸ“¡ Live Radar Signal Visualizer")

    if st.session_state.last_signal is None:
        st.info("â³ Start the simulation to see live radar signal charts.")
    else:
        sig      = st.session_state.last_signal        # (512,)
        spec_2d  = st.session_state.last_spec          # (64, 64)
        cls_name = st.session_state.last_class
        conf     = st.session_state.last_conf
        peak_f   = st.session_state.last_peak_f
        peak_v   = doppler_to_velocity(peak_f)

        # â”€â”€ Detection badge â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
        threat_col = {"CRITICAL":"#FF0000","HIGH":"#FF6600",
                      "MEDIUM":"#FFB300","LOW":"#00AA44"}
        last_det = detections[-1] if detections else None
        lvl_color = threat_col.get(last_det.threat_level if last_det else "LOW", "#888")

        badge_cols = st.columns(5)
        badge_cols[0].markdown(f"**ðŸŽ¯ Target**")
        badge_cols[0].markdown(f"### {cls_name.replace('_',' ').title()}")
        badge_cols[1].markdown("**ðŸŽ² Confidence**")
        badge_cols[1].markdown(f"### {conf:.0%}")
        badge_cols[2].markdown("**âš¡ Peak Doppler**")
        badge_cols[2].markdown(f"### {peak_f:.1f} Hz")
        badge_cols[3].markdown("**ðŸš€ Est. Velocity**")
        badge_cols[3].markdown(f"### {peak_v:.2f} m/s")
        badge_cols[4].markdown("**ðŸ”¥ Threat**")
        if last_det:
            badge_cols[4].markdown(
                f"<span style='color:{lvl_color};font-size:1.4em;font-weight:bold'>"
                f"{last_det.threat_level} ({last_det.threat_score:.1f}/10)</span>",
                unsafe_allow_html=True,
            )
        else:
            badge_cols[4].markdown("### â€”")

        st.markdown("---")

        # â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
        # ROW 1: Waterfall (left) + Current Spectrogram (right)
        # â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
        col_wf, col_spec = st.columns(2)

        # â”€â”€ Waterfall display â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
        with col_wf:
            st.markdown("#### ðŸŒŠ Doppler Waterfall")
            st.caption("Each row = one scan. Newest at top. Frequency â†’ horizontal axis.")

            wf_deque = st.session_state.fft_waterfall
            if wf_deque and len(wf_deque) >= 2:
                # Build 2-D array: rows = time (newest first), cols = freq bins
                waterfall_arr = np.array(list(reversed(wf_deque)))  # (N, 257)
                n_rows, n_cols = waterfall_arr.shape

                # Frequency axis: 0 â†’ 500 Hz (rfft of 512 @ 1 kHz)
                freqs = np.linspace(0, PRF / 2, n_cols)

                fig_wf = go.Figure(go.Heatmap(
                    z=waterfall_arr,
                    x=freqs,
                    y=list(range(n_rows)),
                    colorscale="Plasma",
                    showscale=True,
                    colorbar=dict(title="dB", thickness=12, len=0.8),
                    zsmooth="best",
                ))
                # Mark the peak Doppler frequency as a vertical line
                fig_wf.add_vline(
                    x=peak_f, line_color="cyan", line_width=2, line_dash="dash",
                    annotation_text=f"Peak: {peak_f:.0f} Hz",
                    annotation_font_color="cyan",
                )
                fig_wf.update_layout(
                    template="plotly_dark",
                    height=320,
                    margin=dict(l=10, r=10, t=30, b=40),
                    xaxis=dict(title="Doppler Frequency (Hz)", range=[0, 500]),
                    yaxis=dict(
                        title="Scan (newest â†‘)",
                        showticklabels=False,
                    ),
                    plot_bgcolor="#0a0a1a",
                    paper_bgcolor="#0a0a1a",
                )
                st.plotly_chart(fig_wf, width='stretch')
            else:
                st.info("Collecting waterfall data â€” keep simulation runningâ€¦")

        # â”€â”€ STFT Spectrogram heatmap â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
        with col_spec:
            st.markdown("#### ðŸ”¬ STFT Micro-Doppler Spectrogram")
            st.caption("Time â†’ horizontal. Frequency â†’ vertical. Brighter = stronger return.")

            # spec_2d is (64, 64), already log-power normalised to [0,1]
            # y-axis: 0â€“500 Hz (freq bins), x-axis: 0â€“512 ms (time)
            time_axis = np.linspace(0, SIGNAL_LENGTH / PRF * 1000, 64)   # ms
            freq_axis = np.linspace(0, PRF / 2, 64)                       # Hz

            fig_spec = go.Figure(go.Heatmap(
                z=spec_2d,
                x=time_axis,
                y=freq_axis,
                colorscale="Inferno",
                showscale=True,
                colorbar=dict(title="Norm.", thickness=12, len=0.8),
                zsmooth="best",
            ))
            # Annotate with class name and confidence
            fig_spec.add_annotation(
                x=0.02, y=0.97, xref="paper", yref="paper",
                text=f"<b>{cls_name.replace('_',' ').title()}</b> ({conf:.0%})",
                showarrow=False,
                font=dict(size=13, color="white"),
                bgcolor="rgba(0,0,0,0.55)",
                bordercolor="cyan",
                borderwidth=1,
            )
            fig_spec.update_layout(
                template="plotly_dark",
                height=320,
                margin=dict(l=10, r=10, t=30, b=40),
                xaxis=dict(title="Time (ms)"),
                yaxis=dict(title="Doppler Frequency (Hz)"),
                plot_bgcolor="#0a0a1a",
                paper_bgcolor="#0a0a1a",
            )
            st.plotly_chart(fig_spec, width='stretch')

        # â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
        # ROW 2: Doppler Power Spectrum (left) + PPI Scope (right)
        # â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
        col_fft, col_ppi = st.columns(2)

        # â”€â”€ Doppler power spectrum â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
        with col_fft:
            st.markdown("#### ðŸ“Š Doppler Power Spectrum")
            st.caption("FFT of current signal. Peak bin gives target radial velocity.")

            fft_mag  = np.abs(np.fft.rfft(sig)) ** 2        # power spectrum
            fft_db   = 10 * np.log10(fft_mag + 1e-9)        # dB
            fft_freq = np.fft.rfftfreq(len(sig), d=1.0 / PRF)

            # Smooth with a small moving average for visual clarity
            kernel   = np.ones(5) / 5
            fft_smooth = np.convolve(fft_db, kernel, mode="same")

            peak_bin = int(np.argmax(fft_mag[fft_freq > 5]))  # skip DC

            fig_fft = go.Figure()
            # Filled area under spectrum
            fig_fft.add_trace(go.Scatter(
                x=fft_freq, y=fft_smooth,
                mode="lines",
                fill="tozeroy",
                fillcolor="rgba(0, 200, 255, 0.15)",
                line=dict(color="#00c8ff", width=2),
                name="Power (dB)",
            ))
            # Highlight the peak
            peak_freq_val = fft_freq[fft_freq > 5][peak_bin]
            fig_fft.add_vline(
                x=peak_freq_val, line_color="#ff4444",
                line_width=2, line_dash="dot",
                annotation_text=f"Peak {peak_freq_val:.0f} Hz â†’ {doppler_to_velocity(peak_freq_val):.1f} m/s",
                annotation_font_color="#ff8888",
            )
            fig_fft.update_layout(
                template="plotly_dark",
                height=320,
                margin=dict(l=10, r=10, t=30, b=40),
                xaxis=dict(title="Doppler Frequency (Hz)", range=[0, 500]),
                yaxis=dict(title="Power (dB)"),
                plot_bgcolor="#0a0a1a",
                paper_bgcolor="#0a0a1a",
                showlegend=False,
            )
            st.plotly_chart(fig_fft, width='stretch')

        # â”€â”€ Radar PPI Scope (Plan Position Indicator) â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
        with col_ppi:
            st.markdown("#### ðŸŽ¯ Radar PPI Scope")
            st.caption(
                "Polar plot: angle = azimuth bearing, radius = range (km). "
                "Colour = threat level. Fades with age."
            )

            ppi_tracks = st.session_state.ppi_tracks
            if ppi_tracks:
                n_tracks = len(ppi_tracks)
                # Alpha fades older tracks: newest = 1.0, oldest = 0.15
                alphas = np.linspace(0.15, 1.0, n_tracks)

                # Build per-track colour with alpha
                LEVEL_RGB = {
                    "CRITICAL": (255, 0,   0),
                    "HIGH":     (255, 100, 0),
                    "MEDIUM":   (255, 180, 0),
                    "LOW":      (0,   180, 80),
                }
                STATUS_SYM = {
                    "ESCALATED": "star",
                    "WATCH":     "circle",
                    "LOGGED":    "circle-open",
                }

                fig_ppi = go.Figure()

                # Range rings (decorative, like a real radar scope)
                for r in [2, 4, 6, 8]:
                    theta_ring = np.linspace(0, 360, 361)
                    fig_ppi.add_trace(go.Scatterpolar(
                        r=[r] * 361, theta=theta_ring,
                        mode="lines",
                        line=dict(color="rgba(0,255,100,0.12)", width=1),
                        showlegend=False,
                        hoverinfo="skip",
                    ))

                # Plot each track
                for i, track in enumerate(ppi_tracks):
                    r, g, b = LEVEL_RGB.get(track["threat_level"], (128, 128, 128))
                    a = alphas[i]
                    color = f"rgba({r},{g},{b},{a:.2f})"
                    symbol = STATUS_SYM.get(track["status"], "circle")
                    size   = 8 + track["threat_score"] * 1.5   # bigger = higher threat

                    fig_ppi.add_trace(go.Scatterpolar(
                        r=[track["range_km"]],
                        theta=[track["azimuth"]],
                        mode="markers",
                        marker=dict(
                            color=color,
                            size=size,
                            symbol=symbol,
                            line=dict(color=color, width=1),
                        ),
                        name=track["class_name"],
                        hovertemplate=(
                            f"<b>{track['class_name'].replace('_',' ').title()}</b><br>"
                            f"Range: {track['range_km']:.1f} km<br>"
                            f"Bearing: {track['azimuth']:.0f}Â°<br>"
                            f"Threat: {track['threat_score']:.1f}/10 ({track['threat_level']})<br>"
                            f"Status: {track['status']}<extra></extra>"
                        ),
                        showlegend=False,
                    ))

                # "Origin" â€” the radar installation
                fig_ppi.add_trace(go.Scatterpolar(
                    r=[0], theta=[0], mode="markers",
                    marker=dict(color="white", size=12, symbol="square"),
                    name="Radar Site",
                    hovertemplate="<b>Radar Installation</b><extra></extra>",
                    showlegend=False,
                ))

                fig_ppi.update_layout(
                    polar=dict(
                        bgcolor="#020d14",
                        angularaxis=dict(
                            tickfont=dict(color="#00ff88", size=10),
                            direction="clockwise",
                            rotation=90,       # 0Â° = North
                            gridcolor="rgba(0,255,100,0.15)",
                            linecolor="rgba(0,255,100,0.2)",
                        ),
                        radialaxis=dict(
                            range=[0, 9],
                            tickvals=[2, 4, 6, 8],
                            ticktext=["2km","4km","6km","8km"],
                            tickfont=dict(color="#00ff88", size=9),
                            gridcolor="rgba(0,255,100,0.12)",
                            linecolor="rgba(0,255,100,0.15)",
                        ),
                    ),
                    template="plotly_dark",
                    height=320,
                    margin=dict(l=10, r=10, t=30, b=10),
                    paper_bgcolor="#020d14",
                    plot_bgcolor="#020d14",
                )
                st.plotly_chart(fig_ppi, width='stretch')
            else:
                st.info("PPI scope populates as detections accumulateâ€¦")

        # â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
        # ROW 3: Raw time-domain waveform (full width)
        # â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
        st.markdown("#### ã€°ï¸ Raw Time-Domain Signal")
        st.caption(
            "The raw simulated radar return before any processing. "
            "Micro-Doppler modulations are visible as amplitude variations on the carrier."
        )
        t_axis = np.linspace(0, SIGNAL_LENGTH / PRF * 1000, SIGNAL_LENGTH)   # ms

        fig_sig = go.Figure()
        fig_sig.add_trace(go.Scatter(
            x=t_axis,
            y=sig,
            mode="lines",
            line=dict(color="#00ffcc", width=1),
            fill="tozeroy",
            fillcolor="rgba(0,255,200,0.05)",
            name="Amplitude",
        ))
        fig_sig.update_layout(
            template="plotly_dark",
            height=200,
            margin=dict(l=10, r=10, t=20, b=40),
            xaxis=dict(title="Time (ms)"),
            yaxis=dict(title="Amplitude"),
            plot_bgcolor="#0a0a1a",
            paper_bgcolor="#0a0a1a",
            showlegend=False,
        )
        st.plotly_chart(fig_sig, width='stretch')


# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
# TAB 5: ABOUT / REFERENCE
# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
with tab_about:

    st.subheader("â„¹ï¸ System Architecture & Physics Reference")

    st.markdown("""
    ### Micro-Doppler Physics

    **Micro-Doppler** refers to additional frequency modulations on top of the
    bulk Doppler shift, caused by the micro-motions of target components
    (limb swings, rotor blades, vibrations).

    The Doppler frequency equation:
    > f_d = 2 Ã— v_radial / Î»

    where **v_radial** is the target's radial velocity (m/s) and **Î»** is the
    radar wavelength (0.03 m for 10 GHz X-band radar).

    | Target | Signature |
    |---|---|
    | Human walk | Sinusoidal band at ~80 Hz Â± gait sidebands at Â±1.8 Hz |
    | Human run | Wider band, faster cadence (~3 Hz sidebands) |
    | Animal (dog) | Quadruped 4-limb pattern, dual-cadence sidebands |
    | Drone | Near-DC body + high-freq rotor flash (blade Ã— RPM / 60) |
    | Vehicle | Large steady Doppler, engine micro-tremor only |

    ---

    ### CNN Architecture

    ```
    Input (1 Ã— 64 Ã— 64 spectrogram)
      â””â”€ Conv2d(1â†’16, 3Ã—3) + ReLU + MaxPool â†’ (16 Ã— 32 Ã— 32)
      â””â”€ Conv2d(16â†’32, 3Ã—3) + ReLU + MaxPool â†’ (32 Ã— 16 Ã— 16)
      â””â”€ Conv2d(32â†’64, 3Ã—3) + ReLU + MaxPool â†’ (64 Ã— 8 Ã— 8)
      â””â”€ Flatten â†’ Linear(4096â†’128) + ReLU + Dropout(0.3)
      â””â”€ Linear(128â†’5) â†’ Softmax â†’ class probabilities
    ```

    ---

    ### Threat Scoring Formula

    > Score = 0.4 Ã— type_score + 0.3 Ã— velocity_score + 0.2 Ã— direction_score + 0.1 Ã— cluster_score

    ---

    ### Triage Decision Tree

    ```
    IF confidence < 0.50 AND threat < 4.0  â†’ LOGGED  (silent)
    IF confidence â‰¥ 0.65 AND threat â‰¥ 6.0  â†’ ESCALATED
        IF â‰¥3 simultaneous ESCALATED in 5s  â†’ SWARM ALERT (grouped)
    ELSE                                    â†’ WATCH   (dashboard only)
    ```

    ---

    *Reference: V.C. Chen, "The Micro-Doppler Effect in Radar," Artech House, 2011*
    """)


# â”€â”€ Training progress display (one-time, at bottom of page) â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
if st.session_state.training_log:
    with st.expander("ðŸ“Š CNN Training History", expanded=False):
        hist_df = pd.DataFrame(st.session_state.training_log)
        fig_hist = go.Figure()
        fig_hist.add_trace(go.Scatter(
            x=hist_df["epoch"], y=hist_df["train_acc"],
            name="Train Accuracy", line=dict(color="#4488FF")
        ))
        fig_hist.add_trace(go.Scatter(
            x=hist_df["epoch"], y=hist_df["val_acc"],
            name="Val Accuracy", line=dict(color="#FF4488")
        ))
        fig_hist.update_layout(
            title="Training Accuracy",
            xaxis_title="Epoch",
            yaxis_title="Accuracy",
            yaxis=dict(range=[0, 1]),
            template="plotly_dark",
            height=250,
        )
        st.plotly_chart(fig_hist, width='stretch')

