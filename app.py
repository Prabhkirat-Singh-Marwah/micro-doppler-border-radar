"""
app.py
======
AI-Based Micro-Doppler Signature Classification
Border Surveillance Radar — Coordinator Dashboard
--------------------------------------------------
Entry point: `streamlit run app.py`

DASHBOARD OVERVIEW
-------------------
This Streamlit app ties together all modules into a live demo:

  1. Sidebar controls  — configure and start/stop the simulation
  2. Training panel    — train the CNN on first launch (cached after)
  3. Live feed tab     — auto-updating table of recent detections
  4. Alerts tab        — escalated alerts with Accept/Dismiss buttons
  5. Trend chart tab   — detection volume and threat score over time

SIMULATION LOOP
----------------
When simulation is running, each iteration:
  1. Randomly picks a target class (slightly weighted toward drones for drama)
  2. Generates a synthetic micro-Doppler signal via data_generator.py
  3. Converts it to a spectrogram via preprocessor.py
  4. Classifies it with the trained CNN → (class_name, confidence)
  5. Scores the threat via threat_scorer.py → (threat_score, level, direction)
  6. Routes it through triage_agent.py → status + reason
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

# ── Page configuration ────────────────────────────────────────────────────
st.set_page_config(
    page_title="Micro-Doppler Border Surveillance",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Constants ─────────────────────────────────────────────────────────────
MAX_DETECTIONS_DISPLAYED = 100   # keep last N detections in log
REFRESH_INTERVAL_SEC     = 1.5   # seconds between simulation steps
# Class sampling weights for live simulation (drone & human slightly more common)
CLASS_SAMPLING_WEIGHTS   = [0.20, 0.20, 0.15, 0.25, 0.20]

# Threat level → color for UI
THREAT_COLORS = {
    "CRITICAL": "#FF0000",
    "HIGH":     "#FF6600",
    "MEDIUM":   "#FFB300",
    "LOW":      "#00AA44",
}
STATUS_COLORS = {
    "ESCALATED": "🔴",
    "WATCH":     "🟡",
    "LOGGED":    "🟢",
    "PENDING":   "⚪",
}

# ── Session state initialisation ───────────────────────────────────────────

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
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val

    if st.session_state.triage_agent is None:
        st.session_state.triage_agent = TriageAgent()


_init_state()

# ── Helper: train or load model ───────────────────────────────────────────

@st.cache_resource(show_spinner=False)
def get_trained_model(snr_db: float = 15.0):
    """
    Load model from disk if weights exist, otherwise train from scratch.
    Cached by st.cache_resource — runs only once per Streamlit session.
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


# ── Helper: run one simulation step ───────────────────────────────────────

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

    # ── 1. Get a signal ──────────────────────────────────────────
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

    # ── 2. Convert to spectrogram ────────────────────────────────
    spec = signal_to_spectrogram(raw_sig, fs=PRF)   # (1, 64, 64)

    # ── 3. Classify ───────────────────────────────────────────────
    class_name, confidence, probs = predict(model, spec)

    # ── 4. Estimate velocity from Doppler ─────────────────────────
    peak_f   = estimate_peak_doppler_hz(raw_sig, fs=PRF)
    velocity = doppler_to_velocity(peak_f)

    # ── 5. Compute cluster size (how many escalated in last 5 s) ──
    now = time.time()
    recent_escalated = [
        d for d in st.session_state.detections
        if d.status == "ESCALATED" and (now - d.timestamp) < 5.0
    ]
    cluster_size = len(recent_escalated) + 1  # +1 for current detection

    # ── 6. Score threat ───────────────────────────────────────────
    threat_score, threat_level, direction = score_detection(
        class_name, velocity, cluster_size, rng
    )

    # ── 7. Build Detection object ─────────────────────────────────
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

    # ── 8. Triage ─────────────────────────────────────────────────
    agent.process(detection)

    # ── 9. Append to log (keep last N) ───────────────────────────
    st.session_state.detections.append(detection)
    if len(st.session_state.detections) > MAX_DETECTIONS_DISPLAYED:
        st.session_state.detections = st.session_state.detections[-MAX_DETECTIONS_DISPLAYED:]

    st.session_state.total_steps += 1


# ── Helper: build detections DataFrame ────────────────────────────────────

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
            "Status"      : STATUS_COLORS.get(d.status, "⚪") + " " + d.status,
        })
    return pd.DataFrame(rows)


# ═══════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ═══════════════════════════════════════════════════════════════════════════

with st.sidebar:
    st.markdown("## 📡 🛡️")
    st.title("Border Radar Control")
    st.markdown("---")

    # ── Data source ───────────────────────────────────────────────
    st.subheader("📡 Data Source")
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
                st.success(f"✓ Loaded {len(signals)} signals from CSV")
            except Exception as e:
                st.error(f"Error loading CSV: {e}")

    st.markdown("---")

    # ── Noise / SNR ───────────────────────────────────────────────
    st.subheader("⚙️ Signal Parameters")
    snr = st.slider(
        "Signal-to-Noise Ratio (dB)",
        min_value=5, max_value=30, value=15, step=1,
        help="Higher SNR = cleaner signals, easier to classify. Realistic: 10–20 dB"
    )
    st.session_state.snr_db = float(snr)

    st.markdown("---")

    # ── Model training ────────────────────────────────────────────
    st.subheader("🧠 CNN Model")

    if not st.session_state.model_trained:
        if is_trained(DEFAULT_WEIGHTS_PATH):
            st.info("💾 Saved weights found. Loading...")
        else:
            st.warning("⚠️ No saved weights. Will train on first run (~3 min).")

        if st.button("🔧 Load / Train Model", use_container_width=True):
            with st.spinner("Loading / training model (CPU only)..."):
                model, history = get_trained_model(st.session_state.snr_db)
                st.session_state.model         = model
                st.session_state.model_trained = True
                st.session_state.training_log  = history
            st.success("✅ Model ready!")
            st.rerun()
    else:
        st.success("✅ Model loaded")
        if st.session_state.training_log:
            final = st.session_state.training_log[-1]
            st.metric("Val Accuracy", f"{final['val_acc']:.1%}")

    if st.button("🗑️ Retrain (delete weights)", use_container_width=True,
                 disabled=not st.session_state.model_trained):
        if os.path.exists(DEFAULT_WEIGHTS_PATH):
            os.remove(DEFAULT_WEIGHTS_PATH)
        st.session_state.model_trained = False
        st.session_state.model         = None
        st.cache_resource.clear()
        st.rerun()

    st.markdown("---")

    # ── Simulation control ────────────────────────────────────────
    st.subheader("▶️ Simulation")
    if not st.session_state.model_trained:
        st.warning("Load the model first.")
    else:
        col1, col2 = st.columns(2)
        with col1:
            if st.button("▶ Start", use_container_width=True,
                         disabled=st.session_state.running):
                st.session_state.running = True
                st.rerun()
        with col2:
            if st.button("⏹ Stop", use_container_width=True,
                         disabled=not st.session_state.running):
                st.session_state.running = False
                st.rerun()

        if st.button("🔄 Reset", use_container_width=True):
            st.session_state.running      = False
            st.session_state.detections   = []
            st.session_state.triage_agent = TriageAgent()
            st.session_state.total_steps  = 0
            st.session_state.csv_idx      = 0
            st.rerun()

    st.markdown("---")
    st.caption(
        "College prototype — not for operational use.\n"
        "Radar physics references: Chen (2011), *The Micro-Doppler Effect in Radar*"
    )


# ═══════════════════════════════════════════════════════════════════════════
# MAIN DASHBOARD
# ═══════════════════════════════════════════════════════════════════════════

st.title("🛡️ Micro-Doppler Border Surveillance Radar")
st.caption(
    "AI-based classification of micro-Doppler signatures for multi-target "
    "detection and threat prioritisation in low-altitude border surveillance."
)

# ── Status bar ───────────────────────────────────────────────────────────
detections = st.session_state.detections
agent = st.session_state.triage_agent

status_text = "🟢 RUNNING" if st.session_state.running else "⏹ STOPPED"
col_s1, col_s2, col_s3, col_s4, col_s5 = st.columns(5)
col_s1.metric("Status", status_text)
col_s2.metric("Total Detections", st.session_state.total_steps)
col_s3.metric("Escalated", sum(1 for d in detections if d.status == "ESCALATED"))
col_s4.metric("Pending Alerts", len(agent.get_pending_alerts()))
col_s5.metric(
    "Avg Threat Score",
    f"{np.mean([d.threat_score for d in detections]):.1f}/10" if detections else "—"
)

st.markdown("---")

# ── Run simulation step if running ───────────────────────────────────────
if st.session_state.running and st.session_state.model_trained:
    run_one_step()
    time.sleep(REFRESH_INTERVAL_SEC)
    st.rerun()
elif st.session_state.running and not st.session_state.model_trained:
    st.warning("⚠️ Cannot run simulation — load the model first (sidebar).")
    st.session_state.running = False

# ── Tabs ─────────────────────────────────────────────────────────────────
tab_feed, tab_alerts, tab_chart, tab_about = st.tabs(
    ["📋 Live Detection Feed", "🚨 Escalated Alerts", "📈 Trend Chart", "ℹ️ About"]
)

# ══════════════════════════════════════════
# TAB 1: LIVE DETECTION FEED
# ══════════════════════════════════════════
with tab_feed:
    st.subheader("Live Detection Feed (newest first)")

    if not detections:
        st.info("No detections yet. Load the model and start the simulation.")
    else:
        df = detections_to_df(detections)

        # Color-code rows by threat level using Streamlit's native style
        def highlight_row(row):
            status = row.get("Status", "")
            if "ESCALATED" in status:
                return ["background-color: #3d0000; color: white"] * len(row)
            elif "WATCH" in status:
                return ["background-color: #3d2d00; color: white"] * len(row)
            else:
                return [""] * len(row)

        styled_df = df.style.apply(highlight_row, axis=1)
        st.dataframe(styled_df, use_container_width=True, height=400)

    # Class distribution breakdown
    if detections:
        st.markdown("**Detection breakdown by class:**")
        class_counts = {}
        for d in detections:
            class_counts[d.class_name] = class_counts.get(d.class_name, 0) + 1
        cols = st.columns(len(class_counts))
        for i, (cls, cnt) in enumerate(class_counts.items()):
            pct = cnt / len(detections) * 100
            cols[i].metric(cls.replace("_", " ").title(), f"{cnt}", f"{pct:.0f}%")


# ══════════════════════════════════════════
# TAB 2: ESCALATED ALERTS PANEL
# ══════════════════════════════════════════
with tab_alerts:
    st.subheader("🚨 Coordinator Alerts — Action Required")

    all_alerts = agent.get_alerts()
    pending    = [a for a in all_alerts if a.status == "PENDING"]
    closed     = [a for a in all_alerts if a.status != "PENDING"]

    if not all_alerts:
        st.info("No alerts yet. Escalated detections will appear here.")
    else:
        # ── Pending alerts ─────────────────────────────────────────
        if pending:
            st.markdown(f"**{len(pending)} pending alert(s) — awaiting action:**")
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
                            f"{'🌊' if is_swarm else '⚠️'} **{alert.alert_type}**"
                        )
                        st.caption(f"ID: {alert.alert_id}")

                    with cols_a[1]:
                        st.markdown(f"**{ts_str}** — {alert.summary}")
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
                        if st.button("✅ Accept", key=f"accept_{alert.alert_id}",
                                     use_container_width=True):
                            agent.accept_alert(alert.alert_id)
                            st.rerun()

                    with cols_a[3]:
                        if st.button("❌ Dismiss", key=f"dismiss_{alert.alert_id}",
                                     use_container_width=True):
                            agent.dismiss_alert(alert.alert_id)
                            st.rerun()
        else:
            st.success("✅ All alerts have been actioned.")

        # ── Closed alerts history ──────────────────────────────────
        if closed:
            with st.expander(f"📁 Actioned alerts ({len(closed)} total)"):
                for alert in reversed(closed):
                    ts_str = datetime.datetime.fromtimestamp(alert.timestamp).strftime(
                        "%H:%M:%S"
                    )
                    icon   = "✅" if alert.status == "ACCEPTED" else "❌"
                    st.markdown(
                        f"{icon} **{alert.status}** — `{alert.alert_id}` @ {ts_str} — "
                        f"{alert.summary[:80]}"
                    )


# ══════════════════════════════════════════
# TAB 3: TREND CHART
# ══════════════════════════════════════════
with tab_chart:
    st.subheader("📈 Detection Trend Over Session")

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

        # ── Threat score over time ────────────────────────────────
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
            yaxis_title="Threat Score (0–10)",
            yaxis=dict(range=[0, 10]),
            template="plotly_dark",
            height=300,
            margin=dict(l=40, r=20, t=50, b=40),
        )
        st.plotly_chart(fig_threat, use_container_width=True)

        # ── Detection class distribution pie ──────────────────────
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
            st.plotly_chart(fig_pie, use_container_width=True)

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
            st.plotly_chart(fig_conf, use_container_width=True)

        # ── Status counts over time ───────────────────────────────
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


# ══════════════════════════════════════════
# TAB 4: ABOUT / REFERENCE
# ══════════════════════════════════════════
with tab_about:
    st.subheader("ℹ️ System Architecture & Physics Reference")

    st.markdown("""
    ### Micro-Doppler Physics

    **Micro-Doppler** refers to additional frequency modulations on top of the
    bulk Doppler shift, caused by the micro-motions of target components
    (limb swings, rotor blades, vibrations).

    The Doppler frequency equation:
    > f_d = 2 × v_radial / λ

    where **v_radial** is the target's radial velocity (m/s) and **λ** is the
    radar wavelength (0.03 m for 10 GHz X-band radar).

    | Target | Signature |
    |---|---|
    | Human walk | Sinusoidal band at ~80 Hz ± gait sidebands at ±1.8 Hz |
    | Human run | Wider band, faster cadence (~3 Hz sidebands) |
    | Animal (dog) | Quadruped 4-limb pattern, dual-cadence sidebands |
    | Drone | Near-DC body + high-freq rotor flash (blade × RPM / 60) |
    | Vehicle | Large steady Doppler, engine micro-tremor only |

    ---

    ### CNN Architecture

    ```
    Input (1 × 64 × 64 spectrogram)
      └─ Conv2d(1→16, 3×3) + ReLU + MaxPool → (16 × 32 × 32)
      └─ Conv2d(16→32, 3×3) + ReLU + MaxPool → (32 × 16 × 16)
      └─ Conv2d(32→64, 3×3) + ReLU + MaxPool → (64 × 8 × 8)
      └─ Flatten → Linear(4096→128) + ReLU + Dropout(0.3)
      └─ Linear(128→5) → Softmax → class probabilities
    ```

    ---

    ### Threat Scoring Formula

    > Score = 0.4 × type_score + 0.3 × velocity_score + 0.2 × direction_score + 0.1 × cluster_score

    ---

    ### Triage Decision Tree

    ```
    IF confidence < 0.50 AND threat < 4.0  → LOGGED  (silent)
    IF confidence ≥ 0.65 AND threat ≥ 6.0  → ESCALATED
        IF ≥3 simultaneous ESCALATED in 5s  → SWARM ALERT (grouped)
    ELSE                                    → WATCH   (dashboard only)
    ```

    ---

    *Reference: V.C. Chen, "The Micro-Doppler Effect in Radar," Artech House, 2011*
    """)


# ── Training progress display (one-time, at bottom of page) ───────────────
if st.session_state.training_log:
    with st.expander("📊 CNN Training History", expanded=False):
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
        st.plotly_chart(fig_hist, use_container_width=True)
