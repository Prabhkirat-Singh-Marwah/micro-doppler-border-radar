# 🛡️ AI-Based Micro-Doppler Signature Classification
### Border Surveillance Radar — College Prototype

---

## What This Is

A fully working local Streamlit dashboard that simulates a border surveillance
radar system capable of:

- Generating synthetic micro-Doppler radar signals for 5 target classes
- Converting signals to STFT spectrograms and classifying them with a CNN
- Computing threat priority scores per detection
- Routing detections through a rule-based triage agent
- Displaying everything live in a browser dashboard

---

## Quick Start

### Step 1 — Install dependencies

```powershell
# In your project directory
cd idea_lab

# Install PyTorch CPU-only (important — saves 2 GB vs full install)
pip install torch --index-url https://download.pytorch.org/whl/cpu

# Install everything else
pip install -r requirements.txt
```

### Step 2 — Run the app

```powershell
streamlit run app.py
```

The browser will open automatically at `http://localhost:8501`.

### Step 3 — Use the dashboard

1. Click **"Load / Train Model"** in the sidebar  
   → First run trains the CNN (~3 min on CPU). Weights saved to `classifier_weights.pth`.  
   → Subsequent runs load instantly from the saved file.

2. Click **▶ Start** to begin the live simulation

3. Watch detections stream into the **Live Detection Feed** tab

4. Check **Escalated Alerts** tab for coordinator actions (Accept/Dismiss)

5. View **Trend Chart** for threat score timeline and class distribution

---

## File Structure

```
idea_lab/
├── app.py               # Streamlit dashboard (run this)
├── data_generator.py    # Synthetic micro-Doppler signal generation
├── preprocessor.py      # STFT spectrogram computation + normalization
├── classifier.py        # PyTorch CNN model + train/infer
├── threat_scorer.py     # Threat priority scoring (0–10)
├── triage_agent.py      # Rule-based triage decision engine
├── requirements.txt     # Python dependencies
└── README.md            # This file
```

`classifier_weights.pth` is created automatically after first training run.

---

## Testing Individual Modules

```powershell
python data_generator.py    # prints class distribution
python preprocessor.py      # tests STFT + velocity estimation
python classifier.py        # trains for 5 epochs + runs inference
python threat_scorer.py     # prints threat scores for test cases
python triage_agent.py      # runs 10 detection decision test
```

---

## Using Your Own CSV Data

Upload a CSV file via the sidebar with this format:

```
label,s0,s1,s2,...,s511
human_walk,0.12,0.34,...
drone,0.98,0.87,...
```

- Column `label`: class name string or integer (0–4)
- Remaining columns: signal sample values (up to 512 values; padded/truncated automatically)
- Valid class names: `human_walk`, `human_run`, `animal`, `drone`, `vehicle`

---

## Radar Physics Summary

| Parameter | Value |
|---|---|
| Radar band | X-band (10 GHz) |
| Wavelength | 0.03 m |
| PRF (simulated) | 1000 Hz |
| Signal length | 512 samples (~0.5 s observation) |
| STFT window | Hann, 64 samples, 75% overlap |
| Spectrogram size | 64 × 64 pixels |

### Micro-Doppler Doppler formula
> f_d = 2 × v_radial / λ

---

## Triage Decision Thresholds

| Condition | Status |
|---|---|
| conf < 0.50 AND score < 4.0 | LOGGED (silent) |
| conf ≥ 0.65 AND score ≥ 6.0 | ESCALATED (coordinator alert) |
| ≥3 ESCALATED within 5 s | SWARM ALERT (grouped) |
| Everything else | WATCH (dashboard display only) |

---

*College project — not for operational use.*  
*Reference: V.C. Chen, "The Micro-Doppler Effect in Radar," Artech House, 2011*
