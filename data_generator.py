"""
data_generator.py
=================
Synthetic Micro-Doppler Radar Signal Generator
------------------------------------------------
Generates realistic radar return signals for 5 target classes using
physics-based micro-Doppler models. These are simplified but grounded
in real radar physics — good enough to explain during a viva.

RADAR PHYSICS BACKGROUND
-------------------------
Micro-Doppler refers to additional frequency modulations caused by the
micro-motions of a target (limb swings, rotor blades, vibrations) on
top of the bulk Doppler shift from the target's main body velocity.

  f_doppler = 2 * v_radial / lambda

where:
  v_radial = radial velocity of target (m/s)
  lambda   = radar wavelength (m), e.g. 0.03m for 10 GHz X-band radar

Each target class has a characteristic time-frequency signature:
  - Humans walking/running : periodic sinusoidal limb modulation
  - Animals (dog)          : 4-limb quadruped gait pattern
  - Drones                 : high-frequency rotor blade flash modulations
  - Vehicles               : large steady bulk Doppler, no micro-Doppler

All signals are simulated at a notional Pulse Repetition Frequency (PRF)
of 1000 Hz, giving 512 time samples per signal (~0.5 seconds of observation).
"""

import numpy as np
import pandas as pd
from typing import Tuple, List

# ── Simulation constants ───────────────────────────────────────────────────
SIGNAL_LENGTH = 512       # Number of time samples per signal
PRF           = 1000.0    # Pulse Repetition Frequency (Hz) — sample rate
WAVELENGTH    = 0.03      # Radar wavelength in meters (10 GHz X-band)

# Target class names — order matches label integer indices 0..4
CLASS_NAMES = ["human_walk", "human_run", "animal", "drone", "vehicle"]

# How many samples to generate per class (adjust for more/less data)
SAMPLES_PER_CLASS = 600   # 600 × 5 = 3000 total samples


# ── Helper: Doppler frequency from velocity ────────────────────────────────
def velocity_to_doppler(v_radial_ms: float) -> float:
    """Convert radial velocity (m/s) to Doppler frequency (Hz).
    
    Physics: f_d = 2 * v / lambda
    Factor of 2 because the signal travels to the target AND back.
    """
    return 2.0 * v_radial_ms / WAVELENGTH


# ── Signal generators per class ───────────────────────────────────────────

def _make_human_walk(rng: np.random.Generator, n: int) -> np.ndarray:
    """
    Human Walking Micro-Doppler Model
    -----------------------------------
    A walking human has:
      - Torso: steady ~1.2 m/s radial velocity → bulk Doppler at ~80 Hz
      - Arms/legs: sinusoidal swing at gait cadence ~1.8 Hz
        Arm/leg micro-Doppler amplitude ≈ ±0.5 m/s additional velocity

    The resulting signal is modelled as an amplitude-modulated sinusoid:
      s(t) = A_torso * cos(2π f_torso t)
           + A_limb * cos(2π f_torso t) * cos(2π f_cadence t)
    where the product term creates sidebands at f_torso ± f_cadence.
    """
    t = np.linspace(0, SIGNAL_LENGTH / PRF, SIGNAL_LENGTH)
    signals = []
    for _ in range(n):
        v_torso   = rng.uniform(0.8, 1.6)          # m/s torso speed
        f_torso   = velocity_to_doppler(v_torso)    # bulk Doppler
        cadence   = rng.uniform(1.5, 2.1)           # steps per second
        a_limb    = rng.uniform(0.3, 0.6)           # limb modulation depth
        phase     = rng.uniform(0, 2 * np.pi)

        # Torso return + limb micro-Doppler sidebands
        s = (np.cos(2 * np.pi * f_torso * t + phase)
             + a_limb * np.cos(2 * np.pi * f_torso * t + phase)
             * np.cos(2 * np.pi * cadence * t))
        signals.append(s)
    return np.array(signals)


def _make_human_run(rng: np.random.Generator, n: int) -> np.ndarray:
    """
    Human Running Micro-Doppler Model
    -----------------------------------
    Running differs from walking by:
      - Higher torso velocity: ~3.5 m/s → stronger bulk Doppler
      - Faster cadence: ~3 Hz (more steps per second)
      - Larger limb excursions: arms pump further, legs kick back higher
      - Wider Doppler bandwidth (energy spread over larger freq range)
    """
    t = np.linspace(0, SIGNAL_LENGTH / PRF, SIGNAL_LENGTH)
    signals = []
    for _ in range(n):
        v_torso = rng.uniform(2.5, 5.0)            # faster than walking
        f_torso = velocity_to_doppler(v_torso)
        cadence = rng.uniform(2.5, 3.5)            # faster cadence
        a_limb  = rng.uniform(0.6, 1.0)            # larger limb amplitude
        a_leg   = rng.uniform(0.4, 0.7)            # separate leg contribution
        phase   = rng.uniform(0, 2 * np.pi)

        s = (np.cos(2 * np.pi * f_torso * t + phase)
             + a_limb * np.cos(2 * np.pi * f_torso * t + phase)
             * np.cos(2 * np.pi * cadence * t)
             + a_leg  * np.cos(2 * np.pi * f_torso * t + phase)
             * np.cos(2 * np.pi * (cadence / 2) * t + np.pi / 4))
        signals.append(s)
    return np.array(signals)


def _make_animal(rng: np.random.Generator, n: int) -> np.ndarray:
    """
    Animal (Dog) Quadruped Gait Model
    -----------------------------------
    A dog trotting has:
      - 4 limbs creating a more complex gait pattern than biped
      - Trot cadence ~2.5 Hz for fore-limbs, hind-limbs slightly offset
      - Torso velocity ~1.5–2.5 m/s (variable, lower top speed than human run)
      - Lower body (shorter limbs) → smaller micro-Doppler excursions
      - Hind-limb offset by ~π/4 phase (diagonal trot pattern)
    """
    t = np.linspace(0, SIGNAL_LENGTH / PRF, SIGNAL_LENGTH)
    signals = []
    for _ in range(n):
        v_torso   = rng.uniform(1.0, 2.5)
        f_torso   = velocity_to_doppler(v_torso)
        cadence   = rng.uniform(2.0, 3.0)          # quadruped trot cadence
        a_fore    = rng.uniform(0.2, 0.45)         # fore-limb contribution
        a_hind    = rng.uniform(0.15, 0.35)        # hind-limb (smaller)
        phase_offset = rng.uniform(np.pi/6, np.pi/3)  # diagonal trot offset
        phase = rng.uniform(0, 2 * np.pi)

        s = (np.cos(2 * np.pi * f_torso * t + phase)
             + a_fore * np.cos(2 * np.pi * f_torso * t + phase)
             * np.cos(2 * np.pi * cadence * t)
             + a_hind * np.cos(2 * np.pi * f_torso * t + phase)
             * np.cos(2 * np.pi * cadence * t + phase_offset))
        signals.append(s)
    return np.array(signals)


def _make_drone(rng: np.random.Generator, n: int) -> np.ndarray:
    """
    Drone / UAV Rotor Blade Flash Model
    -------------------------------------
    A multi-rotor drone is characterised by:
      - Very slow or near-zero bulk Doppler (hovering or slow translation)
      - High-frequency rotor blade modulation at the blade passing frequency:
          f_blade = N_blades × RPM / 60
        e.g., 4 blades at 6000 RPM → f_blade = 400 Hz
      - Flash pattern: sharp periodic amplitude spikes (each blade pass)
        modelled as a rectified sinusoid (only positive half-cycles matter)
      - Possibly multiple rotors at slightly different RPMs (adds beat freq)

    Rotor blade flash appears as evenly-spaced sidebands around the
    carrier in the time-frequency spectrogram.
    """
    t = np.linspace(0, SIGNAL_LENGTH / PRF, SIGNAL_LENGTH)
    signals = []
    for _ in range(n):
        v_body   = rng.uniform(0.0, 1.5)           # slow-moving or hovering
        f_body   = velocity_to_doppler(v_body)
        rpm      = rng.uniform(4000, 8000)          # rotor RPM
        n_blades = rng.integers(2, 5)              # 2–4 blades per rotor
        f_blade  = (n_blades * rpm / 60.0)         # blade passing freq (Hz)
        # Clamp to below Nyquist (PRF/2 = 500 Hz)
        f_blade  = min(f_blade, PRF / 2 - 10)

        a_rotor  = rng.uniform(0.8, 1.2)           # strong rotor return
        a_body   = rng.uniform(0.1, 0.3)           # weak body return
        phase    = rng.uniform(0, 2 * np.pi)

        # Body Doppler + periodic rotor blade flash (rectified sine = spikes)
        rotor_flash = a_rotor * np.abs(np.sin(2 * np.pi * f_blade * t + phase))
        s = a_body * np.cos(2 * np.pi * f_body * t) + rotor_flash
        signals.append(s)
    return np.array(signals)


def _make_vehicle(rng: np.random.Generator, n: int) -> np.ndarray:
    """
    Ground Vehicle (Car/Truck) Doppler Model
    -----------------------------------------
    A ground vehicle is characterised by:
      - Large bulk Doppler shift from high speed (10–30 m/s → 667–2000 Hz,
        but we clamp to PRF/2 and normalise the frequency axis)
      - Almost no micro-Doppler — just a slow, steady spectral peak
      - Slight engine vibration: very low-amplitude, low-frequency tremor
        (~10–30 Hz) from the engine block
      - High radar cross-section → strong constant-amplitude return
    """
    t = np.linspace(0, SIGNAL_LENGTH / PRF, SIGNAL_LENGTH)
    signals = []
    for _ in range(n):
        v_body    = rng.uniform(5.0, 20.0)         # 18–72 km/h
        f_body    = velocity_to_doppler(v_body)
        # Doppler may exceed PRF/2 → aliasing (still realistic)
        f_body    = f_body % (PRF / 2)

        a_body    = rng.uniform(0.9, 1.1)          # strong constant return
        f_engine  = rng.uniform(10, 40)            # engine vibration freq
        a_engine  = rng.uniform(0.02, 0.08)        # tiny engine vibration
        phase     = rng.uniform(0, 2 * np.pi)

        s = (a_body   * np.cos(2 * np.pi * f_body   * t + phase)
             + a_engine * np.cos(2 * np.pi * f_engine * t))
        signals.append(s)
    return np.array(signals)


# ── Noise injection ────────────────────────────────────────────────────────

def add_noise(signals: np.ndarray, snr_db: float = 15.0) -> np.ndarray:
    """
    Add Additive White Gaussian Noise (AWGN) at the specified SNR (dB).

    SNR (dB) = 10 * log10(signal_power / noise_power)
    → noise_power = signal_power / 10^(SNR/10)

    Realistic radar SNRs for border surveillance at medium range: 10–20 dB.
    """
    signal_power = np.mean(signals ** 2, axis=1, keepdims=True)
    noise_power  = signal_power / (10 ** (snr_db / 10.0))
    noise        = np.sqrt(noise_power) * np.random.randn(*signals.shape)
    return signals + noise


# ── Public API ─────────────────────────────────────────────────────────────

def generate_dataset(
    samples_per_class: int = SAMPLES_PER_CLASS,
    snr_db: float = 15.0,
    seed: int = 42
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate the full synthetic dataset.

    Returns
    -------
    X : np.ndarray, shape (N, SIGNAL_LENGTH) — raw time-domain signals
    y : np.ndarray, shape (N,)               — integer class labels 0..4
    """
    rng = np.random.default_rng(seed)

    generators = [
        _make_human_walk,
        _make_human_run,
        _make_animal,
        _make_drone,
        _make_vehicle,
    ]

    all_signals, all_labels = [], []
    for label, gen_fn in enumerate(generators):
        sigs = gen_fn(rng, samples_per_class)
        sigs = add_noise(sigs, snr_db=snr_db)
        all_signals.append(sigs)
        all_labels.append(np.full(samples_per_class, label, dtype=np.int64))

    X = np.vstack(all_signals)
    y = np.concatenate(all_labels)

    # Shuffle
    perm = rng.permutation(len(X))
    return X[perm], y[perm]


def generate_single_sample(
    class_label: int,
    snr_db: float = 15.0,
    seed: int | None = None
) -> np.ndarray:
    """
    Generate one signal for a given class label (used for live streaming).

    Parameters
    ----------
    class_label : int   0=human_walk, 1=human_run, 2=animal, 3=drone, 4=vehicle
    snr_db      : float Signal-to-noise ratio in dB
    seed        : optional fixed seed for reproducibility
    """
    rng = np.random.default_rng(seed)
    generators = [
        _make_human_walk,
        _make_human_run,
        _make_animal,
        _make_drone,
        _make_vehicle,
    ]
    sig = generators[class_label](rng, 1)   # shape (1, SIGNAL_LENGTH)
    sig = add_noise(sig, snr_db=snr_db)
    return sig[0]                            # shape (SIGNAL_LENGTH,)


def load_csv(filepath: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load external signal data from a CSV file.

    Expected CSV format
    --------------------
    Column 0: 'label'  — string class name (must match CLASS_NAMES) or integer 0–4
    Columns 1+: signal sample values (s0, s1, ..., sN)

    The signal columns will be resampled/truncated to SIGNAL_LENGTH.

    Example header: label,s0,s1,s2,...,s511
    """
    df = pd.read_csv(filepath)

    # Handle string or integer labels
    if df["label"].dtype == object:
        label_map = {name: i for i, name in enumerate(CLASS_NAMES)}
        y = df["label"].map(label_map).values.astype(np.int64)
    else:
        y = df["label"].values.astype(np.int64)

    # Signal columns are everything after 'label'
    signal_cols = [c for c in df.columns if c != "label"]
    X_raw = df[signal_cols].values.astype(np.float32)

    # Resize each signal to SIGNAL_LENGTH (truncate or zero-pad)
    N = X_raw.shape[0]
    X = np.zeros((N, SIGNAL_LENGTH), dtype=np.float32)
    L = min(X_raw.shape[1], SIGNAL_LENGTH)
    X[:, :L] = X_raw[:, :L]

    return X, y


# ── CLI smoke test ─────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("Generating synthetic dataset...")
    X, y = generate_dataset()
    print(f"Total samples : {len(X)}")
    print(f"Signal length : {X.shape[1]}")
    print(f"Class distribution:")
    for i, name in enumerate(CLASS_NAMES):
        print(f"  [{i}] {name:15s} — {np.sum(y == i)} samples")
    print("\nSample signal stats:")
    print(f"  Mean  : {X.mean():.4f}")
    print(f"  Std   : {X.std():.4f}")
    print(f"  Range : [{X.min():.4f}, {X.max():.4f}]")
    print("\n✓ data_generator.py OK")
