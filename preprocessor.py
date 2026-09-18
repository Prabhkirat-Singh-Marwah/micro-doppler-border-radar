"""
preprocessor.py
===============
STFT Spectrogram Pipeline
--------------------------
Converts raw time-domain radar signals into 2D log-power spectrograms
suitable as input images for the CNN classifier.

WHY SPECTROGRAMS?
-----------------
A raw time-domain signal looks like a 1-D waveform — hard to classify.
The Short-Time Fourier Transform (STFT) slides a window across the signal
and computes the frequency content at each time step, giving a 2-D
time-frequency image (spectrogram). This reveals:

  - Bulk Doppler: a horizontal band at the target's main frequency
  - Micro-Doppler: sinusoidal or periodic fluctuations around that band
  - Rotor flash: high-frequency vertical stripes (drone blades)
  - Gait cadence: repeating modulation bands (human limbs)

These patterns are visually distinct — hence a CNN can classify them
just like it classifies images.

STFT PARAMETERS (explained for viva)
-------------------------------------
  nperseg   = 64   : Each window covers 64 samples = 64 ms at 1 kHz PRF
                     Short enough to track micro-Doppler changes in time,
                     long enough for decent frequency resolution (≈15.6 Hz/bin)
  noverlap  = 48   : 75% overlap → smooth time axis, 16-sample hop
  window    = hann : Reduces spectral leakage at window edges

Output shape after STFT: (freq_bins=33, time_frames≈31)  [for 512 samples]
After resize to (64, 64): consistent square image for CNN.
"""

import numpy as np
from scipy.signal import stft
from scipy.ndimage import zoom
from typing import Union

# ── STFT hyperparameters ───────────────────────────────────────────────────
NPERSEG  = 64     # window size in samples
NOVERLAP = 48     # overlap (75%) — controls time-axis resolution
OUT_H    = 64     # output spectrogram height (freq axis)
OUT_W    = 64     # output spectrogram width  (time axis)
EPS      = 1e-9   # small constant to avoid log(0)


def signal_to_spectrogram(signal: np.ndarray, fs: float = 1000.0) -> np.ndarray:
    """
    Convert a 1-D time-domain radar signal to a 2-D log-power spectrogram.

    Steps
    -----
    1. STFT          → complex spectrogram (freq_bins × time_frames)
    2. Power         → |Z|^2 (power spectral density at each cell)
    3. Log scale     → 10*log10(power + eps)  — compress dynamic range
    4. Resize        → zoom to (OUT_H, OUT_W)  — fixed size for CNN
    5. Normalise     → scale each spectrogram to [0, 1] independently

    Parameters
    ----------
    signal : np.ndarray  shape (signal_length,) — single raw signal
    fs     : float       sample rate (PRF), default 1000 Hz

    Returns
    -------
    spec : np.ndarray  shape (1, OUT_H, OUT_W)  — channel-first for PyTorch
    """
    # Step 1: STFT
    # f = frequency bins, t = time frames, Zxx = complex STFT coefficients
    _, _, Zxx = stft(signal, fs=fs, nperseg=NPERSEG, noverlap=NOVERLAP,
                     window="hann")

    # Step 2 & 3: Log-power spectrogram
    power = np.abs(Zxx) ** 2                  # power (freq × time)
    log_power = 10.0 * np.log10(power + EPS)  # dB scale

    # Step 4: Resize to fixed (OUT_H, OUT_W)
    h, w = log_power.shape
    zoom_h = OUT_H / h
    zoom_w = OUT_W / w
    resized = zoom(log_power, (zoom_h, zoom_w), order=1)  # bilinear

    # Step 5: Per-sample min-max normalisation → [0, 1]
    s_min, s_max = resized.min(), resized.max()
    if s_max - s_min > 0:
        normalised = (resized - s_min) / (s_max - s_min)
    else:
        normalised = np.zeros_like(resized)

    # Add channel dimension: (H, W) → (1, H, W)
    return normalised[np.newaxis, :, :]  # shape (1, 64, 64)


def batch_to_spectrograms(signals: np.ndarray, fs: float = 1000.0) -> np.ndarray:
    """
    Convert a batch of signals to spectrograms.

    Parameters
    ----------
    signals : np.ndarray  shape (N, signal_length)
    fs      : float       sample rate

    Returns
    -------
    specs : np.ndarray  shape (N, 1, OUT_H, OUT_W)
    """
    specs = []
    for i, sig in enumerate(signals):
        spec = signal_to_spectrogram(sig, fs=fs)
        specs.append(spec)
        if (i + 1) % 500 == 0:
            print(f"  Preprocessed {i + 1}/{len(signals)} signals...")
    return np.array(specs, dtype=np.float32)  # (N, 1, 64, 64)


def estimate_peak_doppler_hz(signal: np.ndarray, fs: float = 1000.0) -> float:
    """
    Estimate the peak Doppler frequency from a signal using FFT.

    This is used by the threat scorer to estimate target velocity.
    Returns the frequency bin (in Hz) with the highest power,
    excluding DC (0 Hz).

    Physics: v_radial = f_peak * lambda / 2
    """
    spectrum = np.abs(np.fft.rfft(signal)) ** 2  # power spectrum
    freqs    = np.fft.rfftfreq(len(signal), d=1.0 / fs)

    # Exclude DC and very low frequencies (< 5 Hz) to ignore static clutter
    mask = freqs > 5.0
    if not np.any(mask):
        return 0.0

    peak_idx  = np.argmax(spectrum[mask])
    peak_freq = freqs[mask][peak_idx]
    return float(peak_freq)


def doppler_to_velocity(freq_hz: float, wavelength_m: float = 0.03) -> float:
    """
    Convert a Doppler frequency to estimated radial velocity (m/s).

    v = f_d * lambda / 2
    """
    return freq_hz * wavelength_m / 2.0


# ── CLI smoke test ─────────────────────────────────────────────────────────

if __name__ == "__main__":
    from data_generator import generate_dataset, CLASS_NAMES, PRF

    print("Generating small dataset for spectrogram test...")
    X, y = generate_dataset(samples_per_class=10)

    print(f"Running STFT on {len(X)} signals...")
    specs = batch_to_spectrograms(X, fs=PRF)

    print(f"\nSpectrogram batch shape : {specs.shape}")
    print(f"Value range             : [{specs.min():.4f}, {specs.max():.4f}]")
    print(f"Dtype                   : {specs.dtype}")

    # Test peak Doppler estimation
    sample_signal = X[0]
    peak_f  = estimate_peak_doppler_hz(sample_signal, fs=PRF)
    peak_v  = doppler_to_velocity(peak_f)
    print(f"\nExample signal (class={CLASS_NAMES[y[0]]}):")
    print(f"  Peak Doppler freq : {peak_f:.1f} Hz")
    print(f"  Estimated velocity: {peak_v:.2f} m/s")
    print("\n✓ preprocessor.py OK")
