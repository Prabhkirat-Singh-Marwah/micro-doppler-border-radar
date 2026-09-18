"""
threat_scorer.py
================
Threat Priority Scoring Module
--------------------------------
Computes a numerical threat score (0.0–10.0) for each detected target
and categorises it into a threat level (LOW / MEDIUM / HIGH / CRITICAL).

SCORING RATIONALE (explain this in viva)
-----------------------------------------
Border surveillance systems need to triage many simultaneous detections
in real time. Not all targets are equally dangerous. A threat score
aggregates multiple signals into a single actionable number:

  Score = w_type     × type_score      (most important — what IS it?)
        + w_velocity × velocity_score  (how fast is it moving?)
        + w_direction× direction_score (is it heading toward us?)
        + w_cluster  × cluster_score   (is it part of a group?)

The weights reflect realistic operational priorities:
  - Target type carries the most weight (40%) because a drone crossing
    a border is categorically more concerning than an animal.
  - Velocity (30%) because a fast-moving target gives less response time.
  - Direction (20%) because approaching targets are more threatening
    than departing ones.
  - Cluster size (10%) because swarms multiply the threat.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional
from collections import deque
import time

# ── Class constants ────────────────────────────────────────────────────────
CLASS_NAMES = ["human_walk", "human_run", "animal", "drone", "vehicle"]

# Base threat score for each target type (out of 10)
# Drone is highest — UAVs pose ISR, smuggling, and explosive delivery risks
# Vehicles next — fast, high-payload, hard to stop
# Humans running > walking — intent/urgency indicator
# Animals — lowest, likely false alarm in border context
TYPE_BASE_SCORES = {
    "drone"      : 10.0,
    "vehicle"    : 7.0,
    "human_run"  : 8.0,
    "human_walk" : 6.0,
    "animal"     : 2.0,
}

# Maximum expected radial velocities (m/s) per class — for normalisation
MAX_VELOCITIES = {
    "drone"      : 15.0,   # fast racing drone can do more but border drones ~15
    "vehicle"    : 30.0,
    "human_run"  : 6.0,
    "human_walk" : 2.0,
    "animal"     : 4.0,
}

# Scoring weights (must sum to 1.0)
W_TYPE      = 0.40
W_VELOCITY  = 0.30
W_DIRECTION = 0.20
W_CLUSTER   = 0.10

# Threat level thresholds
THREAT_THRESHOLDS = {
    "CRITICAL": 8.0,
    "HIGH"    : 6.0,
    "MEDIUM"  : 4.0,
    "LOW"     : 0.0,
}

# ── Detection dataclass ───────────────────────────────────────────────────

@dataclass
class Detection:
    """Represents a single target detection event."""
    timestamp  : float               # Unix time of detection
    class_name : str                 # Predicted target class name
    class_idx  : int                 # Integer label 0–4
    confidence : float               # CNN softmax confidence 0–1
    velocity_ms: float               # Estimated radial velocity (m/s)
    direction  : str = "unknown"     # "toward", "away", or "unknown"
    threat_score: float = 0.0        # Computed threat score 0–10
    threat_level: str = "LOW"        # "LOW" / "MEDIUM" / "HIGH" / "CRITICAL"
    status     : str = "PENDING"     # "LOGGED" / "WATCH" / "ESCALATED"
    alert_id   : Optional[str] = None  # Assigned if escalated
    reason     : str = ""            # Reasoning string for logging


# ── Scoring logic ─────────────────────────────────────────────────────────

def compute_threat_score(
    class_name  : str,
    velocity_ms : float,
    direction   : str = "unknown",
    cluster_size: int = 1,
) -> tuple[float, str]:
    """
    Compute the threat score (0–10) and threat level string.

    Parameters
    ----------
    class_name   : str  predicted target class
    velocity_ms  : float estimated radial velocity in m/s (from Doppler)
    direction    : str  "toward", "away", or "unknown"
    cluster_size : int  number of simultaneous detections in current window

    Returns
    -------
    (score, level) : tuple[float, str]
    """
    # --- 1. Type score (0–10, normalised to 0–1 for weighted sum) ---
    type_raw    = TYPE_BASE_SCORES.get(class_name, 5.0)
    type_score  = type_raw / 10.0    # normalise to [0, 1]

    # --- 2. Velocity score ---
    max_v       = MAX_VELOCITIES.get(class_name, 10.0)
    vel_score   = min(velocity_ms / max_v, 1.0)   # clamp to [0, 1]

    # --- 3. Direction score ---
    if direction == "toward":
        dir_score = 1.0    # maximum threat — approaching border
    elif direction == "away":
        dir_score = 0.1    # low threat — retreating
    else:
        dir_score = 0.5    # unknown — neutral assumption

    # --- 4. Cluster score ---
    # Each additional simultaneous detection adds to collective threat
    # Capped at 5 simultaneous targets (beyond 5 the score saturates at 1.0)
    cluster_score = min((cluster_size - 1) / 4.0, 1.0)

    # --- Weighted sum → raw score in [0, 1] ---
    raw = (W_TYPE * type_score
           + W_VELOCITY  * vel_score
           + W_DIRECTION * dir_score
           + W_CLUSTER   * cluster_score)

    # Scale to [0, 10]
    score = round(raw * 10.0, 2)

    # --- Determine threat level ---
    if score >= THREAT_THRESHOLDS["CRITICAL"]:
        level = "CRITICAL"
    elif score >= THREAT_THRESHOLDS["HIGH"]:
        level = "HIGH"
    elif score >= THREAT_THRESHOLDS["MEDIUM"]:
        level = "MEDIUM"
    else:
        level = "LOW"

    return score, level


def assign_direction(class_name: str, rng: np.random.Generator | None = None) -> str:
    """
    Simulate approach/departure direction for a target.

    In a real system this would come from a track filter (Kalman filter)
    tracking the target's range-rate over multiple scans. Here we simulate
    it probabilistically based on class type — drones are more often
    approaching (they are sent across), animals are random, vehicles could go either way.
    """
    if rng is None:
        rng = np.random.default_rng()

    # Probability of 'toward' per class
    probs = {
        "drone"      : 0.70,   # UAVs usually cross toward border
        "vehicle"    : 0.55,   # vehicles can go either way
        "human_run"  : 0.65,   # runners trying to cross
        "human_walk" : 0.50,   # walkers — 50/50
        "animal"     : 0.35,   # animals generally wander
    }
    p = probs.get(class_name, 0.5)
    return "toward" if rng.random() < p else "away"


def score_detection(
    class_name  : str,
    velocity_ms : float,
    cluster_size: int = 1,
    rng         : np.random.Generator | None = None,
) -> tuple[float, str, str]:
    """
    Full scoring pipeline for a single detection.

    Returns (threat_score, threat_level, direction)
    """
    direction = assign_direction(class_name, rng)
    score, level = compute_threat_score(class_name, velocity_ms, direction, cluster_size)
    return score, level, direction


# ── CLI smoke test ─────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("Threat Scorer — test cases\n")
    test_cases = [
        ("drone",       12.0, 3, "High-speed drone swarm"),
        ("human_run",   4.5,  1, "Single running human"),
        ("human_walk",  1.2,  1, "Single walking human"),
        ("animal",      2.0,  1, "Stray animal"),
        ("vehicle",    18.0,  1, "Fast vehicle"),
        ("drone",       2.0,  1, "Slow hovering drone"),
    ]

    rng = np.random.default_rng(99)
    for cls, vel, cluster, desc in test_cases:
        score, level, direction = score_detection(cls, vel, cluster, rng)
        print(f"  {desc}")
        print(f"    class={cls}, vel={vel} m/s, cluster={cluster}, dir={direction}")
        print(f"    → Score: {score}/10  Level: {level}\n")

    print("✓ threat_scorer.py OK")
