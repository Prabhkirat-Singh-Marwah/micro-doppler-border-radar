"""
triage_agent.py
===============
Rule-Based Triage Agent
------------------------
Continuously monitors classifier + threat scoring output and applies
a deterministic decision tree to route each detection to the correct
status bucket: LOGGED, WATCH, or ESCALATED.

WHY RULE-BASED (NOT ML/LLM)?
------------------------------
In safety-critical surveillance, you need:
  1. Explainability — an operator must understand WHY a detection was escalated
  2. Determinism     — the same inputs should always produce the same decision
  3. Auditability    — every decision must be logged with a human-readable reason

An LLM or second ML model would be a black box here. A rule-based agent
is perfectly appropriate and much simpler to validate during a viva.

DECISION LOGIC
---------------
The agent applies this tree in order:

  if confidence < CONF_LOW AND threat_score < THREAT_LOW:
      → LOGGED   (silent — very low confidence + low threat, probably noise)

  elif confidence >= CONF_HIGH AND threat_score >= THREAT_HIGH:
      → ESCALATED (high confidence + high threat → coordinator action needed)

      Swarm check: if ≥ SWARM_THRESHOLD simultaneous ESCALATED detections
      exist within SWARM_WINDOW_SEC seconds → group into a SWARM ALERT

  else:
      → WATCH  (ambiguous — show in dashboard but don't alert)

THRESHOLDS (tune for sensitivity vs. false-alarm rate trade-off)
  CONF_LOW    = 0.50  — below this confidence, result is too uncertain
  CONF_HIGH   = 0.65  — above this, we trust the classifier
  THREAT_LOW  = 4.0   — below this, target is not operationally significant
  THREAT_HIGH = 6.0   — above this, requires coordinator attention
  SWARM_WINDOW_SEC = 5.0  — time window to detect simultaneous threats
  SWARM_THRESHOLD  = 3    — min number of ESCALATED detections to form a swarm
"""

import time
import uuid
import logging
from collections import deque
from dataclasses import dataclass, field
from typing import List, Optional, Deque

from threat_scorer import Detection

# ── Thresholds ─────────────────────────────────────────────────────────────
CONF_LOW          = 0.50   # confidence below → too uncertain to act
CONF_HIGH         = 0.65   # confidence above → trust the classifier
THREAT_LOW        = 4.0    # threat score below → operationally insignificant
THREAT_HIGH       = 6.0    # threat score above → escalate to coordinator
SWARM_WINDOW_SEC  = 5.0    # seconds: window to detect simultaneous threats
SWARM_THRESHOLD   = 3      # ≥ this many simultaneous ESCALATED → swarm alert

# ── Logging setup ──────────────────────────────────────────────────────────
logging.basicConfig(
    level   = logging.INFO,
    format  = "%(asctime)s [TRIAGE] %(message)s",
    datefmt = "%H:%M:%S",
)
logger = logging.getLogger("triage_agent")


# ── Alert dataclass ────────────────────────────────────────────────────────

@dataclass
class Alert:
    """Represents a coordinator alert (ESCALATED or SWARM)."""
    alert_id    : str
    alert_type  : str              # "INDIVIDUAL" or "SWARM"
    timestamp   : float
    detections  : List[Detection]  # 1 for individual, ≥3 for swarm
    status      : str = "PENDING"  # "PENDING" / "ACCEPTED" / "DISMISSED"
    summary     : str = ""         # Human-readable coordinator summary

    def accept(self):
        self.status = "ACCEPTED"

    def dismiss(self):
        self.status = "DISMISSED"


# ── Triage Agent ───────────────────────────────────────────────────────────

class TriageAgent:
    """
    Rule-based triage agent that processes detections and generates alerts.

    Usage
    -----
    agent = TriageAgent()
    agent.process(detection)      # route one detection
    alerts = agent.get_alerts()   # get all pending alerts
    log   = agent.get_log()       # get full decision log
    """

    def __init__(self):
        # Rolling window of recent ESCALATED detections (for swarm detection)
        self._escalated_window: Deque[Detection] = deque()
        # All alerts generated so far
        self._alerts: List[Alert] = []
        # Full decision log (all detections, all decisions)
        self._log: List[dict] = []

    # ── Public interface ───────────────────────────────────────────────────

    def process(self, detection: Detection) -> Detection:
        """
        Route a single detection through the triage decision tree.
        Updates detection.status and detection.reason in-place.

        Returns the detection (with status and reason filled in).
        """
        # ── Step 1: Apply decision tree ───────────────────────────
        conf  = detection.confidence
        score = detection.threat_score

        if conf < CONF_LOW and score < THREAT_LOW:
            # Low confidence + low threat → silent log, no alert
            status = "LOGGED"
            reason = (f"Silent log: confidence {conf:.2f} < {CONF_LOW} "
                      f"AND threat score {score:.1f} < {THREAT_LOW}")

        elif conf >= CONF_HIGH and score >= THREAT_HIGH:
            # High confidence + high threat → escalate
            status = "ESCALATED"
            reason = (f"Escalated: {detection.class_name} detected, "
                      f"confidence {conf:.2f}, threat score {score:.1f}/10, "
                      f"direction={detection.direction}")
            detection.alert_id = str(uuid.uuid4())[:8]
            self._escalated_window.append(detection)

        else:
            # In-between → watch, display on dashboard, no alert
            status = "WATCH"
            reason = (f"Watch: conf={conf:.2f}, threat={score:.1f}/10 — "
                      f"below escalation threshold, above ignore threshold")

        detection.status = status
        detection.reason = reason

        # ── Step 2: Swarm check (only after ESCALATED) ────────────
        if status == "ESCALATED":
            self._check_swarm()

        # ── Step 3: Generate individual alert if ESCALATED + no swarm ──
        if status == "ESCALATED":
            # Check if this detection was consumed by a swarm alert
            already_grouped = any(
                detection in alert.detections
                for alert in self._alerts
                if alert.alert_type == "SWARM"
            )
            if not already_grouped:
                self._create_individual_alert(detection)

        # ── Step 4: Log the decision ───────────────────────────────
        self._log_decision(detection)

        return detection

    def get_alerts(self) -> List[Alert]:
        """Return all alerts (including accepted/dismissed)."""
        return list(self._alerts)

    def get_pending_alerts(self) -> List[Alert]:
        """Return only alerts awaiting coordinator action."""
        return [a for a in self._alerts if a.status == "PENDING"]

    def get_log(self) -> List[dict]:
        """Return the full decision log as a list of dicts."""
        return list(self._log)

    def accept_alert(self, alert_id: str) -> bool:
        """Mark an alert as accepted by the coordinator."""
        for alert in self._alerts:
            if alert.alert_id == alert_id:
                alert.accept()
                logger.info(f"Alert {alert_id} ACCEPTED by coordinator")
                return True
        return False

    def dismiss_alert(self, alert_id: str) -> bool:
        """Mark an alert as dismissed by the coordinator."""
        for alert in self._alerts:
            if alert.alert_id == alert_id:
                alert.dismiss()
                logger.info(f"Alert {alert_id} DISMISSED by coordinator")
                return True
        return False

    # ── Internal helpers ───────────────────────────────────────────────────

    def _prune_escalated_window(self):
        """
        Remove detections from the swarm window that are older than
        SWARM_WINDOW_SEC. We check the window every time a new ESCALATED
        detection arrives.
        """
        now = time.time()
        while self._escalated_window and \
              (now - self._escalated_window[0].timestamp) > SWARM_WINDOW_SEC:
            self._escalated_window.popleft()

    def _check_swarm(self):
        """
        Check if the recent ESCALATED detections constitute a swarm.

        A swarm is defined as ≥ SWARM_THRESHOLD ESCALATED detections within
        SWARM_WINDOW_SEC seconds. When detected, we:
          1. Consume those detections into a single SWARM ALERT
          2. Remove them from the rolling window
          3. Cancel any individual alerts for those detections
        """
        self._prune_escalated_window()

        if len(self._escalated_window) < SWARM_THRESHOLD:
            return   # Not enough simultaneous escalated detections

        # Collect the detections that form the swarm
        swarm_detections = list(self._escalated_window)
        self._escalated_window.clear()   # consume them

        # Remove any individual alerts already created for these detections
        consumed_alert_ids = {d.alert_id for d in swarm_detections if d.alert_id}
        self._alerts = [a for a in self._alerts
                        if a.alert_id not in consumed_alert_ids]

        # Create the aggregated swarm alert
        classes_in_swarm = [d.class_name for d in swarm_detections]
        class_counts = {}
        for c in classes_in_swarm:
            class_counts[c] = class_counts.get(c, 0) + 1
        summary_parts = [f"{v}× {k}" for k, v in class_counts.items()]
        summary = (f"SWARM ALERT: {len(swarm_detections)} simultaneous targets "
                   f"({', '.join(summary_parts)}) — multi-target incursion detected")

        swarm_alert = Alert(
            alert_id    = "SWARM-" + str(uuid.uuid4())[:6].upper(),
            alert_type  = "SWARM",
            timestamp   = time.time(),
            detections  = swarm_detections,
            summary     = summary,
        )
        self._alerts.append(swarm_alert)

        logger.warning(
            f"SWARM ALERT generated: {len(swarm_detections)} targets | {summary}"
        )

    def _create_individual_alert(self, detection: Detection):
        """Create a single-detection ESCALATED alert."""
        alert = Alert(
            alert_id   = detection.alert_id,
            alert_type = "INDIVIDUAL",
            timestamp  = detection.timestamp,
            detections = [detection],
            summary    = detection.reason,
        )
        self._alerts.append(alert)
        logger.warning(f"ALERT {detection.alert_id}: {detection.reason}")

    def _log_decision(self, detection: Detection):
        """Append decision to the in-memory log."""
        entry = {
            "timestamp"   : detection.timestamp,
            "class"       : detection.class_name,
            "confidence"  : round(detection.confidence, 4),
            "threat_score": detection.threat_score,
            "threat_level": detection.threat_level,
            "direction"   : detection.direction,
            "velocity_ms" : round(detection.velocity_ms, 2),
            "status"      : detection.status,
            "reason"      : detection.reason,
        }
        self._log.append(entry)
        # Log to Python logger (appears in console)
        logger.info(
            f"{detection.status:10s} | {detection.class_name:12s} | "
            f"conf={detection.confidence:.2f} | score={detection.threat_score:.1f} | "
            f"{detection.reason[:60]}"
        )


# ── CLI smoke test ─────────────────────────────────────────────────────────

if __name__ == "__main__":
    import numpy as np
    from threat_scorer import score_detection, Detection

    print("Triage Agent — 10 detection smoke test\n")

    agent = TriageAgent()
    rng   = np.random.default_rng(77)

    # Simulate 10 detections
    test_scenarios = [
        ("drone",      12.0, 0.92),   # should ESCALATE
        ("human_run",   4.0, 0.78),   # should ESCALATE or WATCH
        ("animal",      1.5, 0.88),   # likely LOGGED (low threat)
        ("drone",       8.0, 0.71),   # ESCALATE
        ("drone",      10.0, 0.85),   # ESCALATE → may trigger SWARM
        ("vehicle",    15.0, 0.60),   # borderline
        ("human_walk",  1.0, 0.40),   # LOGGED
        ("animal",      2.0, 0.30),   # LOGGED
        ("human_run",   5.0, 0.80),   # ESCALATE
        ("drone",       6.0, 0.55),   # WATCH (low conf)
    ]

    for cls, vel, conf in test_scenarios:
        score, level, direction = score_detection(cls, vel, 1, rng)
        det = Detection(
            timestamp   = time.time(),
            class_name  = cls,
            class_idx   = ["human_walk","human_run","animal","drone","vehicle"].index(cls),
            confidence  = conf,
            velocity_ms = vel,
            direction   = direction,
            threat_score= score,
            threat_level= level,
        )
        agent.process(det)
        time.sleep(0.1)  # simulate slight time gap

    # Print summary
    print("\n--- Decision Log ---")
    for entry in agent.get_log():
        import datetime
        ts = datetime.datetime.fromtimestamp(entry["timestamp"]).strftime("%H:%M:%S")
        print(f"  {ts} | {entry['status']:10s} | {entry['class']:12s} | "
              f"conf={entry['confidence']:.2f} | score={entry['threat_score']:.1f}")

    pending = agent.get_pending_alerts()
    print(f"\n--- Pending Alerts: {len(pending)} ---")
    for alert in pending:
        print(f"  [{alert.alert_type}] {alert.alert_id}: {alert.summary}")

    print("\n✓ triage_agent.py OK")
