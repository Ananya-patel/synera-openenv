"""
Patient simulator — faithful port from ArogyaLink:
  scripts/data_gen/scenario_generator.py  → trajectory generators
  scripts/data_gen/patient_profiles.py    → 5 patient archetypes

Provides build_trajectory(patient_id, n) which becomes the
environment's reset() data source — exactly as the original mock_simulator does,
but without MQTT/HTTP transport.
"""
from __future__ import annotations
import math
import random
from typing import List, Tuple


# ── Trajectory generators (from scenario_generator.py) ────────────────────────

def flat_with_noise(
    baseline_hr: float, baseline_spo2: float, baseline_temp: float, n: int,
    noise_hr: float = 3, noise_spo2: float = 1, noise_temp: float = 0.2,
) -> List[Tuple[float, float, float, int]]:
    """Gaussian noise around baseline. Returns (hr, spo2, temp, motion) tuples."""
    return [
        (
            max(40.0, baseline_hr + random.gauss(0, noise_hr)),
            max(85.0, min(100.0, baseline_spo2 + random.gauss(0, noise_spo2))),
            baseline_temp + random.gauss(0, noise_temp),
            1,
        )
        for _ in range(n)
    ]


def exponential_acceleration(
    start_hr: float, end_hr: float,
    start_spo2: float, end_spo2: float,
    start_temp: float, end_temp: float, n: int,
) -> List[Tuple[float, float, float, int]]:
    """Accelerating curve: HR rises with increasing rate. From scenario_generator.py."""
    out = []
    for i in range(n):
        t = (i + 1) / n
        x = (math.exp(t * 2) - 1) / (math.e ** 2 - 1)
        hr = start_hr + (end_hr - start_hr) * x
        spo2 = start_spo2 + (end_spo2 - start_spo2) * x
        temp = start_temp + (end_temp - start_temp) * x
        out.append((hr, max(85.0, spo2), temp, 1))
    return out


def exertion_spike(
    baseline_hr: float, baseline_spo2: float, peak_hr: float,
    rise_readings: int, total_readings: int, motion_score: int = 8,
) -> List[Tuple[float, float, float, int]]:
    """Fast HR rise then decay with high motion. Returns (hr, spo2, temp, motion)."""
    out = []
    for i in range(total_readings):
        if i < rise_readings:
            frac = (i + 1) / rise_readings
            hr = baseline_hr + (peak_hr - baseline_hr) * frac
        else:
            frac = 1 - (i - rise_readings) / max(1, total_readings - rise_readings)
            hr = baseline_hr + (peak_hr - baseline_hr) * 0.2 * frac
        out.append((hr, baseline_spo2, 37.0, motion_score))
    return out


def glitch_spike(
    baseline_hr: float, baseline_spo2: float, baseline_temp: float,
    n: int, glitch_value: float, glitch_index: int,
) -> List[Tuple[float, float, float, int]]:
    """One corrupt reading at glitch_index, rest normal. Simulates loose PPG sensor."""
    out = []
    for i in range(n):
        if i == glitch_index:
            out.append((glitch_value, baseline_spo2, baseline_temp, 1))
        else:
            out.append((baseline_hr, baseline_spo2, baseline_temp, 1))
    return out


def slow_linear_drift(
    start_hr: float, end_hr: float, baseline_spo2: float,
    baseline_temp: float, n: int,
) -> List[Tuple[float, float, float, int]]:
    """Linear HR drift — zero second derivative, no acceleration."""
    out = []
    for i in range(n):
        t = i / max(1, n - 1)
        hr = start_hr + (end_hr - start_hr) * t
        out.append((hr, baseline_spo2, baseline_temp, 1))
    return out


# ── Patient profiles (from patient_profiles.py) ────────────────────────────────

PATIENT_PROFILES: dict[str, dict] = {
    "PT-0001": {
        "name": "Rajesh Kumar", "age": 62, "gender": "Male",
        "conditions": ["Hypertension", "Type 2 Diabetes"],
        "baseline_hr": 88.0, "baseline_spo2": 96.0, "baseline_temp": 37.0,
        "trajectory": "flat_with_noise",
        "expected_outcome": "STABLE — no alert ever",
    },
    "PT-0002": {
        "name": "Priya Sharma", "age": 34, "gender": "Female",
        "conditions": ["Post-appendectomy Day 2", "Type 2 Diabetes"],
        "baseline_hr": 82.0, "baseline_spo2": 97.0, "baseline_temp": 37.0,
        "trajectory": "exponential_acceleration",
        "expected_outcome": "SYNERA_STATE fired at ~103 BPM",
    },
    "PT-0003": {
        "name": "Arjun Mehta", "age": 28, "gender": "Male",
        "conditions": [],
        "baseline_hr": 68.0, "baseline_spo2": 99.0, "baseline_temp": 37.0,
        "trajectory": "exertion_spike",
        "expected_outcome": "EXERTION_LOGGED — no alert",
    },
    "PT-0004": {
        "name": "Fatima Begum", "age": 71, "gender": "Female",
        "conditions": ["COPD", "Atrial Fibrillation"],
        "baseline_hr": 79.0, "baseline_spo2": 94.0, "baseline_temp": 37.0,
        "trajectory": "glitch_spike",
        "glitch_value": 228.0, "glitch_index": 3,
        "expected_outcome": "ARTIFACT discarded",
    },
    "PT-0005": {
        "name": "Suresh Patel", "age": 55, "gender": "Male",
        "conditions": ["Hypertension", "CKD Stage 3"],
        "baseline_hr": 76.0, "baseline_spo2": 97.0, "baseline_temp": 37.0,
        "trajectory": "slow_linear_drift",
        "expected_outcome": "WATCH tier — no critical alert",
    },
}


def build_trajectory(patient_id: str, n: int = 70) -> List[Tuple[float, float, float, int]]:
    """
    Build a full vital trajectory for a patient.
    Mirrors mock_simulator.py build_trajectory() but returns data instead of publishing.
    """
    p = PATIENT_PROFILES[patient_id]
    bh, bs, bt = p["baseline_hr"], p["baseline_spo2"], p.get("baseline_temp", 37.0)
    traj = p["trajectory"]

    if traj == "flat_with_noise":
        return flat_with_noise(bh, bs, bt, n)

    if traj == "exponential_acceleration":
        stable = flat_with_noise(bh, bs, bt, 15)
        accel = exponential_acceleration(bh, 160.0, bs, 88.0, bt, 38.3, n - 15)
        return stable + accel

    if traj == "exertion_spike":
        return exertion_spike(bh, bs, 116.0, rise_readings=6, total_readings=n, motion_score=8)

    if traj == "glitch_spike":
        return glitch_spike(bh, bs, bt, n, p.get("glitch_value", 228.0), p.get("glitch_index", 3))

    if traj == "slow_linear_drift":
        return slow_linear_drift(bh, 92.0, bs, bt, n)

    return flat_with_noise(bh, bs, bt, n)
