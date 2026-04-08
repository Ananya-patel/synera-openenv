"""
Synera core engine — standalone port.

Ported faithfully from ArogyaLink:
  backend/core/trajectory/window_buffer.py   → WindowBuffer, VitalReading
  backend/core/trajectory/derivatives.py     → derivative math (stdlib, no numpy)
  backend/core/trajectory/calculator.py      → trajectory_verdict()
  backend/core/rules/rule_a_artifact.py      → rule_a_artifact()
  backend/core/rules/rule_b_exertion.py      → rule_b_exertion()
  backend/core/rules/rule_c_trajectory.py    → rule_c_trajectory()

Zero external dependencies — pure Python stdlib only.
"""
from __future__ import annotations
from collections import deque
from dataclasses import dataclass
from typing import Optional

# ── Data model ─────────────────────────────────────────────────────────────────

@dataclass
class VitalReading:
    """Single vital snapshot (one reading from wearable sensor)."""
    heart_rate: Optional[float] = None
    spo2: Optional[float] = None
    temperature: Optional[float] = None
    motion_score: int = 0


# ── Sliding window buffer ──────────────────────────────────────────────────────

WINDOW_SIZE = 10
DT = 5.0  # seconds between readings


class WindowBuffer:
    """Fixed-size sliding window; oldest reading dropped when full."""
    __slots__ = ("_dq",)

    def __init__(self, maxlen: int = WINDOW_SIZE):
        self._dq: deque[VitalReading] = deque(maxlen=maxlen)

    def append(self, r: VitalReading) -> None:
        self._dq.append(r)

    def get_readings(self) -> list[VitalReading]:
        return list(self._dq)

    def get_heart_rates(self) -> list[float]:
        return [r.heart_rate for r in self._dq if r.heart_rate is not None]

    def get_spo2(self) -> list[float]:
        return [r.spo2 for r in self._dq if r.spo2 is not None]

    def get_temperatures(self) -> list[float]:
        return [r.temperature for r in self._dq if r.temperature is not None]

    def last_reading(self) -> Optional[VitalReading]:
        return self._dq[-1] if self._dq else None

    def __len__(self) -> int:
        return len(self._dq)


# ── Derivative maths (stdlib-only port of derivatives.py) ─────────────────────

def _first_derivative(values: list[float], dt: float = DT) -> list[float]:
    """Central difference; forward/backward at edges. Faithful to original."""
    n = len(values)
    if n < 2:
        return [0.0] * n
    rates = [0.0] * n
    rates[0] = (values[1] - values[0]) / dt
    rates[n - 1] = (values[n - 1] - values[n - 2]) / dt
    for i in range(1, n - 1):
        rates[i] = (values[i + 1] - values[i - 1]) / (2.0 * dt)
    return rates


def _second_derivative(rates: list[float], dt: float = DT) -> list[float]:
    return _first_derivative(rates, dt)


def _sigma_deviation(value: float, mean: float, std: float) -> float:
    if std == 0:
        return 0.0
    return abs(value - mean) / std


def _is_sustained_acceleration(accels: list[float], window: int = 3) -> bool:
    """Majority-positive logic — matches original is_sustained_acceleration."""
    if len(accels) < window:
        return False
    tail = accels[-window:]
    pos = sum(1 for a in tail if a > 0)
    return pos >= (window // 2 + 1) if window > 2 else pos >= 1


# ── Trajectory verdict ─────────────────────────────────────────────────────────

def trajectory_verdict(
    buffer: WindowBuffer,
    vital: str,
    baseline_mean: float,
    baseline_std: float,
    accel_window: int = 3,
) -> dict:
    """Faithful port of get_trajectory_verdict() from calculator.py."""
    if vital == "heart_rate":
        values = buffer.get_heart_rates()
    elif vital == "spo2":
        values = buffer.get_spo2()
    elif vital == "temperature":
        values = buffer.get_temperatures()
    else:
        values = buffer.get_heart_rates()

    if len(values) < 3:
        return {
            "deviation_sigma": 0.0, "second_derivative": 0.0,
            "is_sustained_acceleration": False,
            "current_value": values[-1] if values else 0.0,
            "rates": [], "accelerations": [],
        }

    current = values[-1]
    dev = _sigma_deviation(current, baseline_mean, baseline_std)
    rates = _first_derivative(values)
    accels = _second_derivative(rates)
    second_deriv = accels[-1] if accels else 0.0
    sustained = _is_sustained_acceleration(accels, accel_window)

    # Fallback: consistently positive first-derivative (original plateau fix)
    # Requires BOTH positive velocity AND at least one positive acceleration,
    # so a pure linear drift (constant slope, zero acceleration) does NOT fire.
    if not sustained and dev >= 1.5 and len(rates) >= accel_window:
        positive_velocity = sum(1 for r in rates[-accel_window:] if r > 0) >= accel_window
        positive_accel = any(a > 0 for a in accels[-accel_window:])
        if positive_velocity and positive_accel:
            sustained = True

    return {
        "deviation_sigma": dev, "second_derivative": second_deriv,
        "is_sustained_acceleration": sustained, "current_value": current,
        "rates": rates, "accelerations": accels,
    }


# ── Three Synera rules ─────────────────────────────────────────────────────────

def rule_a_artifact(
    reading: VitalReading, buffer: WindowBuffer,
    hr_delta_max: int = 40, spo2_delta_max: int = 5, temp_delta_max: float = 0.5,
) -> bool:
    """Rule A — True if reading is a sensor artifact. Port of rule_a_artifact.py."""
    last = buffer.last_reading()
    if last is None:
        return False
    if reading.heart_rate is not None and last.heart_rate is not None:
        if abs(reading.heart_rate - last.heart_rate) >= hr_delta_max:
            return True
    if reading.spo2 is not None and last.spo2 is not None:
        if abs(reading.spo2 - last.spo2) >= spo2_delta_max:
            return True
    if reading.temperature is not None and last.temperature is not None:
        if abs(reading.temperature - last.temperature) >= temp_delta_max:
            return True
    return False


def rule_b_exertion(
    reading: VitalReading, baseline_hr: float,
    hr_elevation_threshold: int = 15, motion_threshold: int = 4,
) -> bool:
    """Rule B — True if HR elevation is explained by motion. Port of rule_b_exertion.py."""
    motion = reading.motion_score or 0
    if motion <= motion_threshold:
        return False
    if reading.heart_rate is None:
        return False
    return (reading.heart_rate - baseline_hr) > hr_elevation_threshold


def rule_c_trajectory(
    buffer: WindowBuffer, baseline_mean: float, baseline_std: float,
    motion_score: int, vital: str = "heart_rate",
    sigma_threshold: float = 1.5, motion_max: int = 2, accel_window: int = 3,
) -> tuple[bool, dict]:
    """Rule C — Returns (fires_synera_state, verdict). Port of rule_c_trajectory.py."""
    verdict = trajectory_verdict(buffer, vital, baseline_mean, baseline_std, accel_window)
    if motion_score > motion_max:
        return False, verdict
    if verdict["deviation_sigma"] < sigma_threshold:
        return False, verdict
    if not verdict["is_sustained_acceleration"]:
        return False, verdict
    return True, verdict
