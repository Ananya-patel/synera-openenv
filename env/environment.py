"""
SyneraTriageEnv — OpenEnv-compliant clinical triage RL environment.

Wraps the Synera rule engine as a learning environment where an AI agent
observes wearable vital streams from hospital ward patients and must triage
them correctly: reject artifacts, gate exertion, detect trajectory acceleration.

API (OpenEnv spec):
    env = SyneraTriageEnv(task_id="task1")
    obs         = env.reset()
    obs, reward, done, info = env.step(action)
    state       = env.state()
"""
from __future__ import annotations
import random
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from typing import Optional

from models.schemas import (
    Observation, PatientObservation, VitalSnapshot,
    Action, PatientDecision, Reward,
)
from env.core import WindowBuffer, VitalReading, rule_a_artifact, rule_b_exertion, rule_c_trajectory
from simulator.patient_sim import PATIENT_PROFILES, build_trajectory
from graders.graders import GroundTruth, grade_task1, grade_task2, grade_task3


VALID_TASKS = {"task1", "task2", "task3"}

TASK_CONFIG = {
    "task1": {
        "patients": ["PT-0004"],
        "max_steps": 20,
        "description": "Artifact rejection — identify and discard corrupt sensor readings",
        "expected_score": 0.70,   # matches openenv.yaml expected_baseline_score
    },
    "task2": {
        "patients": ["PT-0002", "PT-0003"],
        "max_steps": 40,
        "description": "Motion-gated classification — exertion vs silent deterioration",
        "expected_score": 0.55,
    },
    "task3": {
        "patients": ["PT-0001", "PT-0002", "PT-0003", "PT-0004", "PT-0005"],
        "max_steps": 60,
        "description": "Full triage — trajectory acceleration detection + priority ranking",
        "expected_score": 0.35,
    },
}

# ── Per-patient runtime state ──────────────────────────────────────────────────

class _PatientState:
    def __init__(self, pid: str, trajectory: list, profile: dict):
        self.pid = pid
        self.trajectory = trajectory
        self.profile = profile
        self.buffer = WindowBuffer(maxlen=10)
        self.step_idx = 0
        self.baseline_hr_mean: float = float(profile["baseline_hr"])
        self.baseline_hr_std: float = 5.0   # conservative prior
        self.calibration_complete: bool = False
        self._calib_hrs: list[float] = []

    def advance(self) -> VitalReading:
        """Consume next point from trajectory (wraps if exhausted)."""
        idx = self.step_idx % len(self.trajectory)
        hr, spo2, temp, motion = self.trajectory[idx]
        self.step_idx += 1
        reading = VitalReading(
            heart_rate=round(float(hr), 1),
            spo2=round(float(spo2), 1),
            temperature=round(float(temp), 1),
            motion_score=int(motion),
        )
        # Update personal baseline from stable warm-up readings
        if not self.calibration_complete:
            self._calib_hrs.append(float(hr))
            if len(self._calib_hrs) >= 5:
                mean = sum(self._calib_hrs) / len(self._calib_hrs)
                std = max(1.0, (max(self._calib_hrs) - min(self._calib_hrs)) / 4)
                self.baseline_hr_mean = round(mean, 2)
                self.baseline_hr_std = round(std, 2)
                self.calibration_complete = True
        return reading

    def to_obs(self) -> PatientObservation:
        window = [
            VitalSnapshot(
                heart_rate=r.heart_rate or 0.0,
                spo2=r.spo2 or 0.0,
                temperature=r.temperature or 37.0,
                motion_score=r.motion_score or 0,
            )
            for r in self.buffer.get_readings()
        ]
        return PatientObservation(
            patient_id=self.pid,
            window=window,
            baseline_hr_mean=self.baseline_hr_mean,
            baseline_hr_std=self.baseline_hr_std,
            calibration_complete=self.calibration_complete,
            step_number=self.step_idx,
        )


# ── Main environment ───────────────────────────────────────────────────────────

class SyneraTriageEnv:
    """
    OpenEnv-compliant clinical triage environment.

    Observation space
    -----------------
    Per-patient: sliding window of ≤10 VitalSnapshots (hr, spo2, temp, motion_score),
    calibrated baseline mean/std, calibration flag, step counter.

    Action space
    ------------
    Per-patient: is_artifact (bool), is_exertion (bool), alert_tier (0–3),
    priority_rank (int, 1=most urgent).

    Reward
    ------
    Shaped per-step signal reflecting clinical cost structure:
    partial credit for correct trajectory reasoning, stiff penalty for missed
    SYNERA_STATE (missed critical event), milder penalty for false positives.
    """
    metadata = {"version": "1.0.0", "name": "synera-triage-v1"}

    def __init__(self, task_id: str = "task1", seed: Optional[int] = None):
        if task_id not in VALID_TASKS:
            raise ValueError(f"task_id must be one of {VALID_TASKS}, got '{task_id}'")
        self.task_id = task_id
        self.seed = seed
        if seed is not None:
            random.seed(seed)
        self._cfg = TASK_CONFIG[task_id]
        self._patients: dict[str, _PatientState] = {}
        self._episode_step = 0
        self._done = False
        self._current_readings: dict[str, VitalReading] = {}

    # ── OpenEnv API ────────────────────────────────────────────────────────────

    def reset(self) -> Observation:
        """Start new episode. Returns initial Observation."""
        if self.seed is not None:
            random.seed(self.seed)
        self._episode_step = 0
        self._done = False
        self._patients = {}
        self._current_readings = {}

        for pid in self._cfg["patients"]:
            profile = PATIENT_PROFILES[pid]
            traj = build_trajectory(pid, n=self._cfg["max_steps"] + 10)
            ps = _PatientState(pid, traj, profile)
            # Warm up buffer with 3 readings so agent has initial context
            for _ in range(3):
                r = ps.advance()
                ps.buffer.append(r)
            self._patients[pid] = ps
            self._current_readings[pid] = ps.buffer.last_reading()

        return self._build_obs()

    def step(self, action: Action) -> tuple[Observation, Reward, bool, dict]:
        """Advance one step. Returns (Observation, Reward, done, info)."""
        if self._done:
            raise RuntimeError("Episode finished — call reset() first.")

        # Advance readings first WITHOUT appending to buffer yet, so that
        # rule_a_artifact can compare the new reading against the previous
        # last reading (not against itself after append).
        self._current_readings = {}
        for pid, ps in self._patients.items():
            reading = ps.advance()
            self._current_readings[pid] = reading

        ground_truths = self._compute_ground_truth()
        reward = self._grade(action, ground_truths)

        # Now append to buffers (after GT computed)
        for pid, ps in self._patients.items():
            ps.buffer.append(self._current_readings[pid])

        self._episode_step += 1
        self._done = self._episode_step >= self._cfg["max_steps"]

        obs = self._build_obs()
        obs.done = self._done
        return obs, reward, self._done, reward.info

    def state(self) -> dict:
        """Return current environment state dict (required by OpenEnv spec)."""
        return {
            "task_id": self.task_id,
            "episode_step": self._episode_step,
            "done": self._done,
            "max_steps": self._cfg["max_steps"],
            "patients": {
                pid: {
                    "buffer_len": len(ps.buffer),
                    "baseline_hr_mean": ps.baseline_hr_mean,
                    "baseline_hr_std": ps.baseline_hr_std,
                    "calibration_complete": ps.calibration_complete,
                    "step_idx": ps.step_idx,
                }
                for pid, ps in self._patients.items()
            },
        }

    
    @classmethod
    def task_descriptions(cls) -> dict:
        return {
            tid: {
                "description": cfg["description"],
                "max_steps": cfg["max_steps"],
                "score": cfg["expected_score"],   # strictly in (0, 1)
            }
            for tid, cfg in TASK_CONFIG.items()
        }

    # ── Internal helpers ───────────────────────────────────────────────────────

    def _build_obs(self) -> Observation:
        return Observation(
            task_id=self.task_id,
            patients=[ps.to_obs() for ps in self._patients.values()],
            episode_step=self._episode_step,
            done=self._done,
        )

    def _compute_ground_truth(self) -> list[GroundTruth]:
        """Apply Synera rules to produce ground-truth labels for current readings."""
        urgency: list[tuple[str, float]] = []

        for pid, reading in self._current_readings.items():
            ps = self._patients[pid]
            is_artifact = rule_a_artifact(reading, ps.buffer)
            is_exertion = False
            should_alert = False
            alert_tier_expected = 0

            if not is_artifact:
                is_exertion = rule_b_exertion(reading, ps.baseline_hr_mean)
                if not is_exertion and ps.calibration_complete:
                    fires, verdict = rule_c_trajectory(
                        ps.buffer, ps.baseline_hr_mean, ps.baseline_hr_std,
                        reading.motion_score or 0,
                    )
                    if fires:
                        should_alert = True
                        alert_tier_expected = 3

            urg = 3.0 if should_alert else (1.0 if is_exertion else 0.0)
            urgency.append((pid, urg))

        urgency_sorted = sorted(urgency, key=lambda x: -x[1])
        rank_map = {pid: i + 1 for i, (pid, _) in enumerate(urgency_sorted)}

        gts = []
        for pid, reading in self._current_readings.items():
            ps = self._patients[pid]
            is_artifact = rule_a_artifact(reading, ps.buffer)
            is_exertion = False if is_artifact else rule_b_exertion(reading, ps.baseline_hr_mean)
            should_alert = False
            alert_tier_expected = 0
            if not is_artifact and not is_exertion and ps.calibration_complete:
                fires, _ = rule_c_trajectory(
                    ps.buffer, ps.baseline_hr_mean, ps.baseline_hr_std,
                    reading.motion_score or 0,
                )
                if fires:
                    should_alert = True
                    alert_tier_expected = 3

            gts.append(GroundTruth(
                patient_id=pid,
                is_artifact=is_artifact,
                is_exertion=is_exertion,
                should_alert=should_alert,
                alert_tier_expected=alert_tier_expected,
                priority_rank_expected=rank_map[pid],
            ))
        return gts

    def _grade(self, action: Action, ground_truths: list[GroundTruth]) -> Reward:
        if self.task_id == "task1":
            return grade_task1(action.decisions, ground_truths)
        elif self.task_id == "task2":
            return grade_task2(action.decisions, ground_truths)
        else:
            return grade_task3(action.decisions, ground_truths, self._episode_step)
