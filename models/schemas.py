"""
OpenEnv typed models for the Synera Clinical Triage Environment.

Required by the OpenEnv spec:
  - Typed Observation, Action, Reward Pydantic models
  - step(action) → (Observation, Reward, done, info)
  - reset()      → Observation
  - state()      → dict
"""
from __future__ import annotations
from typing import List
from pydantic import BaseModel, Field, model_validator


# ── Per-reading snapshot (what the agent sees) ─────────────────────────────────

class VitalSnapshot(BaseModel):
    heart_rate: float = Field(..., description="Heart rate BPM")
    spo2: float = Field(..., description="SpO2 %")
    temperature: float = Field(..., description="Temperature °C")
    motion_score: int = Field(..., ge=0, le=10, description="IMU motion 0=rest 10=vigorous")


# ── Observation ────────────────────────────────────────────────────────────────

class PatientObservation(BaseModel):
    patient_id: str
    window: List[VitalSnapshot] = Field(..., description="Up to 10 recent readings, oldest first")
    baseline_hr_mean: float = Field(..., description="Calibrated personal HR baseline mean")
    baseline_hr_std: float = Field(..., description="Calibrated personal HR baseline std")
    calibration_complete: bool = Field(..., description="True once baseline is established")
    step_number: int


class Observation(BaseModel):
    task_id: str
    patients: List[PatientObservation]
    episode_step: int
    done: bool = False


# ── Action ─────────────────────────────────────────────────────────────────────

class PatientDecision(BaseModel):
    patient_id: str
    is_artifact: bool = Field(..., description="True → discard reading as sensor artifact (Rule A)")
    is_exertion: bool = Field(..., description="True → HR explained by motion, no alert (Rule B)")
    alert_tier: int = Field(..., ge=0, le=3,
        description="0=none 1=watch 2=advisory 3=critical/SYNERA_STATE")
    priority_rank: int = Field(..., ge=1,
        description="Urgency rank among all patients; 1 = most urgent")


class Action(BaseModel):
    decisions: List[PatientDecision]


# ── Reward ─────────────────────────────────────────────────────────────────────

class RewardBreakdown(BaseModel):
    artifact_signal: float = 0.0
    exertion_signal: float = 0.0
    trajectory_signal: float = 0.0
    priority_signal: float = 0.0
    false_positive_penalty: float = 0.0
    missed_alert_penalty: float = 0.0




class Reward(BaseModel):
    total: float = Field(..., description="Net step reward, normalized to (0.05, 0.95)")
    score: float = Field(0.5, description="OpenEnv normalized score strictly in (0, 1)")
    breakdown: RewardBreakdown
    info: dict = Field(default_factory=dict)

    @model_validator(mode="after")
    def sync_score(self) -> "Reward":
        # total is already in (0, 1) from graders — clamp only, no remap
        self.score = round(max(0.05, min(0.95, self.total)), 4)
        return self
