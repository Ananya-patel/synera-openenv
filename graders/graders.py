"""
Graders for Tasks 1, 2, 3.

All graders are deterministic and reproducible — pure function of
ground-truth label vs agent action. Scores in [-1.0, 1.0] as per OpenEnv spec.

Ground-truth is computed by the environment applying Synera rules A/B/C
to the actual vital stream — the agent never sees these labels directly.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import List

from models.schemas import PatientDecision, Reward, RewardBreakdown


@dataclass
class GroundTruth:
    patient_id: str
    is_artifact: bool
    is_exertion: bool
    should_alert: bool
    alert_tier_expected: int       # 0-3
    priority_rank_expected: int    # 1-N


# ── Task 1: Artifact Rejection ─────────────────────────────────────────────────

def grade_task1(
    decisions: List[PatientDecision],
    ground_truths: List[GroundTruth],
) -> Reward:
    """
    Task 1 — Artifact vs clean classification.

    Reward table:
      Correct positive (tp):  contributes to F1/accuracy → up to +1.0
      Correct negative (tn):  contributes to accuracy    → up to +1.0
      False positive   (fp):  -0.2  (discarded valid packet — alert fatigue cost)
      False negative   (fn):  -0.8  (missed artifact passed to downstream — high clinical cost)

    The asymmetry (FN >> FP) reflects that a missed artifact corrupts
    downstream trajectory analysis, while a discarded clean reading is merely wasteful.
    """
    gt_map = {g.patient_id: g for g in ground_truths}
    tp = fp = fn = tn = 0

    for d in decisions:
        gt = gt_map.get(d.patient_id)
        if gt is None:
            continue
        if d.is_artifact and gt.is_artifact:
            tp += 1
        elif d.is_artifact and not gt.is_artifact:
            fp += 1
        elif not d.is_artifact and gt.is_artifact:
            fn += 1
        else:
            tn += 1

    precision = tp / (tp + fp) if (tp + fp) > 0 else 1.0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 1.0
    f1        = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    total_p   = tp + fp + fn + tn
    accuracy  = (tp + tn) / total_p if total_p > 0 else 0.0

    artifact_signal_raw = 0.6 * f1 + 0.4 * accuracy
    # Clamp signal to strictly (0, 1) — never exactly 0.0 or 1.0
    artifact_signal = round(max(0.001, min(0.999, artifact_signal_raw)), 3)
    fp_penalty      = round(-0.2 * fp, 3)
    fn_penalty      = round(-0.8 * fn, 3)   # missed artifact: high clinical cost
    total = round(max(0.05, min(0.95, (artifact_signal_raw + fp_penalty + fn_penalty + 1.0) / 2.0)), 4)

    return Reward(
        total=total,
        breakdown=RewardBreakdown(
            artifact_signal=artifact_signal,
            false_positive_penalty=fp_penalty,
            missed_alert_penalty=fn_penalty,
        ),
        info={"tp": tp, "fp": fp, "fn": fn, "tn": tn},
    )


# ── Task 2: Motion-Gated Alert Classification ──────────────────────────────────

def grade_task2(
    decisions: List[PatientDecision],
    ground_truths: List[GroundTruth],
) -> Reward:
    """
    Task 2 — Distinguish exertion from genuine deterioration.

    Reward table (per patient, normalised by n):
      Correct exertion gate (no alert):   +0.30  — agent understood motion context
      Correct deterioration alert:        +0.50  — agent caught genuine decompensation
      Correct silence on stable patient:  +0.10  — agent learned suppression is good
      False alarm on exertion patient:    -0.30  — unnecessary alert (alert fatigue)
      False alarm on stable patient:      -0.20  — unnecessary alert on stable ward
      Missed real deterioration alert:    -0.50  — clinical safety miss
    """
    gt_map = {g.patient_id: g for g in ground_truths}
    n = max(1, len(ground_truths))
    exertion_correct = alert_correct = stable_correct = 0
    false_alarms = missed = stable_fp = 0

    for d in decisions:
        gt = gt_map.get(d.patient_id)
        if gt is None:
            continue
        if gt.is_exertion:
            if d.is_exertion and d.alert_tier < 3:
                exertion_correct += 1          # correctly gated by motion
            elif not d.is_exertion and d.alert_tier >= 2:
                false_alarms += 1              # fired on exertion patient
        elif gt.should_alert:
            if not d.is_exertion and d.alert_tier >= gt.alert_tier_expected:
                alert_correct += 1             # caught genuine deterioration
            else:
                missed += 1                    # missed critical event
        else:
            # Stable patient (no exertion, no alert expected)
            if d.alert_tier >= 2:
                stable_fp += 1                 # false alarm on stable patient
            elif d.alert_tier <= 1:
                stable_correct += 1            # correctly stayed silent

    exertion_signal_raw   = 0.30 * exertion_correct / n
    trajectory_signal_raw = 0.50 * alert_correct / n
    stable_signal_raw     = 0.10 * stable_correct / n
    fp_penalty        = round(-0.30 * false_alarms / n, 3)
    stable_fp_penalty = round(-0.20 * stable_fp / n, 3)
    missed_penalty    = round(-0.50 * missed / n, 3)

    # Clamp signal components to strictly (0, 1) for breakdown display
    exertion_signal   = round(max(0.001, min(0.999, exertion_signal_raw)), 3)
    trajectory_signal = round(max(0.001, min(0.999, trajectory_signal_raw + stable_signal_raw)), 3)

    total = round(max(0.05, min(0.95, (
    exertion_signal_raw + trajectory_signal_raw + stable_signal_raw
    + fp_penalty + stable_fp_penalty + missed_penalty + 1.0) / 2.0)), 4)

    return Reward(
        total=total,
        breakdown=RewardBreakdown(
            exertion_signal=exertion_signal,
            trajectory_signal=trajectory_signal,
            false_positive_penalty=round(fp_penalty + stable_fp_penalty, 3),
            missed_alert_penalty=missed_penalty,
        ),
        info={
            "exertion_correct": exertion_correct,
            "alert_correct": alert_correct,
            "stable_correct": stable_correct,
            "false_alarms": false_alarms,
            "stable_fp": stable_fp,
            "missed": missed,
        },
    )


# ── Task 3: Trajectory + Priority Triage ──────────────────────────────────────

def grade_task3(
    decisions: List[PatientDecision],
    ground_truths: List[GroundTruth],
    episode_step: int,
) -> Reward:
    """
    Task 3 — Trajectory acceleration detection + multi-patient priority ranking.

    Reward components:
      +0.05  per exertion patient correctly NOT given critical alert (motion gate)
      +0.30  correct SYNERA_STATE alert (tier 3 on accelerating patient)
      +0.50  correct priority ordering of ALL patients
      -0.20  false positive: alert_tier >= 2 on stable/artifact patient
      -0.50  missed SYNERA_STATE — HARD deduction, zeroes priority bonus,
             cannot be absorbed by any positive signal

    Normalisation uses the maximum bonus achievable from THIS step's ground truth
    (not a theoretical per-patient ceiling), so a perfect agent always scores ~1.0.
    """
    gt_map = {g.patient_id: g for g in ground_truths}
    n = max(1, len(ground_truths))
    motion_signal = traj_signal = fp_penalty = missed_penalty = 0.0

    for d in decisions:
        gt = gt_map.get(d.patient_id)
        if gt is None:
            continue
        # Motion gate: exertion patient not given critical alert
        if gt.is_exertion and d.alert_tier < 3:
            motion_signal += 0.05
        # Trajectory / SYNERA_STATE detection
        if gt.should_alert and gt.alert_tier_expected == 3:
            if d.alert_tier == 3:
                traj_signal += 0.30
            else:
                missed_penalty -= 0.50          # hard clinical cost
        elif not gt.should_alert and not gt.is_artifact:
            if d.alert_tier >= 2:
                fp_penalty -= 0.20              # alert fatigue cost

    # Priority ranking across all patients
    sorted_gt = sorted(ground_truths, key=lambda g: g.priority_rank_expected)
    sorted_d  = sorted(decisions,     key=lambda d: d.priority_rank)
    rank_matches = sum(
        1 for i, d in enumerate(sorted_d)
        if i < len(sorted_gt) and d.patient_id == sorted_gt[i].patient_id
    )
    # Priority bonus is zeroed when a critical alert was missed (clinical safety)
    has_critical_miss = missed_penalty < 0.0
    priority_signal = 0.0 if has_critical_miss else (0.5 * rank_matches / n)

    # Max achievable bonus from THIS step's actual ground truth composition:
    #   motion_signal max  = 0.05 * #exertion patients
    #   traj_signal max    = 0.30 * #synera patients
    #   priority_signal max = 0.50 (always, from ranking)
    n_exertion = sum(1 for g in ground_truths if g.is_exertion)
    n_synera   = sum(1 for g in ground_truths if g.should_alert and g.alert_tier_expected == 3)
    max_achievable = (0.05 * n_exertion) + (0.30 * n_synera) + 0.50
    if max_achievable <= 0:
        max_achievable = 0.50   # only priority signal possible (all stable)

    bonus_raw  = motion_signal + traj_signal + priority_signal
    bonus_norm = min(1.0, bonus_raw / max_achievable)

    total = round(max(0.05, min(0.95, (bonus_norm + fp_penalty + missed_penalty + 1.0) / 2.0)), 4)

    # Clamp signal breakdown fields to strictly (0, 1) — never exactly 0.0 or 1.0
    traj_display     = round(max(0.001, min(0.999, traj_signal)), 3)
    priority_display = round(max(0.001, min(0.999, priority_signal)), 3)

    return Reward(
        total=total,
        breakdown=RewardBreakdown(
            trajectory_signal=traj_display,
            priority_signal=priority_display,
            false_positive_penalty=round(fp_penalty, 3),
            missed_alert_penalty=round(missed_penalty, 3),
        ),
        info={
            "episode_step": episode_step,
            "rank_matches": rank_matches,
            "n_patients": n,
            "has_critical_miss": has_critical_miss,
            "max_achievable_bonus": round(max_achievable, 3),
        },
    )
