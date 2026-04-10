"""
simulate_local.py — Full local simulation of all 3 Synera tasks.

Runs WITHOUT an LLM or API keys. Uses a deterministic rule-following agent
that applies the Synera rules directly (perfect oracle agent) to verify the
environment produces correct reward signals across all tasks.

Output: coloured terminal simulation with per-step reward breakdown + final scoreboard.
"""
from __future__ import annotations
import sys, os, json
# Force UTF-8 output on Windows
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout = open(sys.stdout.fileno(), mode="w", encoding="utf-8", buffering=1)
    sys.stderr = open(sys.stderr.fileno(), mode="w", encoding="utf-8", buffering=1)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from env.environment import SyneraTriageEnv
from env.core import WindowBuffer, VitalReading, rule_a_artifact, rule_b_exertion, rule_c_trajectory
from models.schemas import Action, PatientDecision
from simulator.patient_sim import PATIENT_PROFILES

# Oracle maintains its own per-patient WindowBuffers to mirror the env's state
# (the agent can reconstruct buffers from the observation window each step)
_oracle_buffers: dict[str, WindowBuffer] = {}
_oracle_initialized: bool = False


# ── ANSI colours ──────────────────────────────────────────────────────────────
RED    = "\033[91m"
GREEN  = "\033[92m"
YELLOW = "\033[93m"
CYAN   = "\033[96m"
BOLD   = "\033[1m"
DIM    = "\033[2m"
RESET  = "\033[0m"

def col(text, c): return f"{c}{text}{RESET}"


# ── Rule-following oracle agent ────────────────────────────────────────────────

def oracle_agent(obs_dict: dict) -> Action:
    """
    Applies Synera Rules A/B/C using the SAME functions as env/core.py.
    Reconstructs WindowBuffers from the observation window so rule_c_trajectory
    gets the full sliding-window trajectory data it needs.

    Two-pass: compute flags + urgency first, then assign ranks.
    """
    global _oracle_buffers
    interim = []  # (pid, is_artifact, is_exertion, alert_tier, urgency)

    for p in obs_dict["patients"]:
        pid = p["patient_id"]
        window = p["window"]

        # Rebuild buffer from observation window (oldest first)
        buf = WindowBuffer(maxlen=10)
        for snap in window:
            buf.append(VitalReading(
                heart_rate=snap["heart_rate"],
                spo2=snap["spo2"],
                temperature=snap["temperature"],
                motion_score=snap["motion_score"],
            ))

        if not window:
            interim.append((pid, False, False, 0, 0.0))
            continue

        # The LATEST reading in the window is what was just appended
        latest_snap = window[-1]
        latest = VitalReading(
            heart_rate=latest_snap["heart_rate"],
            spo2=latest_snap["spo2"],
            temperature=latest_snap["temperature"],
            motion_score=latest_snap["motion_score"],
        )

        # Build a "previous state" buffer (without latest) for Rule A
        prev_buf = WindowBuffer(maxlen=10)
        for snap in window[:-1]:
            prev_buf.append(VitalReading(
                heart_rate=snap["heart_rate"],
                spo2=snap["spo2"],
                temperature=snap["temperature"],
                motion_score=snap["motion_score"],
            ))

        # Rule A — artifact: compare latest against buffer WITHOUT the latest
        is_artifact = rule_a_artifact(latest, prev_buf)

        # Rule B — exertion gating
        baseline_mean = p["baseline_hr_mean"]
        is_exertion = (
            not is_artifact
            and rule_b_exertion(latest, baseline_mean)
        )

        # Rule C — trajectory / SYNERA_STATE (full sliding window via core)
        alert_tier = 0
        if not is_artifact and not is_exertion and p["calibration_complete"]:
            baseline_std = max(p["baseline_hr_std"], 1.0)
            fires, _ = rule_c_trajectory(
                buf, baseline_mean, baseline_std, latest_snap["motion_score"]
            )
            if fires:
                alert_tier = 3
            else:
                # Watch tier: sigma >= 1.0 but not critical
                hrs = buf.get_heart_rates()
                if hrs:
                    dev = abs(hrs[-1] - baseline_mean) / baseline_std
                    if dev >= 1.0:
                        alert_tier = 1

        urg = 3.0 if alert_tier == 3 else (1.5 if is_exertion else (0.5 if alert_tier == 1 else 0.0))
        interim.append((pid, is_artifact, is_exertion, alert_tier, urg))

    # Rank by urgency (two patients with same urgency: stable FIFO order)
    sorted_by_urg = sorted(enumerate(interim), key=lambda x: -x[1][4])
    rank_map = {row[0]: i + 1 for i, (_, row) in enumerate(sorted_by_urg)}

    decisions = [
        PatientDecision(
            patient_id=pid,
            is_artifact=is_artifact,
            is_exertion=is_exertion,
            alert_tier=alert_tier,
            priority_rank=rank_map[pid],
        )
        for pid, is_artifact, is_exertion, alert_tier, _ in interim
    ]
    return Action(decisions=decisions)


# ── Naive agent (always outputs safe-but-dumb no-op) ──────────────────────────

def naive_agent(obs_dict: dict) -> Action:
    """Never fires any alert — null strategy."""
    return Action(decisions=[
        PatientDecision(
            patient_id=p["patient_id"],
            is_artifact=False, is_exertion=False,
            alert_tier=0, priority_rank=i + 1,
        )
        for i, p in enumerate(obs_dict["patients"])
    ])


# ── Pretty-print helpers ───────────────────────────────────────────────────────

def format_decisions(decisions):
    parts = []
    for d in decisions:
        flags = []
        if d.is_artifact: flags.append(col("ARTIFACT", RED))
        if d.is_exertion:  flags.append(col("EXERTION", YELLOW))
        tier_str = {0: DIM+"tier:0"+RESET, 1: "tier:1", 2: col("tier:2", YELLOW), 3: col("tier:3 !!ALERT!!", RED)}
        parts.append(f"  {col(d.patient_id, CYAN)} rank:{d.priority_rank} {tier_str.get(d.alert_tier, 'tier:?')} {' '.join(flags) if flags else DIM+'(stable)'+RESET}")
    return "\n".join(parts)


def format_reward(reward_dict, breakdown_dict, info_dict):
    total = reward_dict["total"]
    colour = GREEN if total > 0 else (RED if total < 0 else DIM)
    line = f"  reward={col(f'{total:+.4f}', BOLD+colour)}"
    # breakdown highlights
    for k, v in breakdown_dict.items():
        if v != 0.0:
            c = GREEN if v > 0 else RED
            line += f"  {k}={col(f'{v:+.3f}', c)}"
    # key info
    for k in ("tp", "fp", "fn", "f1", "exertion_correct", "missed", "has_critical_miss", "rank_matches"):
        if k in info_dict:
            line += f"  {DIM}{k}={info_dict[k]}{RESET}"
    return line


# ── Run one task ───────────────────────────────────────────────────────────────

def run_task(task_id: str, agent_fn, label: str, verbose: bool = True) -> dict:
    env = SyneraTriageEnv(task_id=task_id, seed=42)
    obs = env.reset()

    total_reward = 0.0
    rewards = []
    step = 0

    if verbose:
        print(f"\n{BOLD}{'='*70}{RESET}")
        print(f"{BOLD}  TASK: {task_id.upper()} -- {label}{RESET}")
        patients_str = ", ".join(f"{pid} ({PATIENT_PROFILES[pid]['name']})" for pid in env._cfg["patients"])
        print(f"  Patients: {patients_str}")
        print(f"  Max steps: {env._cfg['max_steps']}")
        print(f"{'='*70}{RESET}")

    done = False
    while not done:
        obs_dict = obs.model_dump()
        action = agent_fn(obs_dict)
        obs, reward, done, info = env.step(action)

        reward_dict = reward.model_dump()
        breakdown = reward_dict["breakdown"]
        step += 1
        total_reward += reward_dict["total"]
        rewards.append(reward_dict["total"])

        if verbose and (step <= 5 or done or any(v != 0 for v in breakdown.values())):
            print(f"\n  {DIM}Step {step:02d}{RESET}")
            print(format_decisions(action.decisions))
            print(format_reward(reward_dict, breakdown, reward_dict["info"]))
            if step == 5 and env._cfg["max_steps"] > 10:
                print(f"  {DIM}... (showing only notable steps) ...{RESET}")

    avg = total_reward / max(1, step)
    # FIXED
    # AFTER
    task_score = round(max(0.0001, min(0.9999, avg)), 4)

    if verbose:
        colour = GREEN if task_score >= 0.7 else (YELLOW if task_score >= 0.4 else RED)
        print(f"\n  {BOLD}RESULT  steps={step}  avg_reward={avg:.4f}  task_score={col(f'{task_score:.4f}', colour+BOLD)}{RESET}")
        print(f"  reward range: [{min(rewards):.4f}, {max(rewards):.4f}]")

    return {"task_id": task_id, "steps": step, "avg_reward": round(avg,4), "task_score": task_score}


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    print(f"\n{BOLD}{col('SYNERA TRIAGE OPENENV — LOCAL SIMULATION', CYAN)}{RESET}")
    print(f"{DIM}Oracle agent (rule-following) vs Naive agent (null strategy){RESET}")

    results = {}

    for task_id in ["task1", "task2", "task3"]:
        r = run_task(task_id, oracle_agent, "Oracle Agent", verbose=True)
        results[task_id] = r

    print(f"\n{BOLD}{'='*70}{RESET}")
    print(f"{BOLD}  FINAL SCOREBOARD -- ORACLE AGENT{RESET}")
    print(f"{'='*70}")
    overall = 0.0
    for tid in ["task1", "task2", "task3"]:
        s = results[tid]["task_score"]
        c = GREEN if s >= 0.7 else (YELLOW if s >= 0.4 else RED)
        print(f"  {tid}: {col(f'{s:.4f}', BOLD+c)}")
        overall += s
    overall /= 3
    oc = GREEN if overall >= 0.7 else (YELLOW if overall >= 0.4 else RED)
    print(f"  {'-'*30}")
    print(f"  overall: {col(f'{overall:.4f}', BOLD+oc)}")
    print(f"{'='*70}\n")

    # Compare with naive
    print(f"{BOLD}  NAIVE AGENT COMPARISON{RESET}")
    print(f"{'-'*50}")
    naive_results = {}
    for task_id in ["task1", "task2", "task3"]:
        r = run_task(task_id, naive_agent, "Naive Agent", verbose=False)
        naive_results[task_id] = r
        s = r["task_score"]
        c = GREEN if s >= 0.7 else (YELLOW if s >= 0.4 else RED)
        oracle_s = results[task_id]["task_score"]
        delta = oracle_s - s
        dc = GREEN if delta >= 0 else RED
        print(f"  {task_id}: naive={col(f'{s:.4f}', c)}  oracle={col(f'{oracle_s:.4f}', GREEN)}  delta={col(f'{delta:+.4f}', dc)}")

    print()


if __name__ == "__main__":
    main()
