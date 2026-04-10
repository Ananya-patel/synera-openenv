---
title: Synera Triage OpenEnv
colorFrom: blue
colorTo: red
sdk: docker
pinned: false
tags:
  - openenv , Meta , Huggingface, Scaler, PyTorch
---

# Synera Triage OpenEnv

A clinical triage RL environment for the OpenEnv Hackathon 2026.


# Synera Clinical Triage Environment

> **Meta × Scaler × Hugging Face × OpenEnv Hackathon 2026**  
> Team Vampire — Harsh Singhal · Anchita Jain · Ananya Patel

An OpenEnv-compliant reinforcement learning environment where an AI agent monitors
hospital ward patients through wearable vital sensors and must triage them in real time —
rejecting sensor artifacts, gating motion-induced false alarms, and detecting silent
physiological deterioration before it becomes a crisis.

---

## Why Hospital Triage?

Nurses monitoring a multi-patient ward face three cognitively distinct challenges every shift:

| Challenge | What it looks like | What Synera calls it |
|---|---|---|
| Sensor noise | Loose PPG electrode → sudden 228 BPM spike | **Artifact** |
| False elevation | Patient walks to bathroom → HR 116 BPM, motion=8 | **Exertion** |
| True deterioration | Post-op patient: 82→103 BPM over 8 readings, accelerating, at rest | **SYNERA_STATE** |

Most RL benchmarks use games or synthetic tasks. This environment uses real clinical
reasoning patterns — the same three-rule engine used in ArogyaLink, a deployed hospital
monitoring system. Every reward, penalty, and trajectory is grounded in actual clinical cost
structure. No other team at this hackathon is likely to bring this domain.

---

## Architecture

```
synera-openenv/
├── app.py                    FastAPI server  (/reset  /step  /state  /health  /tasks)
├── inference.py              Chain-of-thought LLM baseline agent
├── openenv.yaml              OpenEnv metadata spec
├── Dockerfile                HF Spaces container  (EXPOSE 7860)
├── requirements.txt
│
├── env/
│   ├── core.py               Synera rule engine — stdlib-only port of ArogyaLink
│   │                         (WindowBuffer · VitalReading · Rule A/B/C · trajectory math)
│   └── environment.py        SyneraTriageEnv — reset() / step() / state()
│
├── models/
│   └── schemas.py            Pydantic models: Observation · Action · Reward
│
├── graders/
│   └── graders.py            Deterministic graders for all 3 tasks
│
└── simulator/
    └── patient_sim.py        5 patient archetypes + 4 trajectory generators
```

---

## The Three Synera Rules

These are the clinical reasoning primitives the agent must learn. They are **never handed
to the agent directly** — the agent must infer them from trajectory patterns:

```
Rule A  (Artifact)       |hr_t − hr_{t−1}| ≥ 40 BPM
                          → physiologically impossible in one 5-second interval
                          → discard reading, do not propagate to trajectory engine

Rule B  (Exertion)       motion_score > 4  AND  hr > baseline_mean + 15 BPM
                          → elevation explained by physical activity
                          → log as EXERTION_LOGGED, do not alert

Rule C  (SYNERA_STATE)   NOT artifact  AND  NOT exertion  AND  calibration_complete
                          AND  deviation > 1.5σ from personal baseline
                          AND  sustained second-derivative acceleration
                          → fire alert_tier = 3 (critical)
```

The sliding window holds the last 10 readings (50 seconds). Rule C requires analysing the
**shape** of the HR trend — not just the current value — making it the hardest task and
the one that genuinely challenges frontier LLMs.

---

## Patient Profiles

| ID | Name | Age | Conditions | Trajectory | Expected Outcome |
|---|---|---|---|---|---|
| PT-0001 | Rajesh Kumar | 62 | Hypertension, T2DM | Flat noise | STABLE — never alerts |
| PT-0002 | Priya Sharma | 34 | Post-op Day 2, T2DM | Exponential acceleration | SYNERA_STATE at ~103 BPM |
| PT-0003 | Arjun Mehta | 28 | None | Exertion spike (motion=8) | EXERTION_LOGGED — no alert |
| PT-0004 | Fatima Begum | 71 | COPD, AFib | Glitch at index 3 (228 BPM) | ARTIFACT discarded |
| PT-0005 | Suresh Patel | 55 | Hypertension, CKD-3 | Slow linear drift | WATCH tier — no critical alert |

---

## Tasks

### Task 1 — Artifact Rejection `[Easy · 1 patient · 20 steps]`

PT-0004 (Fatima Begum) has a loose PPG sensor. One reading spikes to 228 BPM.
The agent must flag `is_artifact=true` for that exact reading without discarding
the valid readings before and after it.

**Grader:** F1 + accuracy with asymmetric penalties.

| Event | Reward |
|---|---|
| Correct detection (TP) | up to +1.0 (F1-weighted) |
| Correct silence (TN) | up to +1.0 (accuracy-weighted) |
| False positive — discarded valid reading | −0.20 |
| **False negative — missed artifact** | **−0.80** |

The FN penalty is 4× the FP penalty because a missed artifact corrupts downstream
trajectory analysis — a compounding error.

---

### Task 2 — Motion-Gated Classification `[Medium · 2 patients · 40 steps]`

PT-0002 (Priya, post-op) is silently deteriorating via exponential HR acceleration.
PT-0003 (Arjun, healthy) is just walking — elevated HR with `motion_score=8`.
The agent must tell these apart without firing alert fatigue on PT-0003.

**Grader:** Per-patient, per-step shaped rewards.

| Event | Reward |
|---|---|
| Correct exertion gate (no alert on PT-0003) | +0.30 / n |
| Correct deterioration alert (PT-0002 at tier 3) | +0.50 / n |
| Correct silence on stable patient | +0.10 / n |
| False alarm on exertion patient | −0.30 / n |
| False alarm on stable patient | −0.20 / n |
| **Missed real deterioration** | **−0.50 / n** |

---

### Task 3 — Trajectory Triage + Priority Ranking `[Hard · 5 patients · 60 steps]`

All five patients are active. The agent must simultaneously:
1. Apply Rule A/B/C per-patient across their individual sliding windows
2. Detect when PT-0002 crosses the SYNERA_STATE threshold
3. Correctly rank all 5 patients by urgency (1 = most critical)

**Grader:** Normalised against the maximum achievable bonus from the current step's
ground truth — a perfect agent scores 1.0 every step regardless of how many patients
are in each state class.

| Event | Reward |
|---|---|
| Motion gate correct (exertion patient, no critical alert) | +0.05 / patient |
| SYNERA_STATE detected (tier 3, accelerating patient) | +0.30 / patient |
| Priority ranking correct across all N patients | +0.50 |
| False positive (tier ≥ 2 on stable patient) | −0.20 |
| **Missed SYNERA_STATE** | **−0.50, zeroes priority bonus** |

The missed-critical penalty **zeroes the priority bonus** for the same step — even a
perfect rank order cannot compensate for letting a deteriorating patient go unnoticed.

---
## Why This Environment is Challenging for LLM Agents

- Requires temporal reasoning over sliding windows (not single-step classification)
- Second-derivative trend detection (acceleration, not just thresholding)
- Multi-patient parallel reasoning (Task 3)
- Confound handling (artifact vs exertion vs deterioration)
- Asymmetric cost-sensitive decision making

This prevents shortcut policies and requires genuine reasoning.


## Observation & Action Schema

### Observation

```json
{
  "task_id": "task3",
  "episode_step": 12,
  "done": false,
  "patients": [
    {
      "patient_id": "PT-0002",
      "window": [
        {"heart_rate": 84.1, "spo2": 97.0, "temperature": 37.1, "motion_score": 1},
        {"heart_rate": 85.8, "spo2": 96.9, "temperature": 37.1, "motion_score": 1},
        {"heart_rate": 88.2, "spo2": 96.7, "temperature": 37.2, "motion_score": 1}
      ],
      "baseline_hr_mean": 82.3,
      "baseline_hr_std": 2.1,
      "calibration_complete": true,
      "step_number": 15
    }
  ]
}
```

### Action

```json
{
  "decisions": [
    {
      "patient_id": "PT-0002",
      "is_artifact": false,
      "is_exertion": false,
      "alert_tier": 3,
      "priority_rank": 1
    }
  ]
}
```

`alert_tier`: 0 = no alert · 1 = watch · 2 = advisory · 3 = critical / SYNERA_STATE  
`priority_rank`: 1 = most urgent, N = least urgent

---

## Baseline Agent — Chain-of-Thought LLM Reasoner

The `inference.py` baseline is **not a rule-following oracle**. It is a chain-of-thought
clinical reasoning agent that:

1. **Receives a clinical briefing**, not raw JSON — each patient's window is rendered
   with sigma-deviation annotations, HR trend labels, and delta arrays
2. **Reasons inside `<reasoning>` tags** before producing JSON — judges can inspect the
   model's actual clinical thinking in every `[STEP]` log
3. **Derives rules from context**, not from an explicit lookup table — this is what
   Phase 2 agentic evaluation (Nemotron 3 Super) tests

Sample `[STEP]` log:
```json
{
  "event": "STEP",
  "task_id": "task3",
  "step": 18,
  "reward": 0.9216,
  "reasoning_snippet": "PT-0002 shows HR rising from 82 to 103 BPM across 8 readings
    while at rest (motion=1). Deltas: [1.2, 1.5, 1.9, 2.4, 3.1] — each interval larger
    than the last, indicating exponential acceleration. Sigma deviation: 4.2 from
    personal baseline. SYNERA_STATE warranted. PT-0003 motion_score=8 with HR 112:
    classic exertion pattern, no alert. Rank: PT-0002 first..."
}
```

---

## Simulation Results

Verified locally with a deterministic oracle agent (applies same rule engine as the
environment) vs a naive agent (never fires any alert). Seed=42 for reproducibility.

### Oracle Agent Scores

| Task | Steps | Avg Reward | Task Score | Interpretation |
|---|---|---|---|---|
| task1 — Artifact | 20 | +0.850 | **0.9250** | Correctly flags glitch, minor lag penalty |
| task2 — Motion-gate | 40 | +0.233 | **0.6162** | Correctly separates exertion vs deterioration |
| task3 — Full triage | 60 | +0.843 | **0.9216** | Near-perfect 5-patient trajectory detection |
| **Overall** | — | — | **0.8209** | |

### Oracle vs Naive Comparison

| Task | Naive Score | Oracle Score | Delta |
|---|---|---|---|
| task1 | 0.9100 | **0.9250** | +0.015 |
| task2 | 0.4544 | **0.6162** | +0.162 |
| task3 | 0.0914 | **0.9216** | **+0.830** |

The Task 3 delta of **+0.83** is the core proof of concept: the environment is not
solvable by doing nothing. Intelligent trajectory reasoning produces 10× better scores
than the null strategy on the hardest task.

### Per-Step Reward Distribution (Task 3)

The shaped reward function produces meaningful signals throughout the episode:

```
Step  1–2:   +1.00   All patients stable, priority ranking perfect
Step  3:     −0.44   Early trajectory miss (calibration just completing)
Step  4:     +0.55   Trajectory detected + partial ranking correct
Step 22:     +1.00   PT-0002 SYNERA_STATE + perfect 5-patient rank
Step 36:     +1.00   Two simultaneous SYNERA_STATE events, all ranked correctly
```

---

## Key Engineering Decisions

### Sliding Window Before Ground Truth (Critical Bug Fix)

The original environment appended new readings to the buffer **before** computing
ground truth, causing `rule_a_artifact` to compare a reading against itself (delta = 0
always). Ground truth never detected any artifact — making Task 1 effectively ungraded.

**Fix:** New readings are generated and ground truth is computed first. Buffer append
happens afterwards. This is now the correct causal order.

### Asymmetric Reward Design

Clinical cost structure is asymmetric — the penalties are not symmetric with the rewards:

```
Task 1:  FP (discarded valid) = −0.20    FN (missed artifact)  = −0.80
Task 2:  FP (false alarm)     = −0.20    FN (missed alert)      = −0.50
Task 3:  FP (alert fatigue)   = −0.20    FN (SYNERA_STATE miss) = −0.50 + zeros priority
```

This mirrors real hospital cost: missing a deteriorating patient has worse consequences
than generating a spurious alert.

### Normalised Task 3 Grader

Task 3 reward is normalised against the maximum achievable bonus from the **current
step's actual ground truth composition**, not a theoretical ceiling. On steps where
no patient is in SYNERA_STATE, the maximum achievable is just the priority signal (0.50).
A perfect agent scores 1.0 regardless of which patients happen to be alerting.

---

## Known Failure Modes

- Early calibration phase may cause trajectory misclassification
- High noise + motion overlap can confuse Rule A vs B
- Gradual drift patients (PT-0005) may be under-prioritized

Future work includes adaptive baselines and multimodal fusion.

## API Reference

### Endpoints

| Method | Path | Body | Response |
|---|---|---|---|
| `GET` | `/health` | — | `{"status": "ok"}` |
| `GET` | `/tasks` | — | Task descriptions dict |
| `POST` | `/reset` | `{"task_id": "task1", "seed": 42}` | `Observation` |
| `POST` | `/step` | `Action` | `{observation, reward, done, info}` |
| `GET` | `/state` | — | Environment state dict |

### Environment Variables

| Variable | Required | Default | Description |
|---|---|---|---|
| `API_BASE_URL` | Yes | — | LLM API endpoint (inference.py) |
| `MODEL_NAME` | Yes | — | LLM model identifier |
| `HF_TOKEN` | Yes | — | API / HF authentication token |
| `TASK_ID` | No | `task1` | Default task on server startup |
| `SEED` | No | `42` | Random seed for reproducibility |
| `PORT` | No | `7860` | HTTP port |

---

## Quick Start

### Run locally

```bash
pip install -r requirements.txt
python app.py
```

### Test the environment

```bash
# Health check
curl http://localhost:7860/health

# Start Task 3 episode
curl -s -X POST http://localhost:7860/reset \
  -H "Content-Type: application/json" \
  -d '{"task_id": "task3", "seed": 42}' | python -m json.tool

# Submit an action
curl -s -X POST http://localhost:7860/step \
  -H "Content-Type: application/json" \
  -d '{"decisions": [{"patient_id":"PT-0002","is_artifact":false,"is_exertion":false,"alert_tier":3,"priority_rank":1}]}'
```

### Run the LLM baseline agent

```bash
export API_BASE_URL=https://api.openai.com/v1
export MODEL_NAME=gpt-4o-mini
export HF_TOKEN=your_token_here
export ENV_BASE_URL=http://localhost:7860
python inference.py
```

Expected output format:
```
{"event": "START", "task_id": "task1", "agent": "chain-of-thought clinical reasoner", ...}
{"event": "STEP", "step": 1, "reward": 1.0, "reasoning_snippet": "...", ...}
{"event": "END", "task_score": 0.9250, "reasoning_rate": 1.0, ...}
```

### Docker / HF Spaces deployment

```bash
docker build -t synera-triage .
docker run -p 7860:7860 \
  -e API_BASE_URL=https://api.openai.com/v1 \
  -e MODEL_NAME=gpt-4o-mini \
  -e HF_TOKEN=your_token \
  synera-triage
```

For Hugging Face Spaces: create a **Docker** Space, push this folder, and add
`API_BASE_URL`, `MODEL_NAME`, `HF_TOKEN` in Space Settings → Secrets.

---

## Hackathon Compliance Checklist

| Requirement | Status |
|---|---|
| Real-world task (not games/toys) | ✅ Hospital clinical triage |
| Typed Pydantic Observation / Action / Reward | ✅ All 5 models in `models/schemas.py` |
| `reset()` / `step()` / `state()` API | ✅ `SyneraTriageEnv` |
| `openenv.yaml` with full metadata | ✅ Tasks, obs/action space, reward description |
| Minimum 3 tasks, easy → hard | ✅ task1 / task2 / task3 |
| Rewards in [−1.0, 1.0] | ✅ Clamped in all graders |
| Shaped reward (not binary end-of-episode) | ✅ 6 distinct per-step signal components |
| Baseline `inference.py` with OpenAI client | ✅ Chain-of-thought clinical reasoner |
| `temperature=0.0` for reproducibility | ✅ |
| `API_BASE_URL` / `MODEL_NAME` / `HF_TOKEN` from env vars | ✅ |
| `[START]` / `[STEP]` / `[END]` stdout logs | ✅ With `reasoning_snippet` per step |
| Dockerfile with `EXPOSE 7860` + `HEALTHCHECK` | ✅ |
| All `__init__.py` package files present | ✅ |
| Ground truth computed before buffer append | ✅ (critical bug fixed) |

---

## Authors

**Harsh Singhal · Anchita Jain · Ananya Patel**  
Team Vampire — Meta × Scaler × Hugging Face × OpenEnv Hackathon 2026

---

## License

MIT
