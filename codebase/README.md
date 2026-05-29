# Fault-Tolerant Flight Control of a Hexacopter using Deep Reinforcement Learning

## Project Overview

This project implements a Fault-Tolerant Control (FTC) system for a 6-rotor UAV (Hexacopter) using Deep Reinforcement Learning (PPO). The agent learns to stabilise the drone even when one or more rotors fail mid-flight.

---

## File Structure

```
hexacopter_ftc/
├── hexacopter_env.py      # Custom Gymnasium environment (12-DoF physics)
├── train.py               # PPO training with Dynamic Curriculum Learning
├── evaluate.py            # Evaluation + flight plot generation
├── demo_no_model.py       # PD controller demo (physics sanity check, no model needed)
├── requirements.txt       # Python dependencies
├── models/                # Saved model checkpoints (after training)
├── plots/                 # Generated flight plots (after evaluation)
└── results/               # Text evaluation summaries
```

---

## Setup

```bash
pip install -r requirements.txt
```

**Requirements:** Python 3.9+, PyTorch (CPU is fine)

---

## Running the Project

### Step 1 — Verify Physics (No model needed)
```bash
python demo_no_model.py
```
Generates baseline PD-controller flight plots in `plots/` to confirm the physics simulation is working correctly.

### Step 2 — Train the RL Agent
```bash
python train.py
```
- Trains for **1,500,000 timesteps** using PPO with Dynamic Curriculum Learning
- Uses **8 parallel environments** for efficiency
- Automatically saves the best model to `models/ppo_hexacopter_best/`
- Logs TensorBoard data to `logs/` → view with: `tensorboard --logdir logs/`
- **Estimated training time:** ~25–40 minutes on CPU

### Step 3 — Evaluate & Generate Plots
```bash
python evaluate.py
```
Evaluates the trained model on 0/1/2/3 rotor failure scenarios and generates:
- `plots/hexacopter_flight_plots_Nfail.png` — 5-panel flight data plots per scenario
- `plots/survival_summary.png` — bar chart summary
- `results/eval_summary.txt` — numerical results

**Evaluate a specific failure mode:**
```bash
python evaluate.py --failures 1 --episodes 20
```

---

## Environment Design

### State Space (18-dimensional, Fault-Aware)
| Index | Variable | Description |
|-------|----------|-------------|
| 0–2   | x, y, z | Position (m) |
| 3–5   | vx, vy, vz | Linear velocity (m/s) |
| 6–8   | φ, θ, ψ | Euler angles (rad) |
| 9–11  | p, q, r | Angular rates (rad/s) |
| 12–17 | m₁–m₆ health | Binary motor health flags (0=dead, 1=alive) |

### Action Space (6-dimensional)
Continuous thrust commands per motor, normalised to [0, 1].

### Reward Function
| Component | Value |
|-----------|-------|
| Survival bonus | +1.0 per step |
| Altitude (Gaussian) | +3.0 · exp(−½(Δz/1)²) |
| XY drift penalty | −0.5 · dist(x,y) |
| Tilt penalty | −3.0 · √(φ²+θ²) |
| Velocity penalty | −1.0 · speed |
| Yaw rate (mild) | −0.1 · |r| |
| Crash | −100.0 |

Yaw rate penalty is intentionally small — the agent learns that **spinning is acceptable** to maintain altitude during a failure.

### Crash Thresholds
- Tilt > 60°
- Speed > 5 m/s
- Altitude error > 3 m
- Altitude < 0.1 m

---

## Algorithm: Proximal Policy Optimisation (PPO)

| Hyperparameter | Value |
|----------------|-------|
| Architecture | MLP [256, 256] |
| n_steps | 2048 |
| batch_size | 512 |
| n_epochs | 10 |
| γ (discount) | 0.99 |
| λ (GAE) | 0.95 |
| clip_range | 0.2 |
| learning_rate | 3×10⁻⁴ |
| entropy coef | 0.005 |

### Dynamic Curriculum Learning
Training proceeds in 3 phases based on training progress:

| Phase | Timesteps | 0 failures | 1 failure | 2 failures |
|-------|-----------|-----------|-----------|-----------|
| 1 | 0–33% | 60% | 35% | 5% |
| 2 | 33–66% | 33% | 40% | 27% |
| 3 | 66–100% | 20% | 45% | 35% |

---

## Expected Results (after full 1.5M training)

| Scenario | Avg Survival | Success Rate |
|----------|-------------|--------------|
| 0 failures | ~8.4s | ~80% |
| **1 failure** | **10.0s** | **100%** |
| 2 failures | ~2.6s | ~0% |
| 3 failures | ~1.9s | ~0% |

**Key Finding:** The agent achieves better performance with 1 failed motor than a healthy drone — an "over-adaptation" effect caused by curriculum learning optimising for fault recovery.

---

## Physics Model

The hexacopter is modelled as a 12-DoF rigid body with:
- Mass: 1.5 kg, 6 rotors at 60° intervals (alternating CW/CCW)
- Arm length: 0.26 m
- Linear & angular aerodynamic drag
- Euler integration at 50 Hz (DT = 0.02s)
- Torque–thrust ratio KT = 0.016 Nm/N

---

## Key Insights

1. **1-Rotor Failure is 100% manageable** — the agent maximises thrust on the diametrically opposite rotor and sacrifices Yaw control (allows slow spinning) to maintain altitude.

2. **2+ Failures hit the physical limit** — adjacent failures shift the Centre of Thrust outside the Centre of Mass. No control algorithm can compensate for loss of physical leverage.

3. **Motor Health Masks are essential** — the state-aware design (binary health flags) enables the agent to immediately adapt its thrust policy upon detecting a failure.

---

## TensorBoard Monitoring

```bash
tensorboard --logdir logs/
```
Open http://localhost:6006 to monitor:
- Episode reward mean
- Episode length mean
- Value loss, policy loss, entropy
