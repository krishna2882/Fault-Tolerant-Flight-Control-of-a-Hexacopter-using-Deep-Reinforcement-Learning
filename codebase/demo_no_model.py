"""
demo_no_model.py
================
Generates flight plots WITHOUT needing a trained model.
Uses a hand-crafted PD controller as baseline (for demonstration / sanity check).
Also useful to verify the physics environment is working before training.

Usage:
    python demo_no_model.py
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from hexacopter_env import (
    HexacopterEnv, HOVER_THRUST, MAX_THRUST, DT, TARGET_Z
)

os.makedirs("plots",   exist_ok=True)
os.makedirs("results", exist_ok=True)


# ── Simple PD Hover Controller ────────────────────────────────────────────────
class PDController:
    """
    Minimal PD controller that attempts to keep the drone at TARGET_Z.
    Not fault-tolerant — just useful for physics sanity checks.
    """

    KP_ALT   = 1.5
    KD_ALT   = 0.8
    KP_ANGLE = 0.6
    KD_ANGLE = 0.3

    def __init__(self, health_mask):
        self.health = health_mask.copy()

    def act(self, obs):
        """obs: 18-dim array (12 states + 6 health)"""
        z, vz        = obs[2], obs[5]
        phi, theta   = obs[6], obs[7]
        p, q         = obs[9], obs[10]

        # Altitude control → base thrust
        alt_err      = TARGET_Z - z
        base_thrust  = HOVER_THRUST + self.KP_ALT * alt_err - self.KD_ALT * vz

        # Roll / Pitch correction → differential thrust
        roll_corr    = self.KP_ANGLE * phi   + self.KD_ANGLE * p
        pitch_corr   = self.KP_ANGLE * theta + self.KD_ANGLE * q

        # Distribute correction across 6 motors
        # Motors 0,3 (left/right) handle roll
        # Motors 1,4 (front/back diagonals) handle pitch
        action = np.ones(6) * (base_thrust / MAX_THRUST)
        action[0] -= roll_corr / MAX_THRUST
        action[3] += roll_corr / MAX_THRUST
        action[1] -= pitch_corr / MAX_THRUST
        action[4] += pitch_corr / MAX_THRUST

        action = np.clip(action, 0.0, 1.0)
        action *= self.health   # zero out failed motors
        return action


def run_pd_episode(n_fail, max_steps=500):
    """Run a PD-controlled episode; return trajectory."""
    env  = HexacopterEnv(num_failed_motors=n_fail, max_steps=max_steps)
    obs, _ = env.reset()

    failed_motors = [i for i in range(6) if env.get_health_mask()[i] == 0.0]
    ctrl  = PDController(env.get_health_mask())

    traj  = []
    done  = False
    step  = 0

    while not done and step < max_steps:
        action = ctrl.act(obs)
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        state   = obs[:12].copy()
        thrusts = np.clip(action, 0, 1) * MAX_THRUST * env.get_health_mask()

        traj.append({
            "t":       step * DT,
            "state":   state,
            "thrusts": thrusts,
            "reward":  reward,
            "info":    info,
        })
        step += 1

    env.close()
    return traj, failed_motors


def make_flight_plot(traj, n_fail, failed_motors, filename, subtitle="PD Controller"):
    times   = np.array([d["t"]        for d in traj])
    states  = np.array([d["state"]    for d in traj])
    thrusts = np.array([d["thrusts"]  for d in traj])

    MOTOR_COLORS = ["#e74c3c","#f39c12","#1abc9c","#3498db","#9b59b6","#e67e22"]

    fig = plt.figure(figsize=(14, 12))
    fig.patch.set_facecolor("white")

    if n_fail == 0:
        main_title = "Hexacopter Flight Data: 0 Rotor Failures (Healthy)"
    elif n_fail == 1 and len(failed_motors) == 1:
        main_title = f"Hexacopter Flight Data: 1 Rotor Failure (Motor {failed_motors[0]+1} Failed)"
    else:
        mnames     = ", ".join(f"Motor {m+1}" for m in failed_motors)
        main_title = f"Hexacopter Flight Data: {n_fail} Rotor Failures ({mnames} Failed)"

    fig.suptitle(main_title, fontsize=13, fontweight="bold", y=0.98)

    gs = gridspec.GridSpec(3, 2, figure=fig, hspace=0.45, wspace=0.35)

    # Panel 1: Linear Positions
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(times, states[:, 0], color="#e74c3c", label="X Position")
    ax1.plot(times, states[:, 1], color="#2ecc71", label="Y Position")
    ax1.plot(times, states[:, 2], color="#9b59b6", label="Z Position (Altitude)")
    ax1.axhline(TARGET_Z, color="gray", linestyle="--", linewidth=1, label="Target Z")
    ax1.set_title("Linear Positions (m)", fontsize=10)
    ax1.set_xlabel("Time (seconds)"); ax1.set_ylabel("Metres")
    ax1.legend(fontsize=7); ax1.grid(True, alpha=0.3)

    # Panel 2: Linear Velocities
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.plot(times, states[:, 3], color="#e74c3c", label="Vx")
    ax2.plot(times, states[:, 4], color="#2ecc71", label="Vy")
    ax2.plot(times, states[:, 5], color="#9b59b6", label="Vz")
    ax2.set_title("Linear Velocities (m/s)", fontsize=10)
    ax2.set_xlabel("Time (seconds)"); ax2.set_ylabel("Velocity")
    ax2.legend(fontsize=7); ax2.grid(True, alpha=0.3)

    # Panel 3: Euler Angles
    ax3 = fig.add_subplot(gs[1, 0])
    ax3.plot(times, np.rad2deg(states[:, 6]),  color="#e74c3c", label="Roll (Phi)")
    ax3.plot(times, np.rad2deg(states[:, 7]),  color="#3498db", label="Pitch (Theta)")
    ax3.plot(times, np.rad2deg(states[:, 8]),  color="#2ecc71", label="Yaw (Psi)")
    ax3.set_title("Euler Angles (Degrees)", fontsize=10)
    ax3.set_xlabel("Time (seconds)"); ax3.set_ylabel("Degrees")
    ax3.legend(fontsize=7); ax3.grid(True, alpha=0.3)

    # Panel 4: Angular Velocities
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.plot(times, states[:,  9], color="#e74c3c", label="Roll Rate (p)")
    ax4.plot(times, states[:, 10], color="#3498db", label="Pitch Rate (q)")
    ax4.plot(times, states[:, 11], color="#9b59b6", label="Yaw Rate (r)")
    ax4.set_title("Angular Velocities (rad/s)", fontsize=10)
    ax4.set_xlabel("Time (seconds)"); ax4.set_ylabel("Rad/s")
    ax4.legend(fontsize=7); ax4.grid(True, alpha=0.3)

    # Panel 5: Motor Thrusts
    ax5 = fig.add_subplot(gs[2, :])
    for i in range(6):
        failed = i in failed_motors
        lbl    = f"Motor {i+1}" + (" (FAILED)" if failed else "")
        ax5.plot(times, thrusts[:, i],
                 color="#e74c3c" if failed else MOTOR_COLORS[i],
                 linestyle="--" if failed else "-",
                 linewidth=2 if failed else 1.2,
                 label=lbl)
    ax5.set_title("Motor Thrust Outputs (RPM Proxy)", fontsize=10)
    ax5.set_xlabel("Time (seconds)"); ax5.set_ylabel("Thrust (Newtons)")
    ax5.legend(fontsize=7, ncol=3); ax5.grid(True, alpha=0.3)

    plt.savefig(filename, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → {filename}")


def main():
    print("=" * 55)
    print("  Hexacopter FTC — Physics Demo (PD Controller)")
    print("=" * 55)

    summary = []

    for n_fail in [0, 1, 2, 3]:
        print(f"\n  Scenario: {n_fail} rotor(s) failed …")
        traj, failed = run_pd_episode(n_fail)

        t_end    = traj[-1]["t"]
        crashed  = traj[-1]["info"]["crashed"]
        status   = "CRASHED" if crashed else "SURVIVED (10s)"

        print(f"    Survival: {t_end:.2f}s  |  {status}")
        print(f"    Failed motors: {[m+1 for m in failed]}")

        fname = f"plots/demo_pd_{n_fail}fail.png"
        make_flight_plot(traj, n_fail, failed, fname)
        summary.append((n_fail, t_end, crashed))

    # Summary to file
    with open("results/demo_summary.txt", "w") as f:
        f.write("PD Controller Baseline — Physics Demo\n")
        f.write("=" * 40 + "\n")
        for n_fail, t, c in summary:
            f.write(f"\n{n_fail} failure(s): {t:.2f}s — {'CRASHED' if c else 'OK'}\n")

    print("\n✓ Demo complete. Plots in ./plots/  Results in ./results/")


if __name__ == "__main__":
    main()
