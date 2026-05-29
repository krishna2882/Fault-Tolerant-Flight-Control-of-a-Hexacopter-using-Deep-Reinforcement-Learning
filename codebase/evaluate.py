"""
evaluate.py
===========
Evaluates the trained PPO model under different failure modes and generates
publication-quality plots matching the hexacopter_flight_plots.png format.

Usage:
    python evaluate.py                    # uses best saved model
    python evaluate.py --model <path>     # specify model path
    python evaluate.py --failures 1       # only evaluate 1-rotor failure

Outputs:
    plots/hexacopter_flight_plots_<n>fail.png
    plots/survival_summary.png
    results/eval_summary.txt
"""

import argparse
import os
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from stable_baselines3 import PPO

from hexacopter_env import HexacopterEnv, DT, TARGET_Z

os.makedirs("plots",   exist_ok=True)
os.makedirs("results", exist_ok=True)


# ── Colour scheme ─────────────────────────────────────────────────────────────
C = {
    "x":     "#e74c3c",   # red
    "y":     "#2ecc71",   # green
    "z":     "#9b59b6",   # purple
    "vx":    "#e74c3c",
    "vy":    "#2ecc71",
    "vz":    "#9b59b6",
    "phi":   "#e74c3c",
    "theta": "#3498db",
    "psi":   "#2ecc71",
    "p":     "#e74c3c",
    "q":     "#3498db",
    "r":     "#9b59b6",
}

MOTOR_COLORS = [
    "#e74c3c", "#f39c12", "#1abc9c",
    "#3498db", "#9b59b6", "#e74c3c",
]


def run_episode(model, env, deterministic=True):
    """Run a single episode; return trajectory arrays."""
    obs, _ = env.reset()
    done = False
    traj = []
    step = 0

    while not done:
        action, _ = model.predict(obs, deterministic=deterministic)
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        # state = obs[:12], health = obs[12:]
        state  = obs[:12].copy()
        health = env.get_health_mask()
        # Reconstruct motor thrusts from action
        from hexacopter_env import MAX_THRUST
        thrusts = np.clip(action, 0, 1) * MAX_THRUST * health

        traj.append({
            "t":       step * DT,
            "state":   state,
            "thrusts": thrusts,
            "health":  health.copy(),
            "reward":  reward,
            "info":    info,
        })
        step += 1
        if step > 600:   # safety cap
            break

    return traj


def make_flight_plot(traj, n_fail, failed_motors, filename):
    """
    Recreate the 5-panel hexacopter_flight_plots figure.
    Panels: Linear Positions, Linear Velocities,
            Euler Angles, Angular Velocities, Motor Thrust Outputs.
    """
    times   = np.array([d["t"] for d in traj])
    states  = np.array([d["state"] for d in traj])
    thrusts = np.array([d["thrusts"] for d in traj])

    fig = plt.figure(figsize=(14, 12))
    fig.patch.set_facecolor("white")
    title = f"Hexacopter Flight Data: {n_fail} Rotor Failure"
    if n_fail == 1 and len(failed_motors) == 1:
        title += f" (Motor {failed_motors[0]+1} Failed)"
    elif n_fail > 1:
        mnames = ", ".join(f"Motor {m+1}" for m in failed_motors)
        title += f" ({mnames} Failed)"
    elif n_fail == 0:
        title = "Hexacopter Flight Data: 0 Rotor Failures (Healthy)"
    fig.suptitle(title, fontsize=13, fontweight="bold", y=0.98)

    gs = gridspec.GridSpec(3, 2, figure=fig, hspace=0.45, wspace=0.35)

    # ── Panel 1: Linear Positions ─────────────────────────────────────────
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(times, states[:, 0], color=C["x"],  label="X Position")
    ax1.plot(times, states[:, 1], color=C["y"],  label="Y Position")
    ax1.plot(times, states[:, 2], color=C["z"],  label="Z Position (Altitude)")
    ax1.axhline(TARGET_Z, color="gray", linestyle="--", linewidth=1, label="Target Z")
    ax1.set_title("Linear Positions (m)", fontsize=10)
    ax1.set_xlabel("Time (seconds)"); ax1.set_ylabel("Metres")
    ax1.legend(fontsize=7, loc="upper right"); ax1.grid(True, alpha=0.3)

    # ── Panel 2: Linear Velocities ────────────────────────────────────────
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.plot(times, states[:, 3], color=C["vx"], label="Vx")
    ax2.plot(times, states[:, 4], color=C["vy"], label="Vy")
    ax2.plot(times, states[:, 5], color=C["vz"], label="Vz")
    ax2.set_title("Linear Velocities (m/s)", fontsize=10)
    ax2.set_xlabel("Time (seconds)"); ax2.set_ylabel("Velocity")
    ax2.legend(fontsize=7, loc="upper right"); ax2.grid(True, alpha=0.3)

    # ── Panel 3: Euler Angles ─────────────────────────────────────────────
    ax3 = fig.add_subplot(gs[1, 0])
    ax3.plot(times, np.rad2deg(states[:, 6]),  color=C["phi"],   label="Roll (Phi)")
    ax3.plot(times, np.rad2deg(states[:, 7]),  color=C["theta"], label="Pitch (Theta)")
    ax3.plot(times, np.rad2deg(states[:, 8]),  color=C["psi"],   label="Yaw (Psi)")
    ax3.set_title("Euler Angles (Degrees)", fontsize=10)
    ax3.set_xlabel("Time (seconds)"); ax3.set_ylabel("Degrees")
    ax3.legend(fontsize=7, loc="upper left"); ax3.grid(True, alpha=0.3)

    # ── Panel 4: Angular Velocities ───────────────────────────────────────
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.plot(times, states[:, 9],  color=C["p"], label="Roll Rate (p)")
    ax4.plot(times, states[:, 10], color=C["q"], label="Pitch Rate (q)")
    ax4.plot(times, states[:, 11], color=C["r"], label="Yaw Rate (r)")
    ax4.set_title("Angular Velocities (rad/s)", fontsize=10)
    ax4.set_xlabel("Time (seconds)"); ax4.set_ylabel("Rad/s")
    ax4.legend(fontsize=7, loc="upper left"); ax4.grid(True, alpha=0.3)

    # ── Panel 5: Motor Thrust Outputs ─────────────────────────────────────
    ax5 = fig.add_subplot(gs[2, :])
    for i in range(6):
        lbl    = f"Motor {i+1}"
        failed = i in failed_motors
        style  = "--" if failed else "-"
        lw     = 2   if failed else 1.2
        color  = "#e74c3c" if failed else MOTOR_COLORS[i]
        if failed:
            lbl += " (FAILED)"
        ax5.plot(times, thrusts[:, i], color=color,
                 linestyle=style, linewidth=lw, label=lbl)
    ax5.set_title("Motor Thrust Outputs (RPM Proxy)", fontsize=10)
    ax5.set_xlabel("Time (seconds)"); ax5.set_ylabel("Thrust (Newtons)")
    ax5.legend(fontsize=7, ncol=3, loc="upper right"); ax5.grid(True, alpha=0.3)

    plt.savefig(filename, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓ Saved → {filename}")


def evaluate_scenario(model, n_fail, n_episodes=10, plot=True):
    """Run multiple episodes for a given failure count; return survival stats."""
    survival_times = []
    success_count  = 0

    for ep in range(n_episodes):
        env  = HexacopterEnv(num_failed_motors=n_fail, max_steps=500)
        traj = run_episode(model, env)

        t_survive = traj[-1]["t"]
        crashed   = traj[-1]["info"]["crashed"]
        survival_times.append(t_survive)
        if not crashed:
            success_count += 1

        # Save plot for first episode only
        if plot and ep == 0:
            # Find which motors failed
            failed = [i for i in range(6) if env.get_health_mask()[i] == 0.0]
            fname  = f"plots/hexacopter_flight_plots_{n_fail}fail.png"
            make_flight_plot(traj, n_fail, failed, fname)

        env.close()

    avg_survival = np.mean(survival_times)
    success_rate = success_count / n_episodes * 100
    return avg_survival, success_rate, survival_times


def make_summary_plot(results):
    """
    Bar chart: average survival time per failure mode.
    """
    labels  = [f"{k} Rotor(s)\nFailed" for k in sorted(results.keys())]
    avgs    = [results[k]["avg"] for k in sorted(results.keys())]
    srates  = [results[k]["sr"]  for k in sorted(results.keys())]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("Hexacopter FTC Evaluation Summary", fontsize=13, fontweight="bold")

    # Survival time bar chart
    colors = ["#2ecc71" if s == 100 else "#e67e22" if s >= 50 else "#e74c3c"
              for s in srates]
    bars = axes[0].bar(labels, avgs, color=colors, edgecolor="black", linewidth=0.7)
    axes[0].set_title("Average Survival Time (s)")
    axes[0].set_ylabel("Seconds")
    axes[0].axhline(10.0, color="gray", linestyle="--", label="Max episode (10s)")
    axes[0].legend(fontsize=8)
    axes[0].grid(axis="y", alpha=0.3)
    for bar, val in zip(bars, avgs):
        axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                     f"{val:.2f}s", ha="center", va="bottom", fontsize=9)

    # Success rate pie-ish (bar)
    axes[1].bar(labels, srates, color=colors, edgecolor="black", linewidth=0.7)
    axes[1].set_title("Success Rate (%)")
    axes[1].set_ylabel("%")
    axes[1].set_ylim(0, 110)
    axes[1].grid(axis="y", alpha=0.3)
    for bar, val in zip(axes[1].patches, srates):
        axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                     f"{val:.0f}%", ha="center", va="bottom", fontsize=9)

    plt.tight_layout()
    plt.savefig("plots/survival_summary.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  ✓ Saved → plots/survival_summary.png")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",    default="models/ppo_hexacopter_best/best_model",
                        help="Path to saved SB3 model")
    parser.add_argument("--failures", type=int, default=None,
                        help="Evaluate only this failure count (0/1/2/3)")
    parser.add_argument("--episodes", type=int, default=10,
                        help="Episodes per scenario")
    args = parser.parse_args()

    # Try loading model; fall back to final if best not found
    model_path = args.model
    for candidate in [model_path, "models/ppo_hexacopter_final",
                      "models/ppo_hexacopter_best/best_model"]:
        try:
            model = PPO.load(candidate)
            print(f"✓ Loaded model: {candidate}")
            break
        except Exception:
            continue
    else:
        print("✗ No trained model found. Run train.py first.")
        sys.exit(1)

    scenarios = [0, 1, 2, 3] if args.failures is None else [args.failures]

    print("\n" + "=" * 55)
    print("  Evaluation Results")
    print("=" * 55)

    all_results = {}
    for n_fail in scenarios:
        print(f"\n  [{n_fail} Rotor(s) Failed]  ({args.episodes} episodes)")
        avg, sr, times = evaluate_scenario(
            model, n_fail,
            n_episodes=args.episodes,
            plot=True,
        )
        all_results[n_fail] = {"avg": avg, "sr": sr, "times": times}
        print(f"    Avg survival : {avg:.2f}s")
        print(f"    Success rate : {sr:.0f}%")

    # Summary plot
    print("\n  Generating summary plot …")
    make_summary_plot(all_results)

    # Write text summary
    with open("results/eval_summary.txt", "w") as f:
        f.write("Hexacopter FTC — Evaluation Summary\n")
        f.write("=" * 40 + "\n")
        for k, v in sorted(all_results.items()):
            f.write(f"\n{k} Rotor(s) Failed:\n")
            f.write(f"  Avg survival : {v['avg']:.2f}s\n")
            f.write(f"  Success rate : {v['sr']:.0f}%\n")
            f.write(f"  All times    : {[round(t,2) for t in v['times']]}\n")
    print("  ✓ Saved → results/eval_summary.txt")
    print("\nDone.\n")


if __name__ == "__main__":
    main()
