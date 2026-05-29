"""
hexacopter_env.py
=================
Custom Gymnasium environment for a 6-rotor UAV (Hexacopter).
Models 12-DoF rigid body dynamics with fault-aware state space.

State  (18-dim): [x,y,z, vx,vy,vz, phi,theta,psi, p,q,r, m1..m6_health]
Action (6-dim) : Thrust commands per motor [0, MAX_THRUST]
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces


# ── Physical constants ────────────────────────────────────────────────────────
GRAVITY        = 9.81          # m/s²
MASS           = 1.5           # kg
ARM_LENGTH     = 0.26          # m  (centre → rotor)
MAX_THRUST     = 6.0           # N  per motor (max)
MIN_THRUST     = 0.0           # N  per motor
KT             = 0.016         # Torque-to-thrust ratio  (Nm/N)
DT             = 0.02          # s  (50 Hz)
MAX_TILT       = 60.0          # degrees – crash threshold
MAX_VELOCITY   = 5.0           # m/s – crash threshold
MAX_ALT_ERR    = 3.0           # m   – crash threshold

# Inertia tensor (diagonal) kg·m²
IXX = 0.0152
IYY = 0.0152
IZZ = 0.0294

# Aerodynamic drag coefficients
LINEAR_DRAG    = 0.25
ANGULAR_DRAG   = 0.02

# Hover thrust per motor (6 motors hold 1.5 kg)
HOVER_THRUST   = (MASS * GRAVITY) / 6.0

# Rotor positions (evenly at 60° intervals, alternating CW/CCW)
# Motor i is at angle i*60°, direction: +1=CCW, -1=CW
MOTOR_ANGLES   = np.deg2rad([0, 60, 120, 180, 240, 300])
MOTOR_DIRS     = np.array([1, -1, 1, -1, 1, -1], dtype=float)  # CCW / CW

# Precompute rotor x,y positions
ROTOR_X = ARM_LENGTH * np.cos(MOTOR_ANGLES)
ROTOR_Y = ARM_LENGTH * np.sin(MOTOR_ANGLES)

TARGET_Z = 5.0   # Target hover altitude (m)


class HexacopterEnv(gym.Env):
    """
    Fault-Tolerant Hexacopter Hover Environment.

    Parameters
    ----------
    num_failed_motors : int or None
        Fixed number of failed motors per episode.
        None → curriculum mode (0/1/2 randomly chosen each reset).
    max_steps : int
        Episode length in timesteps.
    curriculum_probs : list
        Probability of [0,1,2] failures in curriculum mode.
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        num_failed_motors=None,
        max_steps=500,
        curriculum_probs=None,
        seed=None,
    ):
        super().__init__()

        self.num_failed_motors  = num_failed_motors
        self.max_steps          = max_steps
        self.curriculum_probs   = curriculum_probs or [0.33, 0.34, 0.33]

        # ── Observation space ────────────────────────────────────────────────
        # [x,y,z, vx,vy,vz, phi,theta,psi, p,q,r, health×6]
        obs_low  = np.array(
            [-10,-10, 0, -10,-10,-10, -np.pi,-np.pi,-np.pi,
             -20,-20,-20, 0,0,0,0,0,0], dtype=np.float32)
        obs_high = np.array(
            [ 10, 10,20,  10, 10, 10,  np.pi, np.pi, np.pi,
              20, 20, 20, 1,1,1,1,1,1], dtype=np.float32)
        self.observation_space = spaces.Box(obs_low, obs_high, dtype=np.float32)

        # ── Action space: normalised thrust [0,1] per motor ─────────────────
        self.action_space = spaces.Box(
            low=0.0, high=1.0, shape=(6,), dtype=np.float32
        )

        self.rng = np.random.default_rng(seed)
        self._state  = None
        self._health = None
        self._steps  = 0

    # ── Reset ─────────────────────────────────────────────────────────────────
    def reset(self, seed=None, options=None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)

        # Decide failures
        if self.num_failed_motors is None:
            n_fail = self.rng.choice([0, 1, 2], p=self.curriculum_probs)
        else:
            n_fail = self.num_failed_motors

        # Which motors fail
        self._health = np.ones(6, dtype=np.float32)
        if n_fail > 0:
            failed = self.rng.choice(6, size=n_fail, replace=False)
            self._health[failed] = 0.0

        # State: [x,y,z, vx,vy,vz, phi,theta,psi, p,q,r]
        self._state = np.zeros(12, dtype=np.float64)
        self._state[2] = TARGET_Z          # start at target altitude
        self._steps = 0

        return self._get_obs(), {}

    # ── Step ──────────────────────────────────────────────────────────────────
    def step(self, action):
        action = np.clip(action, 0.0, 1.0).astype(np.float64)

        # Mask dead motors
        thrusts = action * MAX_THRUST * self._health.astype(np.float64)

        # Compute wrench
        total_thrust = np.sum(thrusts)
        tau_roll     = np.sum(thrusts * ROTOR_Y)
        tau_pitch    = -np.sum(thrusts * ROTOR_X)
        tau_yaw      = np.sum(thrusts * MOTOR_DIRS * KT)

        s = self._state
        x,y,z         = s[0], s[1], s[2]
        vx,vy,vz      = s[3], s[4], s[5]
        phi,theta,psi = s[6], s[7], s[8]
        p, q, r       = s[9], s[10], s[11]

        cphi,sphi     = np.cos(phi),   np.sin(phi)
        cth,sth,tth   = np.cos(theta), np.sin(theta), np.tan(theta)
        cpsi,spsi     = np.cos(psi),   np.sin(psi)

        # ── Translational dynamics (body → world) ─────────────────────────
        # Rotation matrix (ZYX convention)
        ax = (total_thrust/MASS) * (spsi*sphi + cpsi*sth*cphi) - LINEAR_DRAG/MASS*vx
        ay = (total_thrust/MASS) * (-cpsi*sphi + spsi*sth*cphi) - LINEAR_DRAG/MASS*vy
        az = (total_thrust/MASS) * (cth*cphi) - GRAVITY - LINEAR_DRAG/MASS*vz

        # ── Rotational dynamics ───────────────────────────────────────────
        pdot = (tau_roll  - (IYY-IZZ)*q*r) / IXX - ANGULAR_DRAG/IXX*p
        qdot = (tau_pitch - (IZZ-IXX)*p*r) / IYY - ANGULAR_DRAG/IYY*q
        rdot = (tau_yaw   - (IXX-IYY)*p*q) / IZZ - ANGULAR_DRAG/IZZ*r

        # ── Euler angle kinematics ────────────────────────────────────────
        phidot   = p + (q*sphi + r*cphi)*tth
        thetadot = q*cphi - r*sphi
        psidot   = (q*sphi + r*cphi)/max(cth, 1e-4)

        # ── Integrate (Euler) ─────────────────────────────────────────────
        s[0] += vx*DT;   s[1] += vy*DT;   s[2] += vz*DT
        s[3] += ax*DT;   s[4] += ay*DT;   s[5] += az*DT
        s[6] += phidot*DT; s[7] += thetadot*DT; s[8] += psidot*DT
        s[9] += pdot*DT; s[10] += qdot*DT; s[11] += rdot*DT

        # Wrap yaw to [-pi, pi]
        s[8] = np.arctan2(np.sin(s[8]), np.cos(s[8]))

        self._steps += 1

        # ── Crash detection ───────────────────────────────────────────────
        tilt_deg = np.rad2deg(np.sqrt(s[6]**2 + s[7]**2))
        speed    = np.sqrt(s[3]**2 + s[4]**2 + s[5]**2)
        alt_err  = abs(s[2] - TARGET_Z)
        crashed  = (
            tilt_deg > MAX_TILT or
            speed    > MAX_VELOCITY or
            alt_err  > MAX_ALT_ERR or
            s[2]     < 0.1
        )
        terminated = crashed
        truncated  = (self._steps >= self.max_steps)

        # ── Reward ────────────────────────────────────────────────────────
        reward = self._compute_reward(s, crashed)

        obs = self._get_obs()
        info = {
            "tilt_deg":  tilt_deg,
            "altitude":  s[2],
            "crashed":   crashed,
            "n_healthy": int(np.sum(self._health)),
        }
        return obs, reward, terminated, truncated, info

    def _compute_reward(self, s, crashed):
        if crashed:
            return -100.0

        # Survival bonus
        reward = 1.0

        # Gaussian altitude reward
        alt_err = s[2] - TARGET_Z
        reward += 3.0 * np.exp(-0.5 * (alt_err / 1.0)**2)

        # XY drift penalty (soft)
        xy_dist = np.sqrt(s[0]**2 + s[1]**2)
        reward -= 0.5 * xy_dist

        # Tilt penalty
        tilt = np.sqrt(s[6]**2 + s[7]**2)
        reward -= 3.0 * tilt

        # Velocity penalty
        vel = np.sqrt(s[3]**2 + s[4]**2 + s[5]**2)
        reward -= 1.0 * vel

        # Yaw penalty is MILD (allow spinning to survive)
        reward -= 0.1 * abs(s[11])   # r = yaw rate

        return float(reward)

    def _get_obs(self):
        obs = np.concatenate([self._state.astype(np.float32), self._health])
        return np.clip(obs, self.observation_space.low, self.observation_space.high)

    def get_health_mask(self):
        return self._health.copy()
