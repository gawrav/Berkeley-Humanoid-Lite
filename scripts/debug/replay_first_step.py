"""Replay first RL_RUNNING step from test logs with different prev_actions init.

Loads the ONNX policy and the logged observations from test_run_024 and test_run_025,
then compares policy output when prev_actions is:
  A) zeros (old behavior)
  B) current joint position in action space (new behavior)

Usage:
    uv run scripts/replay_first_step.py
"""

import json
import numpy as np
import onnxruntime as ort

# Config from policy_latest.yaml
POLICY_PATH = "logs/rsl_rl/biped_v2/2026-02-20_19-27-31/exported/policy.onnx"
ACTION_SCALE = 0.25
ACTION_LIMIT_LOWER = -3.0
ACTION_LIMIT_UPPER = 3.0
NUM_OBS = 45
DEFAULT_JOINT_POSITIONS = np.array([
    0.0, 0.0, -0.2, 0.4, -0.3, 0.0,
    0.0, 0.0, -0.2, 0.4, -0.3, 0.0,
], dtype=np.float32)

JOINT_NAMES = [
    "L_hip_roll", "L_hip_yaw", "L_hip_pitch", "L_knee",
    "L_ankle_pitch", "L_ankle_roll",
    "R_hip_roll", "R_hip_yaw", "R_hip_pitch", "R_knee",
    "R_ankle_pitch", "R_ankle_roll",
]

LOG_FILES = [
    "source/berkeley_humanoid_lite_lowlevel/logs/test_run_024.json",
    "source/berkeley_humanoid_lite_lowlevel/logs/test_run_025.json",
]


def quat_rotate_inverse(q, v):
    q_w = q[0]
    q_vec = q[1:4]
    a = v * (2.0 * q_w ** 2 - 1.0)
    b = np.cross(q_vec, v) * q_w * 2.0
    c = q_vec * (np.dot(q_vec, v)) * 2.0
    return a - b + c


def run_policy(session, obs_vector):
    """Run ONNX policy and return raw + scaled actions."""
    obs = obs_vector.reshape(1, -1).astype(np.float32)
    raw = session.run(None, {"obs": obs})[0][0]
    clipped = np.clip(raw, ACTION_LIMIT_LOWER, ACTION_LIMIT_UPPER)
    scaled = clipped * ACTION_SCALE + DEFAULT_JOINT_POSITIONS
    return raw, clipped, scaled


def build_obs(frame, prev_actions):
    """Build the 45-element observation vector from logged frame data."""
    pi = frame["policy_input"]
    cmd_vel = np.array(pi["command_velocity"], dtype=np.float32)
    ang_vel = np.array(pi["angular_velocity"], dtype=np.float32)
    proj_grav = np.array(pi["projected_gravity"], dtype=np.float32)
    joint_pos_rel = np.array(pi["joint_positions_relative"], dtype=np.float32)
    joint_vel = np.array(pi["joint_velocities"], dtype=np.float32)

    obs = np.concatenate([cmd_vel, ang_vel, proj_grav, joint_pos_rel, joint_vel, prev_actions])
    assert obs.shape[0] == NUM_OBS, f"Expected {NUM_OBS} obs, got {obs.shape[0]}"
    return obs


def main():
    session = ort.InferenceSession(POLICY_PATH)

    for log_file in LOG_FILES:
        print(f"\n{'='*70}")
        print(f"  {log_file}")
        print(f"{'='*70}")

        with open(log_file) as f:
            data = json.load(f)

        # Find first RL_RUNNING frame
        first_running = None
        for frame in data["frames"]:
            if frame["state"] == "RL_RUNNING" and frame["step"] == 1:
                first_running = frame
                break

        if first_running is None:
            print("  No RL_RUNNING frame found!")
            continue

        measured_pos = np.array(first_running["robot_measured"]["joint_positions"], dtype=np.float32)
        logged_prev_actions = np.array(first_running["policy_input"]["prev_actions"], dtype=np.float32)

        # prev_actions in action space for current position
        prev_actions_from_position = (measured_pos - DEFAULT_JOINT_POSITIONS) / ACTION_SCALE

        print(f"\n  Measured joint positions (robot space):")
        for i, name in enumerate(JOINT_NAMES):
            print(f"    {name:<16} {measured_pos[i]:>+8.4f}")

        print(f"\n  prev_actions comparison:")
        print(f"  {'Joint':<16} {'Logged':>10} {'Zeros':>10} {'FromPos':>10}")
        print(f"  {'-'*50}")
        for i, name in enumerate(JOINT_NAMES):
            print(f"  {name:<16} {logged_prev_actions[i]:>+10.4f} {'0.0000':>10} {prev_actions_from_position[i]:>+10.4f}")

        # === Test A: prev_actions = zeros ===
        prev_a_zeros = np.zeros(12, dtype=np.float32)
        obs_zeros = build_obs(first_running, prev_a_zeros)
        raw_z, clip_z, scaled_z = run_policy(session, obs_zeros)

        # === Test B: prev_actions = current position ===
        obs_pos = build_obs(first_running, prev_actions_from_position)
        raw_p, clip_p, scaled_p = run_policy(session, obs_pos)

        # === Test C: prev_actions = logged (garbage from RL_INIT) ===
        obs_logged = build_obs(first_running, logged_prev_actions)
        raw_l, clip_l, scaled_l = run_policy(session, obs_logged)

        print(f"\n  Policy output (scaled to robot space = actual joint targets):")
        print(f"  {'Joint':<16} {'Measured':>9} {'A:zeros':>9} {'B:curPos':>9} {'C:logged':>9} | {'A-meas':>8} {'B-meas':>8} {'C-meas':>8}")
        print(f"  {'-'*94}")
        for i, name in enumerate(JOINT_NAMES):
            da = scaled_z[i] - measured_pos[i]
            db = scaled_p[i] - measured_pos[i]
            dc = scaled_l[i] - measured_pos[i]
            flag = ""
            if max(abs(da), abs(db), abs(dc)) > 0.1:
                flag = " <--"
            print(f"  {name:<16} {measured_pos[i]:>+9.4f} {scaled_z[i]:>+9.4f} {scaled_p[i]:>+9.4f} {scaled_l[i]:>+9.4f}"
                  f" | {da:>+8.4f} {db:>+8.4f} {dc:>+8.4f}{flag}")

        print(f"\n  Raw policy output (action space, ±1 normal range):")
        print(f"  {'Joint':<16} {'A:zeros':>10} {'B:curPos':>10} {'C:logged':>10} | {'A-B diff':>10}")
        print(f"  {'-'*62}")
        for i, name in enumerate(JOINT_NAMES):
            diff = raw_z[i] - raw_p[i]
            flag = " <--" if abs(diff) > 0.1 else ""
            print(f"  {name:<16} {raw_z[i]:>+10.4f} {raw_p[i]:>+10.4f} {raw_l[i]:>+10.4f} | {diff:>+10.4f}{flag}")

        # Summary stats
        delta_z = scaled_z - measured_pos
        delta_p = scaled_p - measured_pos
        delta_l = scaled_l - measured_pos
        print(f"\n  Summary (target - measured position):")
        print(f"    A (zeros):   max |delta| = {np.max(np.abs(delta_z)):.4f} rad, RMS = {np.sqrt(np.mean(delta_z**2)):.4f} rad")
        print(f"    B (curPos):  max |delta| = {np.max(np.abs(delta_p)):.4f} rad, RMS = {np.sqrt(np.mean(delta_p**2)):.4f} rad")
        print(f"    C (logged):  max |delta| = {np.max(np.abs(delta_l)):.4f} rad, RMS = {np.sqrt(np.mean(delta_l**2)):.4f} rad")


if __name__ == "__main__":
    main()
