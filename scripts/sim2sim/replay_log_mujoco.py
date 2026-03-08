# Copyright (c) 2025, The Berkeley Humanoid Lite Project Developers.

"""
Single-step sim-to-real comparison: for each logged frame, reset MuJoCo to
the real robot's measured state, apply the logged action for one policy step,
and compare the sim result with what the real robot actually did next.

This avoids error accumulation and directly measures the per-step physics gap
between MuJoCo and the real hardware.

Modes:
  --headless    Print per-step comparison stats (default: also launches viewer)
  --visual      Visual replay: play back logged joint positions in MuJoCo
                viewer (kinematic, no physics). Shows what the real robot did.

Usage:
    # Headless analysis (recommended for quantitative comparison):
    uv run scripts/sim2sim/replay_log_mujoco.py \
        --log source/berkeley_humanoid_lite_lowlevel/logs/test_run_021.json \
        --headless

    # Visual replay of real robot behavior in MuJoCo viewer:
    uv run scripts/sim2sim/replay_log_mujoco.py \
        --log source/berkeley_humanoid_lite_lowlevel/logs/test_run_021.json \
        --visual

    # Use hardware torque limit (4 Nm) instead of sim default (6 Nm):
    uv run scripts/sim2sim/replay_log_mujoco.py \
        --log source/berkeley_humanoid_lite_lowlevel/logs/test_run_021.json \
        --headless --torque-limit 4.0

    # Skip first N frames:
    uv run scripts/sim2sim/replay_log_mujoco.py \
        --log source/berkeley_humanoid_lite_lowlevel/logs/test_run_021.json \
        --headless --start-frame 20

    # Save comparison CSV:
    uv run scripts/sim2sim/replay_log_mujoco.py \
        --log source/berkeley_humanoid_lite_lowlevel/logs/test_run_021.json \
        --headless --csv comparison.csv

    # Multi-step replay with 80ms actuator delay (joints evolve freely):
    uv run scripts/sim2sim/replay_log_mujoco.py \
        --log source/berkeley_humanoid_lite_lowlevel/logs/test_run_021.json \
        --headless --delay 80

    # Sweep delays (0-160ms) to find best match with real robot:
    uv run scripts/sim2sim/replay_log_mujoco.py \
        --log source/berkeley_humanoid_lite_lowlevel/logs/test_run_021.json \
        --delay-sweep

    # Sweep actuator models (EMA filter tau + torque limit) to find best match:
    uv run scripts/sim2sim/replay_log_mujoco.py \
        --log source/berkeley_humanoid_lite_lowlevel/logs/test_run_021.json \
        --actuator-sweep

    # Free-run sweep: no base pin, no state reset -- the ultimate sim-real test:
    uv run scripts/sim2sim/replay_log_mujoco.py \
        --log source/berkeley_humanoid_lite_lowlevel/logs/test_run_021.json \
        --free-sweep

    # Single free-run with specific config:
    uv run scripts/sim2sim/replay_log_mujoco.py \
        --log source/berkeley_humanoid_lite_lowlevel/logs/test_run_021.json \
        --delay 0 --free --torque-limit 4.0 --ema-tau 10

    # Policy-in-the-loop: load trained policy and run it from log's initial state:
    uv run scripts/sim2sim/replay_log_mujoco.py \
        --log source/berkeley_humanoid_lite_lowlevel/logs/test_run_021.json \
        --policy-run --config configs/policy_latest_hard_v1.yaml

    # Policy-in-the-loop with 4Nm torque limit:
    uv run scripts/sim2sim/replay_log_mujoco.py \
        --log source/berkeley_humanoid_lite_lowlevel/logs/test_run_021.json \
        --policy-run --config configs/policy_latest_hard_v1.yaml --torque-limit 4.0
"""

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np
import torch

# Try to import gamepad
try:
    from berkeley_humanoid_lite_lowlevel.policy.gamepad_direct import Se2Gamepad
    GAMEPAD_AVAILABLE = True
except ImportError:
    GAMEPAD_AVAILABLE = False


JOINT_NAMES = [
    "L_hip_roll", "L_hip_yaw", "L_hip_pitch",
    "L_knee", "L_ankle_pitch", "L_ankle_roll",
    "R_hip_roll", "R_hip_yaw", "R_hip_pitch",
    "R_knee", "R_ankle_pitch", "R_ankle_roll",
]

DEFAULT_JOINT_POSITIONS = np.array(
    [0.0, 0.0, -0.2, 0.4, -0.3, 0.0,
     0.0, 0.0, -0.2, 0.4, -0.3, 0.0],
    dtype=np.float32
)


def load_log(log_path):
    """Load full log file."""
    with open(log_path) as f:
        data = json.load(f)
    return data["metadata"], data["frames"]


def find_rl_running_start(frames):
    """Find the first frame with non-zero actions (policy actively controlling).

    Logging starts at RL_RUNNING, so all frames are active. But the first
    few frames may have transient startup behavior, so we skip to the first
    frame with meaningful actions.
    """
    for i, frame in enumerate(frames):
        actions = frame["policy_output"]["raw_actions"]
        if any(abs(a) > 0.01 for a in actions):
            return i
    return 0


def setup_mujoco(physics_dt):
    """Load MuJoCo model and set timestep."""
    project_root = Path(__file__).resolve().parents[2]
    mjcf_path = (
        project_root
        / "source" / "berkeley_humanoid_lite_assets" / "data" / "robots"
        / "berkeley_humanoid" / "berkeley_humanoid_lite" / "mjcf"
        / "bhl_biped_scene.xml"
    )

    if not mjcf_path.exists():
        print(f"  ERROR: MJCF not found at {mjcf_path}")
        sys.exit(1)

    model = mujoco.MjModel.from_xml_path(str(mjcf_path))
    data = mujoco.MjData(model)
    model.opt.timestep = physics_dt
    return model, data


FOOT_BODY_NAMES = ["leg_left_ankle_roll", "leg_right_ankle_roll"]

# Small offset above ground to ensure contact without penetration
GROUND_CLEARANCE = 0.001  # 1mm


def _get_foot_geom_ids(model):
    """Find geom IDs belonging to foot bodies."""
    foot_geom_ids = []
    for name in FOOT_BODY_NAMES:
        body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
        if body_id < 0:
            continue
        for gid in range(model.ngeom):
            if model.geom_bodyid[gid] == body_id:
                foot_geom_ids.append(gid)
    return foot_geom_ids


def _lowest_foot_z(model, data, foot_geom_ids):
    """Compute the lowest z-coordinate of any foot geom in world frame.

    For box geoms, the lowest z is:
        geom_pos_z - |R[2,0]|*hx - |R[2,1]|*hy - |R[2,2]|*hz
    where R is the geom rotation matrix and hx,hy,hz are half-sizes.
    """
    lowest = float("inf")
    for gid in foot_geom_ids:
        pos = data.geom_xpos[gid]
        mat = data.geom_xmat[gid].reshape(3, 3)
        sz = model.geom_size[gid]
        bottom_z = pos[2] - abs(mat[2, 0]) * sz[0] - abs(mat[2, 1]) * sz[1] - abs(mat[2, 2]) * sz[2]
        lowest = min(lowest, bottom_z)
    return lowest


def compute_base_height(model, data, joint_pos, quat, foot_geom_ids):
    """Compute base z so the lowest foot just touches the ground (z=0).

    Sets qpos with joint positions and orientation at an arbitrary height,
    uses forward kinematics to find foot positions, then adjusts base z
    so the lowest foot bottom is at ground level.
    """
    num_joints = len(JOINT_NAMES)
    ref_z = 1.0  # arbitrary reference height (clearly above ground)

    data.qpos[:] = 0
    data.qpos[2] = ref_z
    data.qpos[3:7] = quat
    data.qpos[7:7 + num_joints] = joint_pos
    mujoco.mj_forward(model, data)

    foot_z = _lowest_foot_z(model, data, foot_geom_ids)
    # Shift base so foot_z lands at GROUND_CLEARANCE
    return ref_z - foot_z + GROUND_CLEARANCE


def set_frame_state(model, data, frame, foot_geom_ids):
    """Set MuJoCo state to match a logged frame with correct ground contact.

    Computes the base height from foot kinematics so the lowest foot
    touches the ground, then sets joint positions, velocities, and orientation.

    Returns the computed base height.
    """
    num_joints = len(JOINT_NAMES)
    joint_pos = np.array(frame["robot_measured"]["joint_positions"], dtype=np.float32)
    joint_vel = np.array(frame["robot_measured"]["joint_velocities"], dtype=np.float32)
    quat = np.array(frame["raw_observations"]["quaternion"], dtype=np.float32)

    base_z = compute_base_height(model, data, joint_pos, quat, foot_geom_ids)

    data.qpos[0:2] = [0, 0]
    data.qpos[2] = base_z
    data.qpos[3:7] = quat
    data.qpos[7:7 + num_joints] = joint_pos
    data.qvel[:] = 0
    data.qvel[6:6 + num_joints] = joint_vel
    mujoco.mj_forward(model, data)

    return base_z


def run_visual_replay(model, data, frames, start_idx, policy_dt):
    """Play back logged joint positions in MuJoCo viewer (kinematic, no physics).

    Sets MuJoCo joint positions to the real measured values each frame,
    with base height computed from foot kinematics so feet touch the ground.
    """
    foot_geom_ids = _get_foot_geom_ids(model)
    viewer = mujoco.viewer.launch_passive(model, data)

    total = len(frames) - start_idx
    print(f"\n  Visual replay: {total} frames at {1/policy_dt:.0f} Hz")
    print(f"  Close the viewer window to stop.\n")

    try:
        fi = start_idx
        while viewer.is_running() and fi < len(frames):
            frame_start = time.perf_counter()

            base_z = set_frame_state(model, data, frames[fi], foot_geom_ids)
            viewer.sync()

            rel = fi - start_idx
            if rel < 3 or rel % 50 == 0:
                t = frames[fi]["timestamp"]
                real_pos = frames[fi]["robot_measured"]["joint_positions"]
                print(f"  Frame {fi:>4} (t={t:.2f}s)  base_z={base_z:.3f}m  "
                      f"L_knee={real_pos[3]:+.3f}  R_knee={real_pos[9]:+.3f}")

            fi += 1

            remaining = policy_dt - (time.perf_counter() - frame_start)
            if remaining > 0:
                time.sleep(remaining)

        if viewer.is_running():
            print(f"\n  Replay complete ({fi - start_idx} frames). Close viewer to exit.")
            while viewer.is_running():
                time.sleep(0.1)

    except KeyboardInterrupt:
        pass

    viewer.close()


def replay(model, data, frames, start_idx, policy_dt, physics_dt,
           joint_kp, joint_kd, effort_limits, action_scale,
           delay_steps=0, ema_tau_ms=0, viewer=None):
    """Single-step replay: reset MuJoCo to real state each frame, step once,
    compare the one-step sim response with the real next-frame measurement.

    When delay_steps > 0, the PD target comes from the action logged N frames
    earlier, simulating motor lag.

    When ema_tau_ms > 0, applies a first-order low-pass (EMA) filter on PD
    torque at each physics substep, matching the real motor controller's
    torque smoothing. The filter state is initialized from the previous
    frame's action to avoid startup transients.

    Returns list of per-frame comparison dicts.
    """
    num_joints = len(JOINT_NAMES)
    physics_substeps = int(np.round(policy_dt / physics_dt))

    # EMA filter: convert time constant to per-substep alpha
    # alpha = 1 means no filtering (instantaneous)
    if ema_tau_ms > 0:
        tau_s = ema_tau_ms / 1000.0
        ema_alpha = 1.0 - np.exp(-physics_dt / tau_s)
    else:
        ema_alpha = 1.0

    results = []

    for fi in range(start_idx, len(frames) - 1):
        step_start = time.perf_counter()
        frame = frames[fi]
        next_frame = frames[fi + 1]

        # --- Reset sim to match this frame's real state ---
        real_pos = np.array(frame["robot_measured"]["joint_positions"], dtype=np.float32)
        real_vel = np.array(frame["robot_measured"]["joint_velocities"], dtype=np.float32)
        real_quat = np.array(frame["raw_observations"]["quaternion"], dtype=np.float32)

        data.qpos[0:2] = [0, 0]
        data.qpos[2] = 0.34
        data.qpos[3:7] = real_quat
        data.qpos[7:7 + num_joints] = real_pos
        data.qvel[:] = 0
        data.qvel[6:6 + num_joints] = real_vel
        mujoco.mj_forward(model, data)

        # --- Apply the (possibly delayed) action for one policy step ---
        action_fi = max(start_idx, fi - delay_steps)
        action_frame = frames[action_fi]
        scaled_actions = np.array(action_frame["policy_output"]["scaled_actions"], dtype=np.float32)
        target_positions = DEFAULT_JOINT_POSITIONS + scaled_actions
        target_tensor = torch.tensor(target_positions, dtype=torch.float32)

        # --- Initialize EMA filter state ---
        # On the real robot, the motor controller runs continuously, so at the
        # start of frame i its filter state ≈ steady-state torque from the
        # previous action. Approximate by computing PD torque from prev action
        # at the current state.
        if ema_alpha < 1.0 and fi > start_idx:
            prev_action_fi = max(start_idx, fi - 1 - delay_steps)
            prev_scaled = np.array(
                frames[prev_action_fi]["policy_output"]["scaled_actions"],
                dtype=np.float32,
            )
            prev_target = torch.tensor(
                DEFAULT_JOINT_POSITIONS + prev_scaled, dtype=torch.float32
            )
            cur_pos_t = torch.tensor(data.sensordata[:num_joints], dtype=torch.float32)
            cur_vel_t = torch.tensor(
                data.sensordata[num_joints:2 * num_joints], dtype=torch.float32
            )
            init_torque = joint_kp * (prev_target - cur_pos_t) + joint_kd * (-cur_vel_t)
            init_torque = torch.clip(init_torque, -effort_limits, effort_limits)
            filtered_torque = init_torque.numpy().copy()
        else:
            filtered_torque = np.zeros(num_joints, dtype=np.float32)

        for _ in range(physics_substeps):
            cur_pos = torch.tensor(data.sensordata[:num_joints], dtype=torch.float32)
            cur_vel = torch.tensor(data.sensordata[num_joints:2 * num_joints], dtype=torch.float32)
            raw_torque = joint_kp * (target_tensor - cur_pos) + joint_kd * (-cur_vel)
            raw_torque = torch.clip(raw_torque, -effort_limits, effort_limits)

            # Apply EMA filter (models motor controller torque smoothing)
            raw_np = raw_torque.numpy()
            filtered_torque = ema_alpha * raw_np + (1.0 - ema_alpha) * filtered_torque
            data.ctrl[:] = filtered_torque
            mujoco.mj_step(model, data)

        # --- Read sim state after one step ---
        sim_pos = data.sensordata[:num_joints].copy()
        sim_vel = data.sensordata[num_joints:2 * num_joints].copy()

        # --- Compare with real NEXT frame (what actually happened) ---
        real_next_pos = np.array(next_frame["robot_measured"]["joint_positions"], dtype=np.float32)
        real_next_vel = np.array(next_frame["robot_measured"]["joint_velocities"], dtype=np.float32)

        pos_error = sim_pos - real_next_pos
        vel_error = sim_vel - real_next_vel

        results.append({
            "frame_idx": fi,
            "step": frame["step"],
            "timestamp": frame["timestamp"],
            "target_pos": target_positions.copy(),
            "real_pos": real_pos.copy(),
            "real_next_pos": real_next_pos.copy(),
            "sim_pos": sim_pos.copy(),
            "real_vel": real_vel.copy(),
            "sim_vel": sim_vel.copy(),
            "pos_error": pos_error.copy(),
            "vel_error": vel_error.copy(),
        })

        if viewer is not None:
            viewer.sync()
            remaining = policy_dt - (time.perf_counter() - step_start)
            if remaining > 0:
                time.sleep(remaining)

        # Print periodic updates
        rel_idx = fi - start_idx
        if rel_idx < 3 or rel_idx % 50 == 0:
            rms = np.sqrt(np.mean(pos_error ** 2))
            worst_j = np.argmax(np.abs(pos_error))
            print(f"  Frame {fi:>4} (t={frame['timestamp']:.2f}s): "
                  f"RMS={rms:.4f} rad  "
                  f"worst={JOINT_NAMES[worst_j]} ({pos_error[worst_j]:+.3f})")

    return results


def replay_multistep(model, data, frames, start_idx, policy_dt, physics_dt,
                     joint_kp, joint_kd, effort_limits, delay_ms,
                     ema_tau_ms=0, pin_base=True, verbose=True, viewer=None):
    """Multi-step replay: joints evolve freely from physics (no per-frame reset).

    Applies logged actions sequentially and lets MuJoCo simulate the result.
    Error accumulates over time — this tests whether the actuator model
    produces trajectories that match reality over extended periods.

    Args:
        delay_ms: Actuator delay in milliseconds. 0 = no delay.
        ema_tau_ms: EMA torque filter time constant in ms. 0 = no filter.
        pin_base: If True, pin base pos/quat from log each frame (gantry mode).
                  If False, base evolves freely from physics (free-run mode).
        verbose: Print per-frame progress.

    Returns list of per-frame comparison dicts, plus base_z in each entry.
    """
    num_joints = len(JOINT_NAMES)
    physics_substeps = int(np.round(policy_dt / physics_dt))
    delay_steps = max(0, int(np.round(delay_ms / (policy_dt * 1000))))

    # EMA filter setup
    if ema_tau_ms > 0:
        tau_s = ema_tau_ms / 1000.0
        ema_alpha = 1.0 - np.exp(-physics_dt / tau_s)
    else:
        ema_alpha = 1.0

    # Initialize sim from the first frame
    frame0 = frames[start_idx]
    real_pos = np.array(frame0["robot_measured"]["joint_positions"], dtype=np.float32)
    real_vel = np.array(frame0["robot_measured"]["joint_velocities"], dtype=np.float32)
    real_quat = np.array(frame0["raw_observations"]["quaternion"], dtype=np.float32)

    if pin_base:
        data.qpos[0:2] = [0, 0]
        data.qpos[2] = 0.34
    else:
        # Place robot with feet on ground using forward kinematics
        foot_geom_ids = _get_foot_geom_ids(model)
        base_z = compute_base_height(model, data, real_pos, real_quat, foot_geom_ids)
        data.qpos[0:2] = [0, 0]
        data.qpos[2] = base_z

    data.qpos[3:7] = real_quat
    data.qpos[7:7 + num_joints] = real_pos
    data.qvel[:] = 0
    data.qvel[6:6 + num_joints] = real_vel
    mujoco.mj_forward(model, data)

    # Build action buffer from all frames (for delay lookback)
    all_actions = []
    for frame in frames:
        sa = np.array(frame["policy_output"]["scaled_actions"], dtype=np.float32)
        all_actions.append(DEFAULT_JOINT_POSITIONS + sa)

    # Initialize EMA filter state from first action
    filtered_torque = np.zeros(num_joints, dtype=np.float32)

    results = []

    for fi in range(start_idx, len(frames) - 1):
        step_start = time.perf_counter()
        frame = frames[fi]
        next_frame = frames[fi + 1]

        if pin_base:
            # Pin base to real pose (perfect gantry)
            real_quat = np.array(frame["raw_observations"]["quaternion"], dtype=np.float32)
            data.qpos[0:2] = [0, 0]
            data.qpos[2] = 0.34
            data.qpos[3:7] = real_quat
            data.qvel[0:6] = 0

        # --- Get the (possibly delayed) action ---
        action_idx = max(start_idx, fi - delay_steps)
        target_positions = all_actions[action_idx]
        target_tensor = torch.tensor(target_positions, dtype=torch.float32)

        # --- Step physics ---
        for _ in range(physics_substeps):
            cur_pos = torch.tensor(data.sensordata[:num_joints], dtype=torch.float32)
            cur_vel = torch.tensor(data.sensordata[num_joints:2 * num_joints], dtype=torch.float32)
            raw_torque = joint_kp * (target_tensor - cur_pos) + joint_kd * (-cur_vel)
            raw_torque = torch.clip(raw_torque, -effort_limits, effort_limits)

            raw_np = raw_torque.numpy()
            filtered_torque = ema_alpha * raw_np + (1.0 - ema_alpha) * filtered_torque
            data.ctrl[:] = filtered_torque
            mujoco.mj_step(model, data)

        # --- Read sim joint state ---
        sim_pos = data.sensordata[:num_joints].copy()
        sim_vel = data.sensordata[num_joints:2 * num_joints].copy()
        base_z = data.qpos[2]

        # --- Compare with real NEXT frame ---
        real_next_pos = np.array(next_frame["robot_measured"]["joint_positions"], dtype=np.float32)
        real_next_vel = np.array(next_frame["robot_measured"]["joint_velocities"], dtype=np.float32)

        pos_error = sim_pos - real_next_pos

        results.append({
            "frame_idx": fi,
            "step": frame["step"],
            "timestamp": frame["timestamp"],
            "target_pos": target_positions.copy(),
            "real_pos": np.array(frame["robot_measured"]["joint_positions"], dtype=np.float32),
            "real_next_pos": real_next_pos.copy(),
            "sim_pos": sim_pos.copy(),
            "real_vel": np.array(frame["robot_measured"]["joint_velocities"], dtype=np.float32),
            "sim_vel": sim_vel.copy(),
            "pos_error": pos_error.copy(),
            "vel_error": (sim_vel - real_next_vel).copy(),
            "base_z": base_z,
        })

        if viewer is not None:
            viewer.sync()
            remaining = policy_dt - (time.perf_counter() - step_start)
            if remaining > 0:
                time.sleep(remaining)

        if verbose:
            rel_idx = fi - start_idx
            if rel_idx < 3 or rel_idx % 50 == 0:
                rms = np.sqrt(np.mean(pos_error ** 2))
                worst_j = np.argmax(np.abs(pos_error))
                z_str = f"  z={base_z:.3f}" if not pin_base else ""
                print(f"  Frame {fi:>4} (t={frame['timestamp']:.2f}s): "
                      f"RMS={rms:.4f} rad  "
                      f"worst={JOINT_NAMES[worst_j]} ({pos_error[worst_j]:+.3f})"
                      f"{z_str}")

    return results


def run_delay_sweep(model, data, frames, start_idx, policy_dt, physics_dt,
                    joint_kp, joint_kd, effort_limits, action_scale):
    """Sweep delay values in single-step mode and report best match.

    For each delay, resets sim to real state each frame but applies an
    action from N steps earlier. This isolates the delay effect from
    error accumulation.
    """
    delays_ms = [0, 20, 40, 60, 80, 100, 120, 160]

    print(f"\n  {'=' * 60}")
    print(f"  Delay Sweep (single-step): testing {len(delays_ms)} delay values")
    print(f"  {'=' * 60}\n")

    sweep_results = []

    for delay_ms in delays_ms:
        delay_steps = max(0, int(np.round(delay_ms / (policy_dt * 1000))))

        results = replay(
            model, data, frames, start_idx, policy_dt, physics_dt,
            joint_kp, joint_kd, effort_limits, action_scale,
            delay_steps=delay_steps,
        )

        all_pos_err = np.array([r["pos_error"] for r in results])
        overall_rms = np.sqrt(np.mean(all_pos_err ** 2))

        per_joint_rms = np.sqrt(np.mean(all_pos_err ** 2, axis=0))

        # Tracking ratio
        all_target = np.array([r["target_pos"] for r in results])
        all_real = np.array([r["real_pos"] for r in results])
        all_sim = np.array([r["sim_pos"] for r in results])
        real_rms = np.sqrt(np.mean((all_target - all_real) ** 2))
        sim_rms = np.sqrt(np.mean((all_target - all_sim) ** 2))
        ratio = real_rms / sim_rms if sim_rms > 1e-6 else float("inf")

        sweep_results.append({
            "delay_ms": delay_ms,
            "delay_steps": delay_steps,
            "overall_rms": overall_rms,
            "per_joint_rms": per_joint_rms,
            "tracking_ratio": ratio,
        })

        print(f"  delay={delay_ms:>4}ms ({delay_steps} steps): "
              f"RMS={overall_rms:.4f} rad  ratio={ratio:.3f}x")

    # Find best delay
    best = min(sweep_results, key=lambda r: r["overall_rms"])
    print(f"\n  {'=' * 60}")
    print(f"  Best match: delay={best['delay_ms']}ms  "
          f"RMS={best['overall_rms']:.4f} rad  "
          f"ratio={best['tracking_ratio']:.3f}x")
    print(f"  {'=' * 60}")

    # Detailed per-joint comparison: best delay vs no delay
    no_delay = sweep_results[0]
    best_ms = best['delay_ms']
    print(f"\n  --- Per-Joint RMS: No Delay vs Best Delay ({best_ms}ms) ---")
    print(f"  {'Joint':<16} {'0ms':>8} {str(best_ms) + 'ms':>8} {'Improvement':>12}")
    print(f"  {'-' * 48}")
    for j in range(len(JOINT_NAMES)):
        r0 = no_delay["per_joint_rms"][j]
        rb = best["per_joint_rms"][j]
        imp = (r0 - rb) / r0 * 100 if r0 > 1e-6 else 0
        flag = " <--" if imp > 10 else ""
        print(f"  {JOINT_NAMES[j]:<16} {r0:>8.4f} {rb:>8.4f} {imp:>+10.1f}%{flag}")

    return sweep_results


def run_actuator_sweep(model, data, frames, start_idx, policy_dt, physics_dt,
                       joint_kp, joint_kd, action_scale):
    """Sweep EMA time constant and torque limit to find the actuator model
    that best matches real robot behavior.

    The real motor controller has:
      - PD position control (Kp=20, Kd=2)
      - Torque clamp (4 Nm on hardware, 6 Nm in training sim)
      - EMA low-pass filter on torque (alpha=0.145 at motor loop rate)

    This sweep tests combinations of:
      - EMA time constant tau: controls how sluggish the actuator responds
      - Torque limit: how much force the motor can apply

    By finding the (tau, torque_limit) that minimizes sim-vs-real error,
    we identify the actuator model to use in training.
    """
    num_joints = len(JOINT_NAMES)

    # Sweep parameters
    tau_values_ms = [0, 5, 10, 20, 30, 50, 80, 120]
    torque_limits = [4.0, 5.0, 6.0]

    print(f"\n  {'=' * 70}")
    print(f"  Actuator Model Sweep: {len(tau_values_ms)} tau x {len(torque_limits)} torque_limit "
          f"= {len(tau_values_ms) * len(torque_limits)} configs")
    print(f"  {'=' * 70}")
    print(f"\n  EMA tau values (ms): {tau_values_ms}")
    print(f"  Torque limits (Nm):  {torque_limits}\n")

    sweep_results = []

    for tlim in torque_limits:
        effort_lim = torch.full((num_joints,), tlim, dtype=torch.float32)
        for tau_ms in tau_values_ms:
            results = replay(
                model, data, frames, start_idx, policy_dt, physics_dt,
                joint_kp, joint_kd, effort_lim, action_scale,
                delay_steps=0, ema_tau_ms=tau_ms,
            )

            all_pos_err = np.array([r["pos_error"] for r in results])
            overall_rms = np.sqrt(np.mean(all_pos_err ** 2))

            per_joint_rms = np.sqrt(np.mean(all_pos_err ** 2, axis=0))

            # Tracking ratio: real_tracking / sim_tracking
            all_target = np.array([r["target_pos"] for r in results])
            all_real = np.array([r["real_pos"] for r in results])
            all_sim = np.array([r["sim_pos"] for r in results])
            real_rms = np.sqrt(np.mean((all_target - all_real) ** 2))
            sim_rms = np.sqrt(np.mean((all_target - all_sim) ** 2))
            ratio = real_rms / sim_rms if sim_rms > 1e-6 else float("inf")

            sweep_results.append({
                "tau_ms": tau_ms,
                "torque_limit": tlim,
                "overall_rms": overall_rms,
                "per_joint_rms": per_joint_rms,
                "tracking_ratio": ratio,
            })

            print(f"  torque={tlim:.0f}Nm  tau={tau_ms:>4}ms: "
                  f"RMS={overall_rms:.4f} rad  ratio={ratio:.3f}x")

        print()  # blank line between torque limit groups

    # --- Find best configuration ---
    best = min(sweep_results, key=lambda r: r["overall_rms"])
    # Also find config with ratio closest to 1.0 (sim matches real tracking)
    best_ratio = min(sweep_results, key=lambda r: abs(r["tracking_ratio"] - 1.0))

    print(f"  {'=' * 70}")
    print(f"  Best by RMS:   torque={best['torque_limit']:.0f}Nm  "
          f"tau={best['tau_ms']}ms  "
          f"RMS={best['overall_rms']:.4f} rad  "
          f"ratio={best['tracking_ratio']:.3f}x")
    print(f"  Best by ratio: torque={best_ratio['torque_limit']:.0f}Nm  "
          f"tau={best_ratio['tau_ms']}ms  "
          f"RMS={best_ratio['overall_rms']:.4f} rad  "
          f"ratio={best_ratio['tracking_ratio']:.3f}x")
    print(f"  {'=' * 70}")

    # --- Detailed comparison: baseline (no filter, 6Nm) vs best ---
    baseline = next(
        (r for r in sweep_results if r["tau_ms"] == 0 and r["torque_limit"] == 6.0),
        sweep_results[0],
    )

    print(f"\n  --- Per-Joint RMS: Baseline (no filter, 6Nm) vs Best ---")
    best_label = f"tau={best['tau_ms']}ms,{best['torque_limit']:.0f}Nm"
    print(f"  {'Joint':<16} {'Baseline':>10} {best_label:>16} {'Improvement':>12}")
    print(f"  {'-' * 58}")
    for j in range(num_joints):
        r0 = baseline["per_joint_rms"][j]
        rb = best["per_joint_rms"][j]
        imp = (r0 - rb) / r0 * 100 if r0 > 1e-6 else 0
        flag = " <--" if imp > 10 else ""
        print(f"  {JOINT_NAMES[j]:<16} {r0:>10.4f} {rb:>16.4f} {imp:>+10.1f}%{flag}")

    overall_imp = (baseline["overall_rms"] - best["overall_rms"]) / baseline["overall_rms"] * 100
    print(f"\n  Overall improvement: {overall_imp:+.1f}%")

    # --- Heatmap-style table ---
    print(f"\n  --- RMS Error Heatmap (rad) ---")
    col_label = "tau \\ tlim"
    header = f"  {col_label:<10}"
    for tlim in torque_limits:
        header += f"  {tlim:.0f}Nm"
    print(header)
    print(f"  {'-' * (10 + 8 * len(torque_limits))}")
    for tau_ms in tau_values_ms:
        row = f"  {tau_ms:>4}ms    "
        for tlim in torque_limits:
            r = next(x for x in sweep_results
                     if x["tau_ms"] == tau_ms and x["torque_limit"] == tlim)
            marker = " *" if r is best else "  "
            row += f"  {r['overall_rms']:.4f}{marker}"
        print(row)
    print(f"\n  * = best configuration")

    # --- Recommendation ---
    print(f"\n  {'=' * 70}")
    print(f"  RECOMMENDATION for training:")
    if best["tau_ms"] > 0:
        tau_s = best["tau_ms"] / 1000.0
        print(f"    - Add first-order actuator lag: tau = {best['tau_ms']}ms")
        print(f"      (In Isaac Lab: use DCMotorCfg or add custom actuator with")
        print(f"       time constant = {tau_s:.3f}s)")
    else:
        print(f"    - No actuator lag needed (instantaneous PD is best match)")
    print(f"    - Torque limit: {best['torque_limit']:.1f} Nm")
    if best["torque_limit"] != 6.0:
        print(f"      (Change from current 6.0 Nm to {best['torque_limit']:.1f} Nm in config)")
    ratio = best["tracking_ratio"]
    if ratio > 1.05:
        print(f"    - Tracking ratio = {ratio:.3f}x (sim still slightly faster than real)")
        print(f"      Consider also: motor armature, cable compliance, friction")
    elif ratio < 0.95:
        print(f"    - Tracking ratio = {ratio:.3f}x (sim slightly slower than real)")
    else:
        print(f"    - Tracking ratio = {ratio:.3f}x (good match!)")
    print(f"  {'=' * 70}")

    return sweep_results


FALL_DROP_THRESHOLD = 0.15  # fell if base z drops >15cm below initial


def run_policy_loop(model, data, frames, start_idx, physics_dt,
                    joint_kp, joint_kd, effort_limits, ema_tau_ms=0,
                    config_path=None, headless=False,
                    obs_noise=False, action_delay_ms=0, gamepad=False):
    """Run the trained policy in closed loop, starting from the log's initial state.

    Instead of replaying logged actions, this mode loads the actual neural
    network policy and lets it control the robot in MuJoCo. Only the initial
    state (joint positions, velocities, orientation) comes from the log.

    Args:
        config_path: Path to policy config YAML (policy checkpoint, action
                     scale, command velocity, observation size, etc.)
        effort_limits: Torque limits for PD control (from --torque-limit).
                       If None, uses the config's effort_limits.
        ema_tau_ms: EMA torque filter time constant in ms (0 = no filter).
        obs_noise: Add training-level uniform noise to observations.
        action_delay_ms: Delay policy output by this many ms (models
                         communication + motor lag).
        gamepad: Enable gamepad/keyboard control of command velocity.
    """
    from omegaconf import OmegaConf
    from berkeley_humanoid_lite_lowlevel.policy.rl_controller import RlController

    num_joints = len(JOINT_NAMES)

    # Load policy config
    with open(config_path) as f:
        cfg = OmegaConf.load(f)

    # Use the config's policy_dt (the rate the policy was trained at)
    policy_dt = cfg.policy_dt
    physics_substeps = int(np.round(policy_dt / physics_dt))

    # Initialize controller and load policy
    controller = RlController(cfg)
    controller.load_policy()

    # Control gains: use CLI overrides if provided, else config values
    if joint_kp is not None:
        ctrl_kp = joint_kp
    else:
        ctrl_kp = torch.tensor(cfg.joint_kp, dtype=torch.float32)
    if joint_kd is not None:
        ctrl_kd = joint_kd
    else:
        ctrl_kd = torch.tensor(cfg.joint_kd, dtype=torch.float32)
    if effort_limits is not None:
        ctrl_effort = effort_limits
    else:
        ctrl_effort = torch.tensor(cfg.effort_limits, dtype=torch.float32)

    # EMA filter setup
    if ema_tau_ms > 0:
        tau_s = ema_tau_ms / 1000.0
        ema_alpha = 1.0 - np.exp(-physics_dt / tau_s)
    else:
        ema_alpha = 1.0

    # Action delay: buffer past actions, apply the delayed one
    from collections import deque
    action_delay_steps = max(0, int(np.round(action_delay_ms / (policy_dt * 1000))))

    # Sensor data offsets (MuJoCo sensordata layout: pos, vel, force, then IMU)
    sensordata_dof_size = 3 * num_joints
    action_indices = list(cfg.action_indices)
    num_actions = len(action_indices)

    # Initialize MuJoCo from log's first frame
    frame0 = frames[start_idx]
    real_pos = np.array(frame0["robot_measured"]["joint_positions"], dtype=np.float32)
    real_vel = np.array(frame0["robot_measured"]["joint_velocities"], dtype=np.float32)
    real_quat = np.array(frame0["raw_observations"]["quaternion"], dtype=np.float32)

    # Command velocity from log's first frame (used as initial / fallback)
    command_velocity = np.array(
        frame0["policy_input"]["command_velocity"], dtype=np.float32
    )

    # Gamepad controller for interactive command velocity
    gamepad_controller = None
    if gamepad:
        if GAMEPAD_AVAILABLE:
            gamepad_controller = Se2Gamepad()
            gamepad_controller.run()
            print(f"  Gamepad: ENABLED")
            print(f"    Left Stick Y  : Forward/Backward")
            print(f"    Left Stick X  : Turn Left/Right")
            print(f"    Right Stick X : Strafe Left/Right")
        else:
            print(f"  WARNING: Gamepad requested but Se2Gamepad not available.")
            print(f"           Install berkeley_humanoid_lite_lowlevel or check imports.")

    foot_geom_ids = _get_foot_geom_ids(model)
    base_z = compute_base_height(model, data, real_pos, real_quat, foot_geom_ids)
    data.qpos[0:2] = [0, 0]
    data.qpos[2] = base_z
    data.qpos[3:7] = real_quat
    data.qpos[7:7 + num_joints] = real_pos
    data.qvel[:] = 0
    data.qvel[6:6 + num_joints] = real_vel
    mujoco.mj_forward(model, data)

    print(f"\n  Policy-in-the-loop mode")
    print(f"  Config: {config_path}")
    print(f"  Policy: {cfg.policy_checkpoint_path}")
    print(f"  Policy rate: {1/policy_dt:.0f} Hz ({physics_substeps} substeps)")
    print(f"  Initial state from log frame {start_idx}")
    print(f"  base_z = {base_z:.3f}m")
    print(f"  Command velocity (from log): {command_velocity.tolist()}")
    tlim_val = ctrl_effort[0].item()
    print(f"  Torque limit: {tlim_val:.1f} Nm")
    if ema_tau_ms > 0:
        print(f"  EMA torque filter: tau={ema_tau_ms:.0f}ms")
    if obs_noise:
        print(f"  Observation noise: ON (training levels)")
        print(f"    ang_vel ±0.3, joint_pos ±0.05, joint_vel ±2.0")
    if action_delay_steps > 0:
        print(f"  Action delay: {action_delay_ms:.0f}ms ({action_delay_steps} steps)")

    # Launch viewer
    viewer = None
    if not headless:
        viewer = mujoco.viewer.launch_passive(model, data)

    filtered_torque = np.zeros(num_joints, dtype=np.float32)
    mode = 3.0  # RL control mode
    step = 0
    initial_z = base_z

    # Action delay buffer: filled with default positions (standing still)
    default_jp = np.array(cfg.default_joint_positions, dtype=np.float32)
    default_targets = np.zeros(num_joints, dtype=np.float32)
    default_targets[action_indices] = default_jp[action_indices] if len(default_jp) == num_joints else default_jp
    action_buffer = deque(
        [default_targets.copy() for _ in range(action_delay_steps + 1)],
        maxlen=action_delay_steps + 1,
    )

    print(f"\n  Running policy... (Ctrl+C or close viewer to stop)\n")

    try:
        while True:
            step_start = time.perf_counter()

            if viewer is not None and not viewer.is_running():
                break

            # --- Construct observation from MuJoCo state ---
            sim_quat = data.sensordata[
                sensordata_dof_size:sensordata_dof_size + 4
            ].copy().astype(np.float32)
            sim_ang_vel = data.sensordata[
                sensordata_dof_size + 4:sensordata_dof_size + 7
            ].copy().astype(np.float32)
            sim_joint_pos = data.sensordata[:num_joints].copy().astype(np.float32)
            sim_joint_vel = data.sensordata[
                num_joints:2 * num_joints
            ].copy().astype(np.float32)

            # Inject observation noise (matches training domain randomization)
            if obs_noise:
                sim_ang_vel += np.random.uniform(-0.3, 0.3, size=3).astype(np.float32)
                sim_joint_pos += np.random.uniform(-0.05, 0.05, size=num_joints).astype(np.float32)
                sim_joint_vel += np.random.uniform(-2.0, 2.0, size=num_joints).astype(np.float32)

            # Update command velocity from gamepad if active
            # Scale gamepad [-1, 1] to training velocity ranges
            if gamepad_controller is not None:
                command_velocity[0] = gamepad_controller.commands["velocity_x"] * 0.5   # ±0.5 m/s
                command_velocity[1] = gamepad_controller.commands["velocity_y"] * 0.25  # ±0.25 m/s
                command_velocity[2] = gamepad_controller.commands["velocity_yaw"] * 1.0 # ±1.0 rad/s

            obs = np.concatenate([
                sim_quat,                                          # [0:4]
                sim_ang_vel,                                       # [4:7]
                sim_joint_pos[action_indices],                     # joint pos
                sim_joint_vel[action_indices],                     # joint vel
                np.array([mode], dtype=np.float32),                # mode
                command_velocity,                                   # cmd vel
            ])

            # --- Run policy ---
            actions = controller.update(obs)  # absolute joint position targets

            # --- Action delay: push new action, use delayed one ---
            new_targets = np.zeros(num_joints, dtype=np.float32)
            new_targets[action_indices] = actions
            action_buffer.append(new_targets)
            target_positions = torch.tensor(
                action_buffer[0], dtype=torch.float32
            )

            for _ in range(physics_substeps):
                cur_pos = torch.tensor(
                    data.sensordata[:num_joints], dtype=torch.float32
                )
                cur_vel = torch.tensor(
                    data.sensordata[num_joints:2 * num_joints],
                    dtype=torch.float32,
                )
                raw_torque = ctrl_kp * (target_positions - cur_pos) + ctrl_kd * (-cur_vel)
                raw_torque = torch.clip(raw_torque, -ctrl_effort, ctrl_effort)

                raw_np = raw_torque.numpy()
                filtered_torque = ema_alpha * raw_np + (1.0 - ema_alpha) * filtered_torque
                data.ctrl[:] = filtered_torque
                mujoco.mj_step(model, data)

            # --- Status ---
            cur_z = data.qpos[2]

            if step < 3 or step % 100 == 0:
                cmd_str = ""
                if gamepad_controller is not None:
                    cmd_str = (f"  cmd=[{command_velocity[0]:+.2f}, "
                               f"{command_velocity[1]:+.2f}, "
                               f"{command_velocity[2]:+.2f}]")
                print(f"  Step {step:>5}: z={cur_z:.3f}m  "
                      f"L_knee={sim_joint_pos[3]:.3f}  "
                      f"R_knee={sim_joint_pos[9]:.3f}{cmd_str}")

            # Check for fall
            if cur_z < initial_z - FALL_DROP_THRESHOLD:
                print(f"\n  FELL at step {step} (z={cur_z:.3f}m, "
                      f"dropped {initial_z - cur_z:.3f}m from initial {initial_z:.3f}m)")
                break

            if viewer is not None:
                viewer.sync()

            # Real-time pacing
            remaining = policy_dt - (time.perf_counter() - step_start)
            if remaining > 0:
                time.sleep(remaining)

            step += 1

    except KeyboardInterrupt:
        print(f"\n  Stopped at step {step}")

    if gamepad_controller is not None:
        gamepad_controller.stop()

    if viewer is not None:
        if viewer.is_running():
            print(f"\n  Policy stopped after {step} steps. Close viewer to exit.")
            while viewer.is_running():
                time.sleep(0.1)
        viewer.close()

    print(f"  Total: {step} steps ({step * policy_dt:.1f}s)")


def run_free_sweep(model, data, frames, start_idx, policy_dt, physics_dt,
                   joint_kp, joint_kd, action_scale):
    """Sweep actuator configs in free-run mode (no base pin, no state reset).

    The sim starts from the real robot's initial state and runs freely with
    logged actions. This is the hardest test: any actuator model mismatch
    causes the sim trajectory to diverge from reality over time.

    Reports RMS error at milestones (100, 250, 500 steps), whether the
    robot falls, and which config stays closest to the real trajectory.
    """
    num_joints = len(JOINT_NAMES)
    total_frames = len(frames) - start_idx - 1

    # Configs to test: focused around the promising single-step results
    configs = [
        {"torque": 4.0, "tau": 0,  "label": "4Nm, no filter"},
        {"torque": 4.0, "tau": 10, "label": "4Nm, tau=10ms"},
        {"torque": 4.0, "tau": 30, "label": "4Nm, tau=30ms"},
        {"torque": 5.0, "tau": 0,  "label": "5Nm, no filter"},
        {"torque": 5.0, "tau": 10, "label": "5Nm, tau=10ms"},
        {"torque": 5.0, "tau": 30, "label": "5Nm, tau=30ms"},
        {"torque": 6.0, "tau": 0,  "label": "6Nm, no filter"},
        {"torque": 6.0, "tau": 10, "label": "6Nm, tau=10ms"},
        {"torque": 6.0, "tau": 30, "label": "6Nm, tau=30ms"},
    ]

    # Milestones for reporting
    milestones = [s for s in [50, 100, 250, 500, 750] if s <= total_frames]
    if total_frames not in milestones:
        milestones.append(total_frames)

    print(f"\n  {'=' * 74}")
    print(f"  Free-Run Actuator Sweep: {len(configs)} configs, "
          f"{total_frames} frames ({total_frames * policy_dt:.1f}s)")
    print(f"  Robot runs freely (no base pin, no state reset)")
    print(f"  {'=' * 74}\n")

    sweep_results = []

    for cfg in configs:
        effort_lim = torch.full((num_joints,), cfg["torque"], dtype=torch.float32)

        print(f"  --- {cfg['label']} ---")
        results = replay_multistep(
            model, data, frames, start_idx, policy_dt, physics_dt,
            joint_kp, joint_kd, effort_lim, delay_ms=0,
            ema_tau_ms=cfg["tau"], pin_base=False, verbose=False,
        )

        all_pos_err = np.array([r["pos_error"] for r in results])
        all_base_z = np.array([r["base_z"] for r in results])

        # Overall RMS
        overall_rms = np.sqrt(np.mean(all_pos_err ** 2))

        # Detect fall: base z drops >15cm below initial height
        initial_z = all_base_z[0] if len(all_base_z) > 0 else 0
        fell_at = None
        for i, z in enumerate(all_base_z):
            if z < initial_z - FALL_DROP_THRESHOLD:
                fell_at = i
                break

        # RMS at milestones (windowed: RMS of errors in [0, milestone])
        milestone_rms = {}
        for ms in milestones:
            idx = min(ms, len(all_pos_err))
            if idx > 0:
                milestone_rms[ms] = np.sqrt(np.mean(all_pos_err[:idx] ** 2))

        # Frames until RMS exceeds thresholds
        cumulative_rms = np.sqrt(
            np.cumsum(all_pos_err ** 2, axis=0).mean(axis=1)
            / np.arange(1, len(all_pos_err) + 1)
        )

        diverge_steps = {}
        for thresh in [0.1, 0.2, 0.5, 1.0]:
            exceeds = np.where(cumulative_rms > thresh)[0]
            diverge_steps[thresh] = int(exceeds[0]) if len(exceeds) > 0 else None

        # z delta: how much did base drop from initial
        min_z = np.min(all_base_z) if len(all_base_z) > 0 else 0
        z_drop = initial_z - min_z

        entry = {
            "label": cfg["label"],
            "torque": cfg["torque"],
            "tau": cfg["tau"],
            "overall_rms": overall_rms,
            "milestone_rms": milestone_rms,
            "fell_at": fell_at,
            "diverge_steps": diverge_steps,
            "initial_z": initial_z,
            "final_base_z": all_base_z[-1] if len(all_base_z) > 0 else 0,
            "z_drop": z_drop,
        }
        sweep_results.append(entry)

        # Print compact summary for this config
        fall_str = f"FELL at step {fell_at}" if fell_at is not None else "stayed up"
        print(f"    RMS={overall_rms:.4f} rad  z: {initial_z:.3f} -> {entry['final_base_z']:.3f}m "
              f"(drop={z_drop:.3f}m)  {fall_str}")
        ms_parts = [f"{ms}:{milestone_rms[ms]:.3f}" for ms in milestones if ms in milestone_rms]
        print(f"    RMS@step: {', '.join(ms_parts)}")
        print()

    # --- Summary table ---
    print(f"  {'=' * 78}")
    print(f"  FREE-RUN SWEEP SUMMARY")
    print(f"  {'=' * 78}")

    # Header
    ms_headers = "".join(f"  @{ms:>4}" for ms in milestones)
    print(f"\n  {'Config':<20} {'RMS':>6} {'z_drop':>6} {'Status':<14}{ms_headers}")
    print(f"  {'-' * (20 + 6 + 6 + 14 + 7 * len(milestones))}")

    for r in sweep_results:
        status = f"FELL@{r['fell_at']}" if r['fell_at'] is not None else "upright"
        ms_vals = ""
        for ms in milestones:
            if ms in r["milestone_rms"]:
                ms_vals += f"  {r['milestone_rms'][ms]:>5.3f}"
            else:
                ms_vals += f"  {'N/A':>5}"
        print(f"  {r['label']:<20} {r['overall_rms']:>6.4f} {r['z_drop']:>6.3f} "
              f"{status:<14}{ms_vals}")

    # Best config
    best = min(sweep_results, key=lambda r: r["overall_rms"])
    print(f"\n  Best overall: {best['label']}  RMS={best['overall_rms']:.4f} rad")

    # Best among configs where robot didn't fall
    upright = [r for r in sweep_results if r["fell_at"] is None]
    if upright:
        best_upright = min(upright, key=lambda r: r["overall_rms"])
        print(f"  Best (stayed upright): {best_upright['label']}  "
              f"RMS={best_upright['overall_rms']:.4f} rad")
    elif all(r["z_drop"] < 0.10 for r in sweep_results):
        print(f"  Note: All configs stayed roughly upright (z_drop < 10cm)")
    else:
        print(f"  WARNING: Robot fell in ALL configs!")

    # Divergence comparison
    print(f"\n  --- Steps until cumulative RMS exceeds threshold ---")
    print(f"  {'Config':<20} {'0.1rad':>8} {'0.2rad':>8} {'0.5rad':>8} {'1.0rad':>8}")
    print(f"  {'-' * 52}")
    for r in sweep_results:
        ds = r["diverge_steps"]
        vals = []
        for thresh in [0.1, 0.2, 0.5, 1.0]:
            v = ds.get(thresh)
            vals.append(f"{v:>8}" if v is not None else f"{'never':>8}")
        print(f"  {r['label']:<20} {''.join(vals)}")

    print(f"\n  {'=' * 78}")

    return sweep_results


def print_summary(results):
    """Print summary statistics of sim-vs-real comparison."""
    if not results:
        print("  No results to summarize.")
        return

    num_joints = len(JOINT_NAMES)
    num_frames = len(results)

    # Collect all errors into arrays
    all_pos_err = np.array([r["pos_error"] for r in results])  # (N, 12)
    all_vel_err = np.array([r["vel_error"] for r in results])

    # Also compute target-vs-real and target-vs-sim tracking errors
    all_target = np.array([r["target_pos"] for r in results])
    all_real = np.array([r["real_pos"] for r in results])
    all_sim = np.array([r["sim_pos"] for r in results])
    real_tracking_err = all_target - all_real  # how well real robot tracks targets
    sim_tracking_err = all_target - all_sim    # how well sim tracks targets

    print(f"\n{'=' * 80}")
    print(f"  SUMMARY: {num_frames} frames replayed")
    print(f"{'=' * 80}")

    # Per-joint position error (sim vs real)
    print(f"\n  --- Sim vs Real: Position Error (rad) ---")
    print(f"  {'Joint':<16} {'Mean':>8} {'RMS':>8} {'Max':>8} {'Std':>8}")
    print(f"  {'-' * 52}")
    for j in range(num_joints):
        errs = all_pos_err[:, j]
        print(f"  {JOINT_NAMES[j]:<16} {np.mean(errs):>+8.4f} {np.sqrt(np.mean(errs**2)):>8.4f} "
              f"{errs[np.argmax(np.abs(errs))]:>+8.4f} {np.std(errs):>8.4f}")

    overall_rms = np.sqrt(np.mean(all_pos_err ** 2))
    print(f"\n  Overall position RMS (sim vs real): {overall_rms:.4f} rad")

    # Per-joint tracking error comparison
    print(f"\n  --- Target Tracking: RMS Error (rad) ---")
    print(f"  {'Joint':<16} {'Real':>8} {'Sim':>8} {'Ratio':>8}  {'Note'}")
    print(f"  {'-' * 60}")
    for j in range(num_joints):
        real_rms = np.sqrt(np.mean(real_tracking_err[:, j] ** 2))
        sim_rms = np.sqrt(np.mean(sim_tracking_err[:, j] ** 2))
        ratio = real_rms / sim_rms if sim_rms > 1e-6 else float("inf")
        note = ""
        if ratio > 5:
            note = "<-- real motor much slower"
        elif ratio > 2:
            note = "<-- significant gap"
        print(f"  {JOINT_NAMES[j]:<16} {real_rms:>8.4f} {sim_rms:>8.4f} {ratio:>8.1f}x  {note}")

    # When does divergence become large?
    cumulative_rms = [np.sqrt(np.mean(r["pos_error"] ** 2)) for r in results]
    for threshold in [0.05, 0.10, 0.20, 0.50]:
        frames_above = [i for i, rms in enumerate(cumulative_rms) if rms > threshold]
        if frames_above:
            first = frames_above[0]
            t = results[first]["timestamp"]
            print(f"\n  Position RMS first exceeds {threshold:.2f} rad at frame {first} (t={t:.2f}s)")
        else:
            print(f"\n  Position RMS never exceeds {threshold:.2f} rad")


def save_csv(results, csv_path):
    """Save per-frame comparison data to CSV."""
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)

        # Header
        header = ["frame", "step", "timestamp"]
        for name in JOINT_NAMES:
            header.extend([
                f"{name}_target", f"{name}_real", f"{name}_sim",
                f"{name}_pos_err", f"{name}_real_vel", f"{name}_sim_vel",
            ])
        writer.writerow(header)

        # Data rows
        for r in results:
            row = [r["frame_idx"], r["step"], f"{r['timestamp']:.6f}"]
            for j in range(len(JOINT_NAMES)):
                row.extend([
                    f"{r['target_pos'][j]:.6f}",
                    f"{r['real_pos'][j]:.6f}",
                    f"{r['sim_pos'][j]:.6f}",
                    f"{r['pos_error'][j]:.6f}",
                    f"{r['real_vel'][j]:.6f}",
                    f"{r['sim_vel'][j]:.6f}",
                ])
            writer.writerow(row)

    print(f"\n  Saved comparison data to {csv_path} ({len(results)} rows)")


def main():
    parser = argparse.ArgumentParser(
        description="Single-step sim-to-real comparison and visual replay of logged robot data")
    parser.add_argument("--log", type=str, required=True,
                        help="Path to log JSON from run_locomotion.py")
    parser.add_argument("--start-frame", type=int, default=-1,
                        help="Frame to start replay from (-1 = auto-detect RL_RUNNING)")
    parser.add_argument("--torque-limit", type=float, default=None,
                        help="Override torque limit (e.g. 4.0 for real hardware value)")
    parser.add_argument("--csv", type=str, default=None,
                        help="Save comparison data to CSV file")
    parser.add_argument("--headless", action="store_true",
                        help="No MuJoCo viewer, just print stats")
    parser.add_argument("--kp", type=float, default=20.0,
                        help="Position gain Kp (default: 20.0)")
    parser.add_argument("--kd", type=float, default=2.0,
                        help="Velocity gain Kd (default: 2.0)")
    parser.add_argument("--visual", action="store_true",
                        help="Visual replay: play back logged joint positions "
                             "in MuJoCo viewer (kinematic, no physics)")
    parser.add_argument("--delay", type=float, default=None,
                        help="Actuator delay in ms for multi-step replay. "
                             "Runs sim freely (no per-frame state reset) with "
                             "base pinned and delayed actions. (e.g. --delay 80)")
    parser.add_argument("--delay-sweep", action="store_true",
                        help="Sweep actuator delays (0-160ms) to find best "
                             "match with real robot. Implies --headless.")
    parser.add_argument("--actuator-sweep", action="store_true",
                        help="Sweep actuator models (EMA filter tau + torque "
                             "limit) to find the sim actuator config that best "
                             "matches real robot. Implies --headless.")
    parser.add_argument("--ema-tau", type=float, default=0,
                        help="EMA torque filter time constant in ms "
                             "(0 = no filter, default). Models the motor "
                             "controller's torque smoothing.")
    parser.add_argument("--free-sweep", action="store_true",
                        help="Sweep actuator configs in free-run mode: "
                             "no base pin, no state reset. Tests which actuator "
                             "model keeps sim closest to real over long runs.")
    parser.add_argument("--free", action="store_true",
                        help="Free-run mode for --delay: disable base pinning, "
                             "let the full robot evolve from physics.")
    parser.add_argument("--policy-run", action="store_true",
                        help="Policy-in-the-loop: load the trained policy and "
                             "run it in closed loop. Initial state from log, "
                             "then the neural network controls the robot. "
                             "Requires --config.")
    parser.add_argument("--config", type=str, default=None,
                        help="Path to policy config YAML (for --policy-run). "
                             "E.g. configs/policy_latest_hard_v1.yaml")
    parser.add_argument("--obs-noise", action="store_true",
                        help="Add training-level observation noise to "
                             "--policy-run (uniform: ang_vel ±0.3, "
                             "joint_pos ±0.05, joint_vel ±2.0)")
    parser.add_argument("--action-delay", type=float, default=0,
                        help="Delay policy actions by N ms in --policy-run "
                             "(models communication + motor lag). "
                             "Real robot has ~60-120ms. E.g. --action-delay 80")
    parser.add_argument("--gamepad", action="store_true",
                        help="Enable gamepad/keyboard control of command "
                             "velocity in --policy-run mode. Left stick Y = "
                             "forward/back, Left stick X = turn, Right stick "
                             "X = strafe.")
    args = parser.parse_args()

    # Load log
    metadata, frames = load_log(args.log)
    policy_dt = metadata["policy_dt"]
    # Use a reasonable physics dt for MuJoCo (0.0005s = 2kHz, matching hardware)
    physics_dt = 0.0005
    num_joints = len(JOINT_NAMES)

    print(f"\n{'=' * 60}")
    print(f"  Sim-to-Real Replay")
    print(f"  Log: {args.log}")
    print(f"  Frames: {len(frames)} ({metadata['duration_seconds']:.1f}s)")
    print(f"  Policy dt: {policy_dt}s ({1/policy_dt:.0f} Hz)")
    print(f"  Physics dt: {physics_dt}s ({1/physics_dt:.0f} Hz)")
    print(f"{'=' * 60}")

    # Determine start frame
    if args.start_frame >= 0:
        start_idx = args.start_frame
        print(f"\n  Starting from frame {start_idx} (user-specified)")
    else:
        start_idx = find_rl_running_start(frames)
        print(f"\n  Auto-detected RL_RUNNING start at frame {start_idx}")

    if start_idx >= len(frames):
        print(f"  ERROR: start frame {start_idx} >= total frames {len(frames)}")
        sys.exit(1)

    # Control gains
    joint_kp = torch.full((num_joints,), args.kp, dtype=torch.float32)
    joint_kd = torch.full((num_joints,), args.kd, dtype=torch.float32)

    # Torque limit: use override, or fall back to 6 Nm (sim default)
    tlim = args.torque_limit if args.torque_limit is not None else 6.0
    effort_limits = torch.full((num_joints,), tlim, dtype=torch.float32)
    print(f"  Gains: Kp={args.kp}, Kd={args.kd}, torque_limit={tlim} Nm")

    # Action scale from metadata (always 0.25)
    action_scale = 0.25

    # Setup MuJoCo
    model, data = setup_mujoco(physics_dt)

    if args.policy_run:
        # Policy-in-the-loop: neural network controls the robot
        if args.config is None:
            print("  ERROR: --policy-run requires --config <path/to/policy.yaml>")
            sys.exit(1)
        run_policy_loop(
            model, data, frames, start_idx, physics_dt,
            joint_kp, joint_kd, effort_limits, ema_tau_ms=args.ema_tau,
            config_path=args.config, headless=args.headless,
            obs_noise=args.obs_noise, action_delay_ms=args.action_delay,
            gamepad=args.gamepad,
        )

    elif args.visual:
        # Visual replay mode: kinematic playback of logged joint positions
        run_visual_replay(model, data, frames, start_idx, policy_dt)

    elif args.free_sweep:
        # Free-run sweep: no base pin, no state reset
        run_free_sweep(
            model, data, frames, start_idx, policy_dt, physics_dt,
            joint_kp, joint_kd, action_scale,
        )

    elif args.actuator_sweep:
        # Sweep EMA tau + torque limit combinations (single-step)
        run_actuator_sweep(
            model, data, frames, start_idx, policy_dt, physics_dt,
            joint_kp, joint_kd, action_scale,
        )

    elif args.delay_sweep:
        # Sweep delay values to find best match
        run_delay_sweep(
            model, data, frames, start_idx, policy_dt, physics_dt,
            joint_kp, joint_kd, effort_limits, action_scale,
        )

    elif args.delay is not None:
        # Multi-step replay with specific actuator delay
        delay_ms = args.delay
        delay_steps = max(0, int(np.round(delay_ms / (policy_dt * 1000))))
        pin_base = not args.free
        mode_str = "free-run" if not pin_base else "base pinned"
        print(f"  Actuator delay: {delay_ms:.0f}ms ({delay_steps} policy steps)")
        if args.ema_tau > 0:
            print(f"  EMA torque filter: tau={args.ema_tau:.0f}ms")
        print(f"  Mode: {mode_str}")

        viewer = None
        if not args.headless:
            viewer = mujoco.viewer.launch_passive(model, data)

        print(f"\n  Multi-step replay: {len(frames) - start_idx} frames "
              f"(joints evolve freely, {mode_str})...\n")
        results = replay_multistep(
            model, data, frames, start_idx, policy_dt, physics_dt,
            joint_kp, joint_kd, effort_limits, delay_ms,
            ema_tau_ms=args.ema_tau, pin_base=pin_base, viewer=viewer,
        )

        print_summary(results)

        if args.csv:
            save_csv(results, args.csv)

        if viewer is not None and viewer.is_running():
            print("\n  Replay complete. Close the viewer window to exit.")
            while viewer.is_running():
                time.sleep(0.1)
            viewer.close()

    else:
        # Single-step comparison mode (default)
        viewer = None
        if not args.headless:
            viewer = mujoco.viewer.launch_passive(model, data)

        if args.ema_tau > 0:
            print(f"  EMA torque filter: tau={args.ema_tau:.0f}ms")
        print(f"\n  Replaying {len(frames) - start_idx} frames...\n")
        results = replay(
            model, data, frames, start_idx, policy_dt, physics_dt,
            joint_kp, joint_kd, effort_limits, action_scale,
            ema_tau_ms=args.ema_tau, viewer=viewer,
        )

        print_summary(results)

        if args.csv:
            save_csv(results, args.csv)

        if viewer is not None and viewer.is_running():
            print("\n  Replay complete. Close the viewer window to exit.")
            while viewer.is_running():
                time.sleep(0.1)
            viewer.close()


if __name__ == "__main__":
    main()
