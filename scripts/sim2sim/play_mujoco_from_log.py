# Copyright (c) 2025, The Berkeley Humanoid Lite Project Developers.

"""
Sim2Sim: Start simulation from logged robot state

Loads a log file captured by run_locomotion.py --log, sets the MuJoCo
simulation to match the robot's state at the start of RL_RUNNING, then
runs the policy to observe what the robot would do in simulation.

Usage (run from project root):
    uv run scripts/sim2sim/play_mujoco_from_log.py \
        --log source/berkeley_humanoid_lite_lowlevel/test_run_003.json \
        --config configs/policy_biped_50hz.yaml

    # Headless mode (no viewer, just prints policy outputs):
    uv run scripts/sim2sim/play_mujoco_from_log.py \
        --log test_run_003.json --headless --max-steps 50

    # Start from a different frame:
    uv run scripts/sim2sim/play_mujoco_from_log.py \
        --log test_run_003.json --frame 10
"""

import json
import argparse
import os
import sys
import time

import numpy as np
import torch

from berkeley_humanoid_lite_lowlevel.policy.rl_controller import RlController
from berkeley_humanoid_lite_lowlevel.policy.config import Cfg


JOINT_NAMES = [
    "L_hip_roll", "L_hip_yaw", "L_hip_pitch",
    "L_knee", "L_ankle_pitch", "L_ankle_roll",
    "R_hip_roll", "R_hip_yaw", "R_hip_pitch",
    "R_knee", "R_ankle_pitch", "R_ankle_roll",
]


def load_log_frame(log_path, frame_idx):
    """Load a specific frame from a run_locomotion.py log file."""
    with open(log_path) as f:
        log_data = json.load(f)

    frames = log_data["frames"]
    if frame_idx >= len(frames):
        raise ValueError(f"Frame {frame_idx} out of range (total: {len(frames)})")

    frame = frames[frame_idx]
    metadata = log_data["metadata"]

    initial_state = {
        "joint_positions": np.array(frame["robot_measured"]["joint_positions"], dtype=np.float32),
        "joint_velocities": np.array(frame["robot_measured"]["joint_velocities"], dtype=np.float32),
        "quaternion": np.array(frame["raw_observations"]["quaternion"], dtype=np.float32),
        "prev_actions": np.array(frame["policy_input"]["prev_actions"], dtype=np.float32),
        "step": frame["step"],
        "timestamp": frame["timestamp"],
    }

    return initial_state, metadata


def print_initial_state(state, default_pos):
    """Print initial joint state comparison."""
    pos = state["joint_positions"]

    print(f"\n  {'Joint':<16} {'Logged':>8} {'Default':>8} {'Delta':>8}")
    print(f"  {'-' * 44}")
    for i, name in enumerate(JOINT_NAMES):
        d = pos[i] - default_pos[i]
        flag = " <--" if abs(d) > 0.3 else ""
        print(f"  {name:<16} {pos[i]:>+8.3f} {default_pos[i]:>+8.3f} {d:>+8.3f}{flag}")
    print()


def setup_mesh_symlinks():
    """Create mesh symlinks for MJCF loading if they don't exist."""
    project_root = os.getcwd()
    mjcf_dir = os.path.join(
        project_root,
        "source", "berkeley_humanoid_lite_assets", "data", "robots",
        "berkeley_humanoid", "berkeley_humanoid_lite", "mjcf"
    )
    meshes_dir = os.path.join(mjcf_dir, "..", "meshes")
    assets_merged = os.path.join(mjcf_dir, "assets", "merged")

    if not os.path.exists(assets_merged) and os.path.exists(meshes_dir):
        os.makedirs(os.path.join(mjcf_dir, "assets"), exist_ok=True)
        os.symlink(os.path.abspath(meshes_dir), assets_merged)
        print(f"  Created mesh symlink: assets/merged -> meshes/")

    return os.path.exists(assets_merged)


def run_headless(cfg, controller, state, max_steps):
    """Run policy forward passes without MuJoCo physics.

    Shows what the policy outputs given the logged state, using simple
    position-update integration (no physics). Useful for checking if
    the policy itself drives the robot away from the initial position.
    """
    print("=== Headless Mode (policy only, no physics) ===\n")

    initial_pos = state["joint_positions"]
    initial_vel = state["joint_velocities"]
    initial_quat = state["quaternion"]
    default_pos = np.array(cfg.default_joint_positions, dtype=np.float32)
    n_steps = max_steps if max_steps > 0 else 20

    current_pos = initial_pos.copy()
    current_vel = initial_vel.copy()

    for step in range(n_steps):
        # Build observation: [quat(4), ang_vel(3), joint_pos(12), joint_vel(12), mode+cmd(4)]
        obs = np.concatenate([
            initial_quat,
            np.zeros(3, dtype=np.float32),
            current_pos,
            current_vel,
            np.array([3.0, 0.0, 0.0, 0.0], dtype=np.float32),
        ])

        actions = controller.update(obs)

        if step < 5 or step % 10 == 0:
            print(f"  Step {step}:")
            print(f"    {'Joint':<16} {'Current':>8} {'Action':>8} {'Delta':>8}")
            print(f"    {'-' * 40}")
            for i, name in enumerate(JOINT_NAMES):
                d = actions[i] - current_pos[i]
                flag = " !!!" if abs(d) > 0.1 else ""
                print(f"    {name:<16} {current_pos[i]:>+8.3f} {actions[i]:>+8.3f} {d:>+8.3f}{flag}")

        # Simple integration: assume robot reaches action target instantly
        current_vel = (actions - current_pos) / cfg.policy_dt
        current_pos = actions.copy()

    print(f"\n  === After {n_steps} steps: Total drift from initial ===")
    print(f"  {'Joint':<16} {'Initial':>8} {'Final':>8} {'Drift':>8}")
    print(f"  {'-' * 40}")
    for i, name in enumerate(JOINT_NAMES):
        d = current_pos[i] - initial_pos[i]
        flag = " !!!" if abs(d) > 0.2 else ""
        print(f"  {name:<16} {initial_pos[i]:>+8.3f} {current_pos[i]:>+8.3f} {d:>+8.3f}{flag}")


def run_mujoco(cfg, controller, state, max_steps):
    """Run full MuJoCo physics simulation starting from logged state."""
    import mujoco
    import mujoco.viewer

    initial_pos = state["joint_positions"]
    initial_vel = state["joint_velocities"]
    initial_quat = state["quaternion"]

    # Load MJCF
    project_root = os.getcwd()
    mjcf_path = os.path.join(
        project_root,
        "source", "berkeley_humanoid_lite_assets", "data", "robots",
        "berkeley_humanoid", "berkeley_humanoid_lite", "mjcf",
        "bhl_biped_scene.xml"
    )

    if not os.path.exists(mjcf_path):
        print(f"  ERROR: MJCF not found at {mjcf_path}")
        print(f"  Make sure you're running from the project root.")
        return

    mj_model = mujoco.MjModel.from_xml_path(mjcf_path)
    mj_data = mujoco.MjData(mj_model)
    mj_model.opt.timestep = cfg.physics_dt

    physics_substeps = int(np.round(cfg.policy_dt / cfg.physics_dt))
    num_joints = cfg.num_joints
    action_indices = list(cfg.action_indices)
    sensordata_dof_size = 3 * mj_model.nu

    joint_kp = torch.tensor(cfg.joint_kp, dtype=torch.float32)
    joint_kd = torch.tensor(cfg.joint_kd, dtype=torch.float32)
    effort_limits = torch.tensor(cfg.effort_limits, dtype=torch.float32)
    gravity_vector = torch.tensor([0.0, 0.0, -1.0])

    # --- Set initial state ---
    # Set joint positions and orientation from log
    mj_data.qpos[3:7] = initial_quat
    mj_data.qpos[7:] = initial_pos
    mj_data.qvel[:] = 0

    # Find appropriate base height by settling under gravity briefly
    mj_data.qpos[2] = 0.35  # start above ground
    mujoco.mj_forward(mj_model, mj_data)

    # Settle: apply PD to hold logged positions while gravity finds ground contact
    target_pos_tensor = torch.tensor(initial_pos, dtype=torch.float32)
    for _ in range(500):
        cur_pos = torch.tensor(mj_data.sensordata[0:num_joints], dtype=torch.float32)
        cur_vel = torch.tensor(mj_data.sensordata[num_joints:2*num_joints], dtype=torch.float32)
        torques = joint_kp * (target_pos_tensor - cur_pos) + joint_kd * (-cur_vel)
        torques = torch.clip(torques, -effort_limits, effort_limits)
        mj_data.ctrl[:] = torques.numpy()
        mujoco.mj_step(mj_model, mj_data)

    settled_z = mj_data.qpos[2]
    print(f"  Settled base height: {settled_z:.4f}m")

    # Re-set state cleanly at settled height
    mj_data.qpos[0:2] = [0, 0]
    mj_data.qpos[2] = settled_z
    mj_data.qpos[3:7] = initial_quat
    mj_data.qpos[7:] = initial_pos
    mj_data.qvel[:] = 0
    mj_data.qvel[6:6 + num_joints] = initial_vel
    mujoco.mj_forward(mj_model, mj_data)

    # Launch viewer
    viewer = mujoco.viewer.launch_passive(mj_model, mj_data)

    print(f"  Policy: {1 / cfg.policy_dt:.0f} Hz | Physics: {1 / cfg.physics_dt:.0f} Hz | Substeps: {physics_substeps}")
    print(f"  Simulation running... (close viewer window to stop)\n")

    def get_obs():
        """Build observation vector matching MujocoSimulator format."""
        base_quat = torch.tensor(
            mj_data.sensordata[sensordata_dof_size:sensordata_dof_size + 4],
            dtype=torch.float32)
        base_ang_vel = torch.tensor(
            mj_data.sensordata[sensordata_dof_size + 4:sensordata_dof_size + 7],
            dtype=torch.float32)
        joint_pos = torch.tensor(
            mj_data.sensordata[0:num_joints], dtype=torch.float32)
        joint_vel = torch.tensor(
            mj_data.sensordata[num_joints:2 * num_joints], dtype=torch.float32)

        return torch.cat([
            base_quat,
            base_ang_vel,
            joint_pos[action_indices],
            joint_vel[action_indices],
            torch.tensor([3.0, 0.0, 0.0, 0.0], dtype=torch.float32),
        ]).numpy()

    step_count = 0

    try:
        while viewer.is_running():
            step_start = time.perf_counter()

            obs = get_obs()
            actions = controller.update(obs)
            actions_tensor = torch.tensor(actions, dtype=torch.float32)

            for _ in range(physics_substeps):
                target = torch.zeros(num_joints)
                target[action_indices] = actions_tensor

                cur_pos = torch.tensor(
                    mj_data.sensordata[0:num_joints], dtype=torch.float32)
                cur_vel = torch.tensor(
                    mj_data.sensordata[num_joints:2 * num_joints], dtype=torch.float32)

                torques = joint_kp * (target - cur_pos) + joint_kd * (-cur_vel)
                torques = torch.clip(torques, -effort_limits, effort_limits)
                mj_data.ctrl[:] = torques.numpy()
                mujoco.mj_step(mj_model, mj_data)

            viewer.sync()
            step_count += 1

            if step_count <= 5 or step_count % 50 == 0:
                base_z = mj_data.qpos[2]
                jp = mj_data.sensordata[0:num_joints]
                print(f"  Step {step_count:>4}: base_z={base_z:.3f}  "
                      f"R_knee={jp[9]:+.3f}  L_knee={jp[3]:+.3f}")

            if 0 < max_steps <= step_count:
                print(f"\n  Reached max steps ({max_steps})")
                break

            elapsed = time.perf_counter() - step_start
            remaining = cfg.policy_dt - elapsed
            if remaining > 0:
                time.sleep(remaining)

    except KeyboardInterrupt:
        pass

    print(f"\n  Simulation ended after {step_count} steps")

    # Print final state comparison
    final_pos = mj_data.sensordata[0:num_joints]
    print(f"\n  === Final vs Initial Joint Positions ===")
    print(f"  {'Joint':<16} {'Initial':>8} {'Final':>8} {'Drift':>8}")
    print(f"  {'-' * 40}")
    for i, name in enumerate(JOINT_NAMES):
        d = final_pos[i] - initial_pos[i]
        flag = " !!!" if abs(d) > 0.2 else ""
        print(f"  {name:<16} {initial_pos[i]:>+8.3f} {final_pos[i]:>+8.3f} {d:>+8.3f}{flag}")

    viewer.close()


def main():
    parser = argparse.ArgumentParser(
        description='Sim2Sim: Play policy from logged robot state')
    parser.add_argument('--log', type=str, required=True,
                        help='Path to log JSON from run_locomotion.py --log')
    parser.add_argument('--config', type=str, default='configs/policy_biped_50hz.yaml',
                        help='Path to policy config YAML')
    parser.add_argument('--frame', type=int, default=0,
                        help='Log frame index to start from (default: 0)')
    parser.add_argument('--headless', action='store_true',
                        help='No MuJoCo viewer - just run policy forward passes')
    parser.add_argument('--max-steps', type=int, default=0,
                        help='Max simulation steps (0=unlimited)')
    args = parser.parse_args()

    # Load initial state from log
    state, metadata = load_log_frame(args.log, args.frame)

    print(f"\n{'=' * 60}")
    print(f"  Sim2Sim: Starting from logged state")
    print(f"  Log: {args.log}")
    print(f"  Frame: {args.frame} (step {state['step']}, t={state['timestamp']:.3f}s)")
    print(f"  Mode: {'headless' if args.headless else 'MuJoCo physics'}")
    print(f"{'=' * 60}")

    # Load config (override sys.argv for Cfg.from_arguments)
    sys.argv = [sys.argv[0], '--config', args.config]
    cfg = Cfg.from_arguments()

    default_pos = np.array(cfg.default_joint_positions, dtype=np.float32)
    print_initial_state(state, default_pos)

    # Initialize policy
    controller = RlController(cfg)
    controller.load_policy()

    # Restore prev_actions from log
    controller.prev_actions[:] = state["prev_actions"]
    print(f"  Loaded prev_actions from log frame")

    if args.headless:
        run_headless(cfg, controller, state, args.max_steps)
    else:
        mesh_ok = setup_mesh_symlinks()
        if not mesh_ok:
            print("  WARNING: Mesh assets not found. Visual rendering may fail.")
            print("  The simulation will still work for physics (collision uses primitives).")
        run_mujoco(cfg, controller, state, args.max_steps)


if __name__ == "__main__":
    main()
