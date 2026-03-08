"""Interactive play script with gamepad/keyboard control."""

"""Launch Isaac Sim Simulator first."""

import argparse

from isaaclab.app import AppLauncher

# local imports
import cli_args  # isort: skip

# add argparse arguments
parser = argparse.ArgumentParser(description="Interactive play with gamepad/keyboard control.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
# append RSL-RL cli arguments
cli_args.add_rsl_rl_args(parser)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
# always enable cameras to record video
if args_cli.video:
    args_cli.enable_cameras = True

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import os
import torch
import threading

from rsl_rl.runners import OnPolicyRunner

import isaaclab.utils.string as string_utils
from isaaclab.envs import DirectMARLEnv, multi_agent_to_single_agent
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper, export_policy_as_jit, export_policy_as_onnx
from isaaclab_tasks.utils import get_checkpoint_path, parse_env_cfg

# Import extensions to set up environment tasks
import berkeley_humanoid_lite.tasks  # noqa: F401

# Try to import gamepad
try:
    from berkeley_humanoid_lite_lowlevel.policy.gamepad_direct import Se2Gamepad
    GAMEPAD_AVAILABLE = True
except ImportError:
    GAMEPAD_AVAILABLE = False
    print("[WARN] Gamepad not available. Using keyboard controls.")


class KeyboardController:
    """Simple keyboard controller using terminal input."""

    def __init__(self):
        self.commands = {
            "velocity_x": 0.0,
            "velocity_y": 0.0,
            "velocity_yaw": 0.0,
        }
        self._stop = False
        self._thread = None

    def run(self):
        """Start keyboard input thread."""
        self._thread = threading.Thread(target=self._input_loop, daemon=True)
        self._thread.start()
        self._print_help()

    def _print_help(self):
        print("\n" + "="*60)
        print("KEYBOARD CONTROLS:")
        print("  w/s  : Forward/Backward")
        print("  a/d  : Turn Left/Right")
        print("  q/e  : Strafe Left/Right")
        print("  x    : Stop (zero velocity)")
        print("  h    : Print this help")
        print("  ESC  : Quit")
        print("="*60 + "\n")

    def _input_loop(self):
        """Read keyboard input in background thread."""
        import sys
        import select
        import tty
        import termios

        old_settings = termios.tcgetattr(sys.stdin)
        try:
            tty.setcbreak(sys.stdin.fileno())
            while not self._stop:
                if select.select([sys.stdin], [], [], 0.1)[0]:
                    key = sys.stdin.read(1)
                    self._handle_key(key)
        except Exception as e:
            print(f"Keyboard error: {e}")
        finally:
            termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_settings)

    def _handle_key(self, key):
        """Handle keyboard input."""
        step = 0.1

        if key == 'w':
            self.commands["velocity_x"] = min(self.commands["velocity_x"] + step, 0.5)
        elif key == 's':
            self.commands["velocity_x"] = max(self.commands["velocity_x"] - step, -0.5)
        elif key == 'a':
            self.commands["velocity_yaw"] = min(self.commands["velocity_yaw"] + step * 2, 1.0)
        elif key == 'd':
            self.commands["velocity_yaw"] = max(self.commands["velocity_yaw"] - step * 2, -1.0)
        elif key == 'q':
            self.commands["velocity_y"] = min(self.commands["velocity_y"] + step, 0.25)
        elif key == 'e':
            self.commands["velocity_y"] = max(self.commands["velocity_y"] - step, -0.25)
        elif key == 'x':
            self.commands["velocity_x"] = 0.0
            self.commands["velocity_y"] = 0.0
            self.commands["velocity_yaw"] = 0.0
        elif key == 'h':
            self._print_help()
        elif key == '\x1b':  # ESC
            self._stop = True

        print(f"\rCmd: vx={self.commands['velocity_x']:.2f}, vy={self.commands['velocity_y']:.2f}, yaw={self.commands['velocity_yaw']:.2f}    ", end="", flush=True)

    def stop(self):
        self._stop = True


def main():
    """Play with RSL-RL agent with interactive control."""
    # parse configuration
    env_cfg = parse_env_cfg(
        args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs, use_fabric=not args_cli.disable_fabric
    )
    agent_cfg: RslRlOnPolicyRunnerCfg = cli_args.parse_rsl_rl_cfg(args_cli.task, args_cli)

    # specify directory for logging experiments
    log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Loading experiment from directory: {log_root_path}")
    resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)
    log_dir = os.path.dirname(resume_path)

    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

    # convert to single-agent instance if required by the RL algorithm
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    # wrap around environment for rsl-rl
    env = RslRlVecEnvWrapper(env)

    print(f"[INFO]: Loading model checkpoint from: {resume_path}")
    # load previously trained model
    ppo_runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    ppo_runner.load(resume_path)

    # obtain the trained policy for inference
    policy = ppo_runner.get_inference_policy(device=env.unwrapped.device)

    # Initialize controller
    if GAMEPAD_AVAILABLE:
        print("[INFO] Using gamepad controller")
        print("  Left Stick Y  : Forward/Backward")
        print("  Left Stick X  : Turn")
        print("  Right Stick X : Strafe")
        controller = Se2Gamepad()
    else:
        print("[INFO] Using keyboard controller")
        controller = KeyboardController()

    controller.run()

    # reset environment
    obs, _ = env.get_observations()
    timestep = 0

    # Get the underlying environment to access command manager
    base_env = env.unwrapped

    print("\n[INFO] Starting interactive play. Control the robot!")

    # simulate environment
    while simulation_app.is_running():
        # run everything in inference mode
        with torch.inference_mode():
            # Get commands from controller
            vx = controller.commands.get("velocity_x", 0.0)
            vy = controller.commands.get("velocity_y", 0.0)
            vyaw = controller.commands.get("velocity_yaw", 0.0)

            # Set commands on the environment's command manager
            # The command tensor is [vx, vy, vyaw] for each env
            try:
                cmd_tensor = base_env.command_manager.get_command("base_velocity")
                cmd_tensor[:, 0] = vx  # lin_vel_x
                cmd_tensor[:, 1] = vy  # lin_vel_y
                cmd_tensor[:, 2] = vyaw  # ang_vel_z
            except Exception as e:
                pass  # Command manager might not be accessible this way

            # agent stepping
            actions = policy(obs)
            # env stepping
            obs, _, _, _ = env.step(actions)

        if args_cli.video:
            timestep += 1
            if timestep == args_cli.video_length:
                break

    # cleanup
    controller.stop()
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
