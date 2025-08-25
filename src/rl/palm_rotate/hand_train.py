import argparse
import os
import pickle
import shutil
import signal
import sys
from importlib import metadata
from math import pi

try:
    try:
        if metadata.version("rsl-rl"):
            raise ImportError
    except metadata.PackageNotFoundError:
        if metadata.version("rsl-rl-lib") != "2.2.4":
            raise ImportError
except (metadata.PackageNotFoundError, ImportError) as e:
    raise ImportError("Please uninstall 'rsl_rl' and install 'rsl-rl-lib==2.2.4'.") from e
import genesis as gs
from hand_env import InHandRotateEnv
from rsl_rl.runners import OnPolicyRunner


def get_train_cfg(exp_name, max_iterations):
    train_cfg_dict = {
        "algorithm": {
            "class_name": "PPO",
            "clip_param": 0.2,
            "desired_kl": 0.01,
            "entropy_coef": 0.01,
            "gamma": 0.99,
            "lam": 0.95,
            "learning_rate": 0.001,
            "max_grad_norm": 1.0,
            "num_learning_epochs": 5,
            "num_mini_batches": 4,
            "schedule": "adaptive",
            "use_clipped_value_loss": True,
            "value_loss_coef": 1.0,
        },
        "init_member_classes": {},
        "policy": {
            "activation": "elu",
            "actor_hidden_dims": [512, 256, 128],
            "critic_hidden_dims": [512, 256, 128],
            "init_noise_std": 1.0,
            "class_name": "ActorCritic",
        },
        "runner": {
            "checkpoint": -1,
            "experiment_name": exp_name,
            "load_run": -1,
            "log_interval": 1,
            "max_iterations": max_iterations,
            "record_interval": -1,
            "resume": False,
            "resume_path": None,
            "run_name": "",
        },
        "runner_class_name": "OnPolicyRunner",
        "num_steps_per_env": 24,
        "save_interval": 100,
        "empirical_normalization": None,
        "seed": 1,
    }

    return train_cfg_dict


def get_cfgs():
    env_cfg = {
        "num_actions": 16,  # should match dofs in the hand (length of joint_names)
        # hand configuration
        # "tactile_sensor_type": "RigidNormalTangentialForceSensor",
        "tactile_sensors": [
            "palm",
            "index_3_tip",
            "middle_3_tip",
            "ring_3_tip",
            "thumb_3_tip",
        ],
        "force_max_clip": 10.0,
        "joint_names": [
            "index_roll",
            "middle_roll",
            "ring_roll",
            "thumb_bend0",
            "index_bend0",
            "middle_bend0",
            "ring_bend0",
            "thumb_roll",
            "index_bend1",
            "middle_bend1",
            "ring_bend1",
            "thumb_bend1",
            "index_bend2",
            "middle_bend2",
            "ring_bend2",
            "thumb_bend2",
        ],
        "default_joint_angles": [0.1, 0.0, -0.1, 0.6, 0.6, 0.6, 0.6, 1.0, 1.0, 1.0, 1.0, 0.7, 0.3, 0.3, 0.4, 0.6],
        "hand_init_pos": [0.0, 0.0, 0.5],
        "hand_init_euler": [0.0, -90.0, 0.0],
        "hand_friction": 1.0,
        # hand pd control parameters
        "kp": 10.0,
        "kd": 0.5,
        # termination conditions
        "max_obj_distance": 0.16,  # terminate if object moves too far from hand
        "max_obj_nontarget_axis_angle_diff": pi / 6.0,  # terminate if the rotation of any other axes is too large
        # object
        "obj_friction": 1.0,
        "obj_density": 1000.0,  # density in kg/m^3
        "obj_base_pos": [0.03, 0.0, 0.56],  # renamed from obj_init_pos
        "obj_init_euler": [0.0, 0.0, 0.0],  # roll, pitch, yaw in degrees
        "obj_size": [0.05, 0.07, 0.1],
        # episode settings
        "episode_length_s": 10.0,
        "resampling_time_s": 2.0,
        "action_scale": 0.5,
        "clip_actions": 10.0,
    }

    # Calculate observation dimensions
    tactile_dim = len(env_cfg["tactile_sensors"])

    obs_cfg = {
        # obj_pos (3) + obj_quat (4) + obj_ang_vel (3) + commands (1) + dof_pos (16) + dof_vel (16) + actions (16) + tactile
        "num_obs": 3 + 4 + 3 + 1 + 16 + 16 + 16 + tactile_dim,
        "obs_scales": {
            "obj_pos": 1.0,
            "obj_quat": 1.0,
            "ang_vel": 0.1,  # normalize angular velocity
            "dof_pos": 1.0,
            "dof_vel": 0.1,
            "tactile": 0.01,  # Reduced tactile scaling for stability
        },
    }

    reward_cfg = {
        # Paper reward weights
        "reward_scales": {
            "delta_rotation": 50.0,  # w_rot: Rotation reward
            "velocity": 0.2,  # w_vel: Velocity penalty
            "fall": 1.0,  # w_fall: Falling penalty
            "work": 0.005,  # w_work: Work penalty
            "torque": 0.005,  # w_torque: Torque penalty
            "is_contact": 0.01,  # w_contact: Contact reward
            "fingertip_distance": 0.05,  # w_dist: Distance reward
        },
        "contact_force_threshold": 0.0001,  # Force threshold to consider finger in contact
        "contacts_clip": 3,  # clip num_contacts reward; don't need all fingers in contact at once
        "rot_clip": 0.157,  # c1: Clipping value for rotation reward
        "dist_epsilon": 0.02,  # epsilon: Small constant for distance calculation
        "dist_scale": 0.1,  # Scaling factor for distance reward
        "dist_min_clip": 0.0,  # c2: Min clip for distance reward
        "dist_max_clip": 1.0,  # c3: Max clip for distance reward
        "early_termination_penalty": 1.0,
    }

    command_cfg = {
        "num_commands": 1,
        "obj_rot_axis": [2],  #  0:x 0:-axis, 1:  1:y 1:-axis, 2:  2:z 2:-axis
    }

    # Domain randomization configuration
    domain_randomization_cfg = {
        "obj_mass_shift": [0.5, 2.0],  # multiplier for base mass
        "obj_friction_ratio": [0.2, 1.5],  # multiplier for base friction
        "obj_com_shift": [-0.01, 0.01],  # shift center of mass (inertial pos)
        "obj_pos_shift_x": [-0.01, 0.01],  # min/max x offset from base position
        "obj_pos_shift_y": [-0.01, 0.01],  # min/max y offset from base position
        "obj_pos_shift_z": [0.0, 0.01],  # min/max z offset from base position
        # "obj_shape": (0.95, 1.05),  # genesis not support different shapes yet
        "kp_gain": [0.66, 1.33],  # Multiplier for base kp
        "kd_gain": [0.80, 1.20],  # Multiplier for base kd
        "joint_observation_noise": [-0.05, 0.05],  # Additive noise
        "action_noise": [-0.06, 0.06],  # Additive noise
    }

    return env_cfg, obs_cfg, reward_cfg, command_cfg, domain_randomization_cfg


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--exp_name", type=str, default="in_hand_rotate")
    parser.add_argument("-v", "--vis", action="store_true", default=False)
    parser.add_argument("-B", "--num_envs", type=int, default=4096)
    parser.add_argument("-i", "--max_iterations", type=int, default=501)
    parser.add_argument(
        "--continue", action="store_true", default=False, help="Continue training from latest checkpoint"
    )
    args = parser.parse_args()

    gs.init(logging_level="warning")

    log_dir = f"logs/{args.exp_name}"
    env_cfg, obs_cfg, reward_cfg, command_cfg, domain_randomization_cfg = get_cfgs()
    train_cfg = get_train_cfg(args.exp_name, args.max_iterations)

    # Handle checkpoint loading for continue training
    if getattr(args, "continue"):
        if not os.path.exists(log_dir):
            print(f"Warning: Log directory {log_dir} does not exist. Starting fresh training.")
            os.makedirs(log_dir, exist_ok=True)
            # Save configs only when starting fresh
            pickle.dump(
                [env_cfg, obs_cfg, reward_cfg, command_cfg, train_cfg, domain_randomization_cfg],
                open(f"{log_dir}/cfgs.pkl", "wb"),
            )
        else:
            # Find the latest checkpoint
            checkpoint_files = [f for f in os.listdir(log_dir) if f.startswith("model_") and f.endswith(".pt")]
            if checkpoint_files:
                # Extract iteration numbers and find the latest
                iterations = [int(f.split("_")[1].split(".")[0]) for f in checkpoint_files]
                latest_iteration = max(iterations)
                train_cfg["runner"]["resume"] = True
                train_cfg["runner"]["load_run"] = latest_iteration
                print(f"Continuing training from iteration {latest_iteration}")
                # Don't overwrite configs when continuing - this preserves tensorboard logs
            else:
                print(f"Warning: No checkpoints found in {log_dir}. Starting fresh training.")
                os.makedirs(log_dir, exist_ok=True)
                # Save configs for fresh training in existing directory
                pickle.dump(
                    [env_cfg, obs_cfg, reward_cfg, command_cfg, train_cfg, domain_randomization_cfg],
                    open(f"{log_dir}/cfgs.pkl", "wb"),
                )
    else:
        # Original behavior - remove existing log directory
        if os.path.exists(log_dir):
            shutil.rmtree(log_dir)
        os.makedirs(log_dir, exist_ok=True)
        # Save configs for new training
        pickle.dump(
            [env_cfg, obs_cfg, reward_cfg, command_cfg, train_cfg, domain_randomization_cfg],
            open(f"{log_dir}/cfgs.pkl", "wb"),
        )

    env = InHandRotateEnv(
        num_envs=args.num_envs,
        env_cfg=env_cfg,
        obs_cfg=obs_cfg,
        reward_cfg=reward_cfg,
        command_cfg=command_cfg,
        domain_randomization_cfg=domain_randomization_cfg,
        show_viewer=args.vis,
    )

    runner = OnPolicyRunner(env, train_cfg, log_dir, device=gs.device)
    runner.git_status_repos = []

    # Set up signal handler for graceful shutdown
    def signal_handler(signum, frame):
        print(f"\nReceived signal {signum}. Attempting to save checkpoint...")
        try:
            current_iteration = getattr(runner, "current_learning_iteration", 0)
            if hasattr(runner, "alg") and hasattr(runner.alg, "actor_critic"):
                checkpoint_path = f"{log_dir}/model_{current_iteration}_interrupted.pt"
                runner.save(checkpoint_path)
                print(f"Checkpoint saved to: {checkpoint_path}")
            else:
                print("Warning: Could not save checkpoint - training may not have started yet")
        except Exception as e:
            print(f"Error saving checkpoint: {e}")
        print("Training terminated.")
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)

    try:
        runner.learn(num_learning_iterations=args.max_iterations, init_at_random_ep_len=True)
    except KeyboardInterrupt:
        print("\nTraining interrupted by user.")
        signal_handler(signal.SIGINT, None)


if __name__ == "__main__":
    main()

"""
# training
python examples/sensors/in_hand_rotate/hand_train.py
"""
