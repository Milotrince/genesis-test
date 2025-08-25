import argparse
import os
import pickle
import shutil
from importlib import metadata
from math import pi

import numpy as np

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
            "desired_kl": 0.02,
            "entropy_coef": 0.001,
            "gamma": 0.99,
            "lam": 0.95,
            "learning_rate": 0.005,
            "max_grad_norm": 1.0,
            "num_learning_epochs": 5,
            "num_mini_batches": 64,  # 32768 is what is used for HORA?
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
        "num_steps_per_env": 8,  # horizon length ?
        "save_interval": 100,
        "empirical_normalization": None,
        "seed": 1,
    }

    return train_cfg_dict


def get_cfgs():
    env_cfg = {
        "num_actions": 16,  # should match dofs in the hand (length of joint_names)
        # ===== hand configuration ======
        "tactile_sensor_type": "normal_tangential",  # options: boolean, force, force_magnitude, normal_tangential
        "force_max_clip": 10.0,  # clip force sensor data, if applicable
        # tactile_sensor_grid_size is used to divide each link the tactile sensor is associated with into a grid
        # will use the *GridSensor type for value other than (1, 1, 1)
        "tactile_sensor_grid_size": (1, 1, 1),
        "tactile_sensors": [
            # "index_3_tip",
            # "middle_3_tip",
            # "ring_3_tip",
            # "thumb_3_tip",
        ],
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
        "hand_friction": 1.0,
        # hand pd control parameters
        "kp": 3.0,
        "kd": 0.1,
        "torque_clip": -0.5,
        # termination conditions
        "min_obj_z": 0.6,  # terminate if object falls below z
        "max_obj_distance": 0.1,  # 0.05,  # terminate if object moves too far from starting pos
        "max_obj_nontarget_axis_angle_diff": pi / 4.0,  # terminate if the rotation of any other axes is too large
        # ===== object configuration ======
        "obj_friction": 1.0,
        "obj_density": 1000.0,  # density in kg/m^3
        # ----- from vis_hand_env
        "default_joint_angles": [
            0.1029,
            0.1343,
            0.0645,
            1.1922,
            1.3538,
            1.1274,
            1.1962,
            1.0774,
            0.2687,
            0.1853,
            0.5328,
            0.1505,
            0.1862,
            0.5078,
            0.2611,
            0.8198,
        ],
        "hand_init_pos": [0.0, 0.0, 0.5],
        "hand_init_euler": [0.0, -90.0, 0.0],
        "obj_base_pos": [0.01, 0.0, 0.64],
        "obj_init_euler": [0.0, 0.0, 0.0],  # roll, pitch, yaw in degrees
        "obj_morph": "cylinder",
        "obj_size": 0.05,
        # ===== episode settings ======
        "episode_length_s": 20.0,
        "resampling_time_s": 5.0,
        "action_scale": 0.01,
        "clip_actions": 1.0,
    }

    # Calculate observation dimensions
    per_sensor_dim = 1
    if env_cfg["tactile_sensor_type"] == "force":
        per_sensor_dim = 3
    elif env_cfg["tactile_sensor_type"] == "normal_tangential":
        per_sensor_dim = 4
    tactile_dim = np.prod([len(env_cfg["tactile_sensors"]), per_sensor_dim, *env_cfg["tactile_sensor_grid_size"]])

    reward_cfg = {
        # Paper reward weights
        "reward_scales": {
            "delta_rotation": 100.0,  # r_rot
            "hand_pose": 10.0,  # r_pose: deviation from hand pose penalty
            "torque": 1.0,  # r_torque: torque penalty
            "work": 5.0,  # r_work: energy consumption penalty
            "obj_linvel": 30.0,  # r_linvel: object linear velocity penalty
            "alive": 1.0,  # r_alive: alive reward
            # from paper ---
            # "delta_rotation": 1.0,  # r_rot
            # "hand_pose": 0.3,  # r_pose: deviation from hand pose penalty
            # "torque": 0.1,  # r_torque: torque penalty
            # "work": 2.0,  # r_work: energy consumption penalty
            # "obj_linvel": 0.3,  # r_linvel: object linear velocity penalty
        },
        "rot_clip": 0.5,  # c1: Clipping value for rotation reward
        "early_termination_penalty": 0.1,
    }

    command_cfg = {
        "num_commands": 1,
        "obj_rot_axis": [2],  #  0:x 0:-axis, 1:  1:y 1:-axis, 2:  2:z 2:-axis
    }

    # Domain randomization configuration
    domain_randomization_cfg = {
        "obj_mass_shift": [0.8, 1.2],  # multiplier for base mass
        "obj_friction_ratio": [1.0, 1.5],  # multiplier for base friction
        "obj_com_shift": [-0.01, 0.01],  # shift center of mass (inertial pos)
        # "obj_pos_shift_x": [-0.01, 0.01],  # min/max x offset from base position
        # "obj_pos_shift_y": [-0.01, 0.01],  # min/max y offset from base position
        # "obj_pos_shift_z": [0.0, 0.01],  # min/max z offset from base position
        "obj_pos_shift_x": [0, 0],  # min/max x offset from base position
        "obj_pos_shift_y": [0, 0],  # min/max y offset from base position
        "obj_pos_shift_z": [0, 0],  # min/max z offset from base position
        # "obj_shape": (0.95, 1.05),  # genesis not support different shapes yet
        "kp_gain": [1.0, 1.0],  # Multiplier for base kp
        "kd_gain": [1.0, 1.0],  # Multiplier for base kd
        # "joint_observation_noise": [-0.05, 0.05],  # Additive noise
        "joint_observation_noise": [-0.01, 0.01],  # Additive noise
        # "action_noise": [-0.06, 0.06],  # Additive noise
        "action_noise": [-0.01, 0.01],  # Additive noise
        "gravity_x": [-0.0, 0.0],  # change force field (same as reorienting whole hand and object)
        "gravity_y": [-0.0, 0.0],
        "gravity_z": [-4.0, -4.0],
        # "gravity_x": [-3.0, 3.0],  # change force field (same as reorienting whole hand and object)
        # "gravity_y": [-3.0, 3.0],
        # "gravity_z": [-10.0, 10.0],
    }

    obs_cfg = {
        # commands + dof_pos (16) + tactile
        "num_obs": command_cfg["num_commands"] + 16 + tactile_dim,
        "obs_scales": {
            "tactile": 1.0,
        },
    }

    priv_cfg = {
        "num_priv_obs": 3 + 4 + 3,  # obj_pos + obj_quat + obj_ang_vel
    }

    return env_cfg, obs_cfg, reward_cfg, command_cfg, domain_randomization_cfg, priv_cfg


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
    cfgs = get_cfgs()
    env_cfg, obs_cfg, reward_cfg, command_cfg, domain_randomization_cfg, priv_cfg = cfgs
    train_cfg = get_train_cfg(args.exp_name, args.max_iterations)

    all_cfgs = [env_cfg, obs_cfg, reward_cfg, command_cfg, train_cfg, domain_randomization_cfg, priv_cfg]

    # Handle checkpoint loading for continue training
    if getattr(args, "continue"):
        if not os.path.exists(log_dir):
            print(f"Warning: Log directory {log_dir} does not exist. Starting fresh training.")
            os.makedirs(log_dir, exist_ok=True)
            # Save configs only when starting fresh
            pickle.dump(all_cfgs, open(f"{log_dir}/cfgs.pkl", "wb"))
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
                pickle.dump(all_cfgs, open(f"{log_dir}/cfgs.pkl", "wb"))
    else:
        # Original behavior - remove existing log directory
        if os.path.exists(log_dir):
            shutil.rmtree(log_dir)
        os.makedirs(log_dir, exist_ok=True)
        # Save configs for new training
        pickle.dump(all_cfgs, open(f"{log_dir}/cfgs.pkl", "wb"))

    env = InHandRotateEnv(
        num_envs=args.num_envs,
        env_cfg=env_cfg,
        obs_cfg=obs_cfg,
        priv_cfg=priv_cfg,
        reward_cfg=reward_cfg,
        command_cfg=command_cfg,
        domain_randomization_cfg=domain_randomization_cfg,
        show_viewer=args.vis,
        record_video=False,
    )

    runner = OnPolicyRunner(env, train_cfg, log_dir, device=gs.device)

    runner.learn(num_learning_iterations=args.max_iterations, init_at_random_ep_len=True)


if __name__ == "__main__":
    main()

"""
# training
python examples/sensors/in_hand_rotate/hand_train.py
"""
