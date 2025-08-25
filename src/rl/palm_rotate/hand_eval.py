import argparse
import os
import pickle
from importlib import metadata

import torch

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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-e",
        "--exp_name",
        type=str,
        default="in_hand_rotate",
        help="Experiment name, should match name of folder in logs/",
    )
    parser.add_argument("--ckpt", type=str, required=True, help="Checkpoint to load, e.g. '500' for model_500.pt")
    args = parser.parse_args()

    gs.init()

    log_dir = f"logs/{args.exp_name}"
    env_cfg, obs_cfg, reward_cfg, command_cfg, train_cfg, domain_randomization_cfg = pickle.load(
        open(f"logs/{args.exp_name}/cfgs.pkl", "rb")
    )

    reward_cfg["reward_scales"] = {}

    env = InHandRotateEnv(
        num_envs=1,
        env_cfg=env_cfg,
        obs_cfg=obs_cfg,
        reward_cfg=reward_cfg,
        command_cfg=command_cfg,
        domain_randomization_cfg=domain_randomization_cfg,
        show_viewer=True,
    )

    runner = OnPolicyRunner(env, train_cfg, log_dir, device=gs.device)
    resume_path = os.path.join(log_dir, f"model_{args.ckpt}.pt")
    runner.load(resume_path)
    policy = runner.get_inference_policy(device=gs.device)

    obs, _ = env.reset()
    try:
        with torch.no_grad():
            while True:
                actions = policy(obs)
                obs, rews, dones, infos = env.step(actions)
    except (KeyboardInterrupt, gs.GenesisException) as error:
        if (
            isinstance(error, gs.GenesisException)
            and str(error) == "Viewer closed."
            or isinstance(error, KeyboardInterrupt)
        ):
            return
        raise error
    finally:
        env.cleanup()


if __name__ == "__main__":
    main()

"""
# evaluation
python examples/sensors/in_hand_rotate/hand_eval.py -e in_hand_rotate --ckpt 100
"""
