import argparse
import os
import pickle

import torch
from gem_env import GemEnv
from rsl_rl.runners import OnPolicyRunner

import genesis as gs

from gem_manual import ManualController


def print_rewards(env: GemEnv):
    print("------------------------------------")
    for name, reward_func in env.reward_functions.items():
        print(f"{name}: {reward_func()[0]:.2f}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--exp_name", type=str, default="gem")
    args = parser.parse_args()

    gs.init(logging_level="error")

    log_dir = f"logs/{args.exp_name}"
    env_cfg, obs_cfg, reward_cfg, command_cfg, train_cfg, action_cfg = pickle.load(
        open(f"logs/{args.exp_name}/cfgs.pkl", "rb"))

    # visualize the target
    env_cfg["visualize_target"] = True
    # set the max FPS for visualization
    env_cfg["max_visualize_FPS"] = 60

    env = GemEnv(
        num_envs=1,
        env_cfg=env_cfg,
        obs_cfg=obs_cfg,
        reward_cfg=reward_cfg,
        command_cfg=command_cfg,
        action_cfg=action_cfg,
        show_viewer=True,
    )

    env.reset()

    controller = ManualController()

    with torch.no_grad():
        while controller.running:
            steering, speed = controller.update()
            steering = steering / action_cfg["action_scales"]["steering"]
            speed = speed / action_cfg["action_scales"]["velocity"]
            actions = torch.tensor([steering, speed], device="cuda:0").unsqueeze(0)
            obs, _, rews, dones, infos = env.step(actions)
            print_rewards(env)

if __name__ == "__main__":
    main()