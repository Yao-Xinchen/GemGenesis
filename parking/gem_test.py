import argparse
import os
import sys

import torch

import genesis as gs

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from gem_env import GemEnv
from gem_train import get_cfgs
from basic.gem_manual import ManualController
from basic.gem_plot import GemPlot


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--exp_name", type=str, default="gem")
    args = parser.parse_args()

    gs.init(logging_level="error")

    env_cfg, obs_cfg, reward_cfg, action_cfg = get_cfgs()

    # visualize the target
    env_cfg["visualize_target"] = True
    # set the max FPS for visualization
    env_cfg["max_visualize_FPS"] = 60
    # disable resetting
    env_cfg["episode_length_s"] = 10000000.
    env_cfg["resampling_time_s"] = 10000000.

    env = GemEnv(
        num_envs=1,
        env_cfg=env_cfg,
        obs_cfg=obs_cfg,
        reward_cfg=reward_cfg,
        action_cfg=action_cfg,
        show_viewer=True,
    )

    env.reset()

    controller = ManualController()

    # Initialize the dual visualizer with names of reward functions and observation count
    reward_names = list(env.reward_functions.keys())

    # Get sample observation to determine dimension
    sample_obs = env.get_observations()
    obs_dim = sample_obs.shape[1]

    visualizer = GemPlot(reward_names, obs_dim)

    with torch.no_grad():
        while controller.running:
            steering, speed = controller.update()
            steering = steering / action_cfg["action_scales"]["steering"]
            speed = speed / action_cfg["action_scales"]["velocity"]
            actions = torch.tensor([steering, speed], device="cuda:0").unsqueeze(0)
            obs, _, rews, dones, infos = env.step(actions)

            # Collect reward values with scales applied
            reward_values = {}
            for name, reward_func in env.reward_functions.items():
                raw_value = reward_func()[0].item()
                scale = reward_cfg["reward_scales"].get(name, 1.0)  # Default to 1.0 if no scale
                reward_values[name] = raw_value * scale

            # Update the visualization with indexed observations
            visualizer.update(reward_values, obs)


if __name__ == "__main__":
    main()
