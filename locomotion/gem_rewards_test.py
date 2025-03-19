import argparse
import os
import pickle
import time
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

import torch
from gem_env import GemEnv
from rsl_rl.runners import OnPolicyRunner

import genesis as gs

from gem_manual import ManualController
from gem_train import get_cfgs


class RewardBarVisualizer:
    def __init__(self, reward_names):
        self.reward_names = reward_names
        self.reward_values = {name: 0.0 for name in reward_names}

        # Setup plot
        plt.ion()  # Interactive mode on
        self.fig, self.ax = plt.subplots(figsize=(10, 6))
        self.bars = self.ax.barh(reward_names, [0] * len(reward_names))

        # Set colors for bars
        colors = plt.cm.viridis(np.linspace(0, 1, len(reward_names)))
        for bar, color in zip(self.bars, colors):
            bar.set_color(color)

        self.ax.set_xlim(-1, 1)  # Initial range, will adjust dynamically
        self.ax.set_title('Real-time Reward Values')
        self.ax.grid(True, axis='x')

        # Add value labels on bars
        self.labels = []
        for bar in self.bars:
            label = self.ax.text(
                0.01,
                bar.get_y() + bar.get_height() / 2,
                '0.00',
                va='center'
            )
            self.labels.append(label)

        plt.tight_layout()
        plt.show(block=False)

    def update(self, reward_values):
        # Update stored values
        for name, value in reward_values.items():
            if name in self.reward_values:
                self.reward_values[name] = value

        # Get values in correct order
        values = [self.reward_values[name] for name in self.reward_names]

        # Update bar heights
        for bar, value in zip(self.bars, values):
            bar.set_width(value)

        # Find min and max for adjusting the axis limits
        min_val = min(values)
        max_val = max(values)
        margin = max(0.001, (max_val - min_val) * 0.2)
        self.ax.set_xlim(min(min_val - margin, -0.001), max(max_val + margin, 0.001))

        # Update labels
        for label, value, bar in zip(self.labels, values, self.bars):
            label.set_position((min(value, 0.01) if value < 0 else 0.01, bar.get_y() + bar.get_height() / 2))
            label.set_text(f"{value:.2f}")

        # Update the figure
        self.fig.canvas.draw_idle()
        self.fig.canvas.flush_events()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--exp_name", type=str, default="gem")
    args = parser.parse_args()

    gs.init(logging_level="error")

    log_dir = f"logs/{args.exp_name}"
    env_cfg, obs_cfg, reward_cfg, command_cfg, action_cfg = get_cfgs()

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
        command_cfg=command_cfg,
        action_cfg=action_cfg,
        show_viewer=True,
    )

    env.reset()

    controller = ManualController()

    # Initialize the reward visualizer with the names of reward functions
    reward_names = list(env.reward_functions.keys())
    visualizer = RewardBarVisualizer(reward_names)

    with torch.no_grad():
        while controller.running:
            steering, speed = controller.update()
            steering = steering / action_cfg["action_scales"]["steering"]
            speed = speed / action_cfg["action_scales"]["velocity"]
            actions = torch.tensor([steering, speed], device="cuda:0").unsqueeze(0)
            obs, _, rews, dones, infos = env.step(actions)

            # Collect reward values
            reward_values = {}
            for name, reward_func in env.reward_functions.items():
                raw_value = reward_func()[0].item()
                # Multiply by scale if available
                if name in reward_cfg["reward_scales"]:
                    scaled_value = raw_value * reward_cfg["reward_scales"][name]
                    reward_values[name] = scaled_value

            # Update the visualization
            visualizer.update(reward_values)


if __name__ == "__main__":
    main()
