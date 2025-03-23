import numpy as np
from matplotlib import pyplot as plt

class GemPlot:
    def __init__(self, reward_names, obs_count):
        self.reward_names = reward_names
        self.obs_count = obs_count
        self.reward_values = {name: 0.0 for name in reward_names}
        self.obs_values = {f"obs_{i}": 0.0 for i in range(obs_count)}

        # Setup plot
        plt.ion()  # Interactive mode on
        self.fig, (self.ax_reward, self.ax_obs) = plt.subplots(1, 2, figsize=(16, 8))

        # Reward bars
        self.reward_bars = self.ax_reward.barh(range(len(reward_names)), [0] * len(reward_names))
        self.ax_reward.set_yticks(range(len(reward_names)))
        self.ax_reward.set_yticklabels(reward_names)
        self.ax_reward.set_xlim(-2, 2)
        self.ax_reward.set_title('Reward Values')
        self.ax_reward.grid(True, axis='x')

        # Set colors for reward bars
        reward_colors = plt.cm.viridis(np.linspace(0, 1, len(reward_names)))
        for bar, color in zip(self.reward_bars, reward_colors):
            bar.set_color(color)

        # Observation bars
        obs_labels = [f"obs_{i}" for i in range(obs_count)]
        self.obs_bars = self.ax_obs.barh(range(obs_count), [0] * obs_count)
        self.ax_obs.set_yticks(range(obs_count))
        self.ax_obs.set_yticklabels(obs_labels)
        self.ax_obs.set_xlim(-2, 2)
        self.ax_obs.set_title('Observation Values')
        self.ax_obs.grid(True, axis='x')

        # Set colors for obs bars
        obs_colors = plt.cm.plasma(np.linspace(0, 1, obs_count))
        for bar, color in zip(self.obs_bars, obs_colors):
            bar.set_color(color)

        # Add value labels on bars
        self.reward_labels = []
        for bar in self.reward_bars:
            label = self.ax_reward.text(
                0.01,
                bar.get_y() + bar.get_height() / 2,
                '0.00',
                va='center'
            )
            self.reward_labels.append(label)

        self.obs_labels = []
        for bar in self.obs_bars:
            label = self.ax_obs.text(
                0.01,
                bar.get_y() + bar.get_height() / 2,
                '0.00',
                va='center'
            )
            self.obs_labels.append(label)

        plt.tight_layout()
        plt.show(block=False)

    def update(self, reward_values, obs_tensor):
        # Update stored reward values
        for name, value in reward_values.items():
            if name in self.reward_values:
                self.reward_values[name] = value

        # Get reward values in correct order
        reward_vals = [self.reward_values[name] for name in self.reward_names]

        # Update reward bar widths
        for bar, value in zip(self.reward_bars, reward_vals):
            bar.set_width(value)

        # Update reward labels
        min_val = min(reward_vals)
        max_val = max(reward_vals)
        margin = max(0.001, (max_val - min_val) * 0.2)
        self.ax_reward.set_xlim(min(min_val - margin, -0.001), max(max_val + margin, 0.001))

        for label, value, bar in zip(self.reward_labels, reward_vals, self.reward_bars):
            position = min(value, 0.001) if value < 0 else 0.001
            label.set_position((position, bar.get_y() + bar.get_height() / 2))
            label.set_text(f"{value:.4f}")

        # Update observation bars directly from tensor
        obs_values = obs_tensor[0].cpu().tolist()  # Get first env's observations

        # Update obs bar widths
        for i, (bar, value) in enumerate(zip(self.obs_bars, obs_values)):
            bar.set_width(value)

            # Update obs label
            label = self.obs_labels[i]
            position = min(value, 0.01) if value < 0 else 0.01
            label.set_position((position, bar.get_y() + bar.get_height() / 2))
            label.set_text(f"{value:.4f}")

        # Update the figure
        self.fig.canvas.draw_idle()
        self.fig.canvas.flush_events()
