import argparse
import os
import pickle
import shutil

from gem_env import GemEnv
from rsl_rl.runners import OnPolicyRunner

import genesis as gs


def get_train_cfg(exp_name, max_iterations):
    train_cfg_dict = {
        "algorithm": {
            "clip_param": 0.2,
            "desired_kl": 0.01,
            "entropy_coef": 0.004,
            "gamma": 0.99,
            "lam": 0.95,
            "learning_rate": 0.0001,
            "max_grad_norm": 1.0,
            "num_learning_epochs": 5,
            "num_mini_batches": 4,
            "schedule": "adaptive",
            "use_clipped_value_loss": True,
            "value_loss_coef": 1.0,
        },
        "init_member_classes": {},
        "policy": {
            "activation": "tanh",
            "actor_hidden_dims": [128, 128],
            "critic_hidden_dims": [128, 128],
            "init_noise_std": 1.0,
        },
        "runner": {
            "algorithm_class_name": "PPO",
            "checkpoint": -1,
            "experiment_name": exp_name,
            "load_run": -1,
            "log_interval": 1,
            "max_iterations": max_iterations,
            "num_steps_per_env": 100,
            "policy_class_name": "ActorCritic",
            "record_interval": -1,
            "resume": False,
            "resume_path": None,
            "run_name": "",
            "runner_class_name": "runner_class_name",
            "save_interval": 50,
        },
        "runner_class_name": "OnPolicyRunner",
        "seed": 1,
    }

    return train_cfg_dict


def get_cfgs():
    env_cfg = {
        "num_actions": 2,
        # physics
        "base_x_range": [-8.0, 8.0],
        "base_y_range": [3.0, 3.1],
        "base_yaw_range": [-0.25, 0.25],
        "wall_height": 1.5,
        "wall_thickness": 0.1,
        # termination
        "termination_if_roll_greater_than": .8,  # rad
        "termination_if_pitch_greater_than": .8,
        "episode_length_s": 30.0,
        "at_target_threshold_x": 1.0,
        "at_target_threshold_y": 0.4,
        # "resampling_time_s": 30.0,
        "simulate_action_latency": True,
        "clip_actions": 1.0,
        # visualization
        "visualize_target": False,
        "visualize_camera": False,
        "max_visualize_FPS": 60,
    }
    obs_cfg = {
        "num_obs": 10,
        "obs_scales": {
            "rel_dist": 0.2,
            "lin_vel": 0.15,
            "ang_vel": 1.2,
            "rel_yaw_cos_square": 1.0,
            "base_euler": 1.0,
        },
    }
    reward_cfg = {
        "reward_scales": {
            "dist": 2.,
            "alignment": 2.0,
            "dist_y": 8.,
            "smoothness": -1.0,
            "incline": -5.0,
            # "collision": -0.001,
            "success": 1000.0,
        },
    }
    action_cfg = {
        "action_scales": {
            "steering": 0.15,
            "velocity": 5.0,
        },
        "action_limits": {
            "steering_max": 0.3,
            "velocity_max": 10.0,
        },
    }

    return env_cfg, obs_cfg, reward_cfg, action_cfg


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--exp_name", type=str, default="gem")
    parser.add_argument("-B", "--num_envs", type=int, default=8192)
    parser.add_argument("--max_iterations", type=int, default=2500)
    args = parser.parse_args()

    gs.init(logging_level="error")

    log_dir = f"logs/{args.exp_name}"
    env_cfg, obs_cfg, reward_cfg, action_cfg = get_cfgs()
    train_cfg = get_train_cfg(args.exp_name, args.max_iterations)

    if os.path.exists(log_dir):
        shutil.rmtree(log_dir)
    os.makedirs(log_dir, exist_ok=True)

    env = GemEnv(
        num_envs=args.num_envs,
        env_cfg=env_cfg,
        obs_cfg=obs_cfg,
        reward_cfg=reward_cfg,
        action_cfg=action_cfg,
        show_viewer=False,
    )

    runner = OnPolicyRunner(env, train_cfg, log_dir, device="cuda:0")

    pickle.dump(
        [env_cfg, obs_cfg, reward_cfg, train_cfg, action_cfg],
        open(f"{log_dir}/cfgs.pkl", "wb"),
    )

    runner.learn(num_learning_iterations=args.max_iterations, init_at_random_ep_len=True)


if __name__ == "__main__":
    main()
