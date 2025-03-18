import torch
import math
import genesis as gs
from genesis.ext.trimesh.path.packing import visualize
from genesis.utils.geom import quat_to_xyz, transform_by_quat, inv_quat, transform_quat_by_quat

from gem_ackermann import GemAckermann


def gs_rand_float(lower, upper, shape, device):
    return (upper - lower) * torch.rand(size=shape, device=device) + lower


class GemEnv:
    def __init__(self, num_envs, env_cfg, obs_cfg, reward_cfg, command_cfg, show_viewer=False, device="cuda"):
        self.device = torch.device(device)

        self.locomotion = GemAckermann(wheel_diameter=0.59, wheel_base=1.75, steer_dist_half=0.6, device=device)

        self.num_envs = num_envs
        self.num_obs = obs_cfg["num_obs"]
        self.num_privileged_obs = None
        self.num_actions = env_cfg["num_actions"]
        self.num_commands = command_cfg["num_commands"]

        self.simulate_action_latency = env_cfg["simulate_action_latency"]
        self.dt = 0.01  # run in 100hz
        self.max_episode_length = math.ceil(env_cfg["episode_length_s"] / self.dt)

        self.env_cfg = env_cfg
        self.obs_cfg = obs_cfg
        self.reward_cfg = reward_cfg
        self.command_cfg = command_cfg

        self.obs_scales = obs_cfg["obs_scales"]
        self.reward_scales = reward_cfg["reward_scales"]

        # create scene
        self.scene = gs.Scene(
            sim_options=gs.options.SimOptions(dt=self.dt, substeps=2),
            viewer_options=gs.options.ViewerOptions(
                max_FPS=env_cfg["max_visualize_FPS"],
                camera_pos=(3.0, 0.0, 3.0),
                camera_lookat=(0.0, 0.0, 1.0),
                camera_fov=40,
            ),
            vis_options=gs.options.VisOptions(n_rendered_envs=10),
            rigid_options=gs.options.RigidOptions(
                dt=self.dt,
                constraint_solver=gs.constraint_solver.Newton,
                enable_collision=True,
                enable_joint_limit=True,
            ),
            show_viewer=show_viewer,
        )

        # add plane
        self.scene.add_entity(gs.morphs.Plane())

        # add target
        if self.env_cfg["visualize_target"]:
            self.target = self.scene.add_entity(
                morph=gs.morphs.Box(
                    pos=(1.0, 0.0, 0.0),
                    size=(4.0, 2.5, 0.01),
                    fixed=True,
                    visualization=True,
                    collision=False,
                ),
                surface=gs.surfaces.Glass(
                    color=(1.0, 0.4, 0.0),
                )
            )
        else:
            self.target = None

        # add camera
        if self.env_cfg["visualize_camera"]:
            self.cam = self.scene.add_camera(
                res=(640, 480),
                pos=(3.5, 0.0, 2.5),
                lookat=(0, 0, 0.5),
                fov=30,
                GUI=True,
            )

        # add gem
        self.base_init_pos = torch.tensor(self.env_cfg["base_init_pos"], device=self.device)
        self.base_init_quat = torch.tensor(self.env_cfg["base_init_quat"], device=self.device)
        self.inv_base_init_quat = inv_quat(self.base_init_quat)
        self.gem = self.scene.add_entity(gs.morphs.URDF(file="assets/gem/urdf/gem.urdf"))
        pos_joints = [
            "left_steering_hinge_joint",
            "right_steering_hinge_joint",
        ]
        self.pos_idx = [self.gem.get_joint(name).dof_idx_local for name in pos_joints]
        vel_joints = [
            "front_left_wheel_joint",
            "front_right_wheel_joint",
            "rear_left_wheel_joint",
            "rear_right_wheel_joint",
        ]
        self.vel_idx = [self.gem.get_joint(name).dof_idx_local for name in vel_joints]
        joint_idx = self.pos_idx + self.vel_idx
        kp_tensor = torch.tensor([35., 35., 0., 0., 0., 0.], device=device)
        kv_tensor = torch.tensor([5., 5., 30., 30., 30., 30.], device=device)
        self.gem.set_dofs_kp(kp=kp_tensor, dofs_idx_local=joint_idx)
        self.gem.set_dofs_kv(kv=kv_tensor, dofs_idx_local=joint_idx)

        # build scene
        self.scene.build(n_envs=num_envs)

        # prepare reward functions and multiply reward scales by dt
        self.reward_functions, self.episode_sums = dict(), dict()
        for name in self.reward_scales.keys():
            self.reward_scales[name] *= self.dt
            self.reward_functions[name] = getattr(self, "_reward_" + name)
            self.episode_sums[name] = torch.zeros((self.num_envs,), device=self.device, dtype=gs.tc_float)

        # initialize buffers
        self.obs_buf = torch.zeros((self.num_envs, self.num_obs), device=self.device, dtype=gs.tc_float)
        self.rew_buf = torch.zeros((self.num_envs,), device=self.device, dtype=gs.tc_float)
        self.reset_buf = torch.ones((self.num_envs,), device=self.device, dtype=gs.tc_int)
        self.episode_length_buf = torch.zeros((self.num_envs,), device=self.device, dtype=gs.tc_int)
        self.commands = torch.zeros((self.num_envs, self.num_commands), device=self.device, dtype=gs.tc_float)

        self.actions = torch.zeros((self.num_envs, self.num_actions), device=self.device, dtype=gs.tc_float)
        self.last_actions = torch.zeros_like(self.actions)

        self.base_pos = torch.zeros((self.num_envs, 3), device=self.device, dtype=gs.tc_float)
        self.base_quat = torch.zeros((self.num_envs, 4), device=self.device, dtype=gs.tc_float)
        self.base_lin_vel = torch.zeros((self.num_envs, 3), device=self.device, dtype=gs.tc_float)
        self.base_ang_vel = torch.zeros((self.num_envs, 3), device=self.device, dtype=gs.tc_float)
        self.last_base_pos = torch.zeros_like(self.base_pos)

        self.extras = dict()  # extra information for logging

        self.steering = torch.zeros((self.num_envs,), device=self.device, dtype=gs.tc_float)
        self.speed = torch.zeros((self.num_envs,), device=self.device, dtype=gs.tc_float)
        self.outputs = torch.zeros((self.num_envs, len(joint_idx)), device=self.device, dtype=gs.tc_float)

    def _at_target(self):
        gem_pos = self.base_pos
        target_pos = self.target.get_pos()
        return torch.norm(gem_pos - target_pos, dim=1) < 0.3

    def step(self, actions):
        self.actions = torch.clip(actions, -self.env_cfg["clip_actions"], self.env_cfg["clip_actions"])
        exec_actions = self.actions

        # control the gem
        self.steering = actions[:, 0]
        self.speed = actions[:, 1]
        self.outputs = self.locomotion.control(steering=self.steering, velocity=self.speed)
        self.gem.control_dofs_position(self.outputs[0, :2], self.pos_idx)
        self.gem.control_dofs_velocity(self.outputs[0, 2:], self.vel_idx)

        # step the scene
        self.scene.step()

        # TODO
