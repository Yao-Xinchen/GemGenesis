import torch
import math
import genesis as gs
from genesis.ext.trimesh.path.packing import visualize
from genesis.utils.geom import xyz_to_quat, quat_to_xyz, transform_by_quat, inv_quat, transform_quat_by_quat
from sympy.physics.units import degree

from gem_ackermann import GemAckermann


def gs_rand_float(lower, upper, shape, device):
    return (upper - lower) * torch.rand(size=shape, device=device) + lower


r2d = 180.0 / math.pi
d2r = math.pi / 180.0


class GemEnv:
    def __init__(self, num_envs, env_cfg, obs_cfg, reward_cfg, command_cfg, action_cfg, show_viewer=False,
                 device="cuda"):
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
        self.action_cfg = action_cfg

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

        # build scene
        self.scene.build(n_envs=num_envs)

        # set gains
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
        self.crash_condition = torch.zeros((self.num_envs,), device=self.device, dtype=gs.tc_int)
        self.rel_pos = torch.zeros((self.num_envs, 3), device=self.device, dtype=gs.tc_float)
        self.last_rel_pos = torch.zeros_like(self.rel_pos)
        self.base_euler = torch.zeros((self.num_envs, 3), device=self.device, dtype=gs.tc_float)
        self.rel_yaw = torch.zeros((self.num_envs,), device=self.device, dtype=gs.tc_float)

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

    def _resample_commands(self, envs_idx):
        # command [x, y, orientation]
        self.commands[envs_idx, 0] = gs_rand_float(*self.command_cfg["pos_x_range"], (len(envs_idx),), self.device)
        self.commands[envs_idx, 1] = gs_rand_float(*self.command_cfg["pos_y_range"], (len(envs_idx),), self.device)
        self.commands[envs_idx, 2] = gs_rand_float(-torch.pi, torch.pi, (len(envs_idx),), self.device)

        if self.target is not None:
            zeros = torch.zeros_like(self.commands[envs_idx, 0])
            # position
            pos = torch.stack([self.commands[envs_idx, 0], self.commands[envs_idx, 1], zeros], dim=1)
            self.target.set_pos(pos, zero_velocity=True, envs_idx=envs_idx)
            # orientation
            euler = torch.stack([zeros, zeros, self.commands[envs_idx, 2]], dim=1)
            self.target.set_quat(xyz_to_quat(euler * r2d), zero_velocity=True, envs_idx=envs_idx)

    def _at_target(self):
        at_target = (
            (torch.norm(self.rel_pos, dim=1) < self.env_cfg["at_target_threshold"]).nonzero(as_tuple=False).flatten()
        )
        return at_target

    def step(self, actions):
        self.actions = torch.clip(actions, -self.env_cfg["clip_actions"], self.env_cfg["clip_actions"])

        # control the gem
        self.steering = actions[:, 0] * self.action_cfg["action_scales"]["steering"]
        self.speed = actions[:, 1] * self.action_cfg["action_scales"]["velocity"]
        self.outputs = self.locomotion.control(steering=self.steering, velocity=self.speed)
        self.gem.control_dofs_position(self.outputs[:, :2], self.pos_idx)
        self.gem.control_dofs_velocity(self.outputs[:, 2:], self.vel_idx)

        # step the scene
        self.scene.step()

        # update buffers
        self.episode_length_buf += 1
        self.last_base_pos[:] = self.base_pos[:]
        self.base_pos[:] = self.gem.get_pos()
        self.rel_pos = self.commands[:, :2] - self.base_pos[:, :2]
        self.last_rel_pos = self.commands[:, :2] - self.last_base_pos[:, :2]
        self.base_quat[:] = self.gem.get_quat()
        self.base_euler = quat_to_xyz(
            transform_quat_by_quat(torch.ones_like(self.base_quat) * self.inv_base_init_quat, self.base_quat),
        ) * d2r
        self.rel_yaw = (self.commands[:, 2] - self.base_euler[:, 2] + math.pi) % (2 * math.pi) - math.pi
        inv_base_quat = inv_quat(self.base_quat)
        self.base_lin_vel[:] = transform_by_quat(self.gem.get_vel(), inv_base_quat)
        self.base_ang_vel[:] = transform_by_quat(self.gem.get_ang(), inv_base_quat)

        # # resample targets
        # envs_idx = self._at_target()
        # self._resample_commands(envs_idx)

        self.crash_condition = (
                (torch.abs(self.base_euler[:, 1]) > self.env_cfg["termination_if_pitch_greater_than"])
                | (torch.abs(self.base_euler[:, 0]) > self.env_cfg["termination_if_roll_greater_than"])
                | (torch.abs(self.rel_pos[:, 0]) > self.env_cfg["termination_if_x_greater_than"])
                | (torch.abs(self.rel_pos[:, 1]) > self.env_cfg["termination_if_y_greater_than"])
        )
        self.reset_buf = (self.episode_length_buf > self.max_episode_length) | self.crash_condition

        time_out_idx = (self.episode_length_buf > self.max_episode_length).nonzero(as_tuple=False).flatten()
        self.extras["time_outs"] = torch.zeros_like(self.reset_buf, device=self.device, dtype=gs.tc_float)
        self.extras["time_outs"][time_out_idx] = 1.0

        self.reset_idx(self.reset_buf.nonzero(as_tuple=False).flatten())

        # compute rewards
        self.rew_buf[:] = 0.0
        for name, reward_func in self.reward_functions.items():
            rew = reward_func() * self.reward_scales[name]
            self.rew_buf += rew
            self.episode_sums[name] += rew

        self.obs_buf = torch.cat(
            [
                self.rel_pos[:, :2] * self.obs_scales["rel_pos"],  # 2
                self.last_rel_pos[:, :2] * self.obs_scales["rel_pos"],  # 2
                self.rel_yaw.unsqueeze(1) * self.obs_scales["rel_yaw"],  # 1
                torch.norm(self.base_lin_vel, dim=1).unsqueeze(1) * self.obs_scales["lin_vel"],  # 1
                self.base_ang_vel[:, 2].unsqueeze(1) * self.obs_scales["ang_vel"],  # 1
            ],
            dim=-1,
        )

        self.last_actions[:] = self.actions[:]

        return self.obs_buf, None, self.rew_buf, self.reset_buf, self.extras

    def get_observations(self):
        return self.obs_buf

    def get_privileged_observations(self):
        return None

    def reset_idx(self, envs_idx):
        if len(envs_idx) == 0:
            return

        # reset gem
        self.base_pos[envs_idx] = self.base_init_pos
        self.last_base_pos[envs_idx] = self.base_init_pos
        self.rel_pos = self.commands - self.base_pos
        self.last_rel_pos = self.commands - self.last_base_pos
        self.base_quat[envs_idx] = self.base_init_quat.reshape(1, -1)
        self.gem.set_pos(self.base_pos[envs_idx], zero_velocity=True, envs_idx=envs_idx)
        self.gem.set_quat(self.base_quat[envs_idx], zero_velocity=True, envs_idx=envs_idx)
        self.base_lin_vel[envs_idx] = 0
        self.base_ang_vel[envs_idx] = 0
        self.gem.zero_all_dofs_velocity(envs_idx)

        # reset buffers
        self.last_actions[envs_idx] = 0.0
        self.actions[envs_idx] = 0.0  # Reset current actions too
        self.episode_length_buf[envs_idx] = 0
        self.reset_buf[envs_idx] = True
        self.crash_condition[envs_idx] = 0  # Reset crash condition
        self.rew_buf[envs_idx] = 0.0  # Reset reward buffer

        # Reset control outputs
        self.steering[envs_idx] = 0.0
        self.speed[envs_idx] = 0.0
        self.outputs[envs_idx] = 0.0

        # Reset base euler and rel_yaw
        self.base_euler[envs_idx, :] = 0.0
        self.rel_yaw = (self.commands[:, 2] - self.base_euler[:, 2] + math.pi) % (2 * math.pi) - math.pi

        # fill extras
        self.extras["episode"] = {}
        for key in self.episode_sums.keys():
            self.extras["episode"]["rew_" + key] = (
                    torch.mean(self.episode_sums[key][envs_idx]).item() / self.env_cfg["episode_length_s"]
            )
            self.episode_sums[key][envs_idx] = 0.0

        self._resample_commands(envs_idx)

    def reset(self):
        self.reset_buf[:] = True
        self.reset_idx(torch.arange(self.num_envs, device=self.device))
        return self.obs_buf, None

    # ------------ reward functions----------------
    def _reward_dist_reduction(self):
        dist_decrease = torch.norm(self.last_rel_pos, dim=1) - torch.norm(self.rel_pos, dim=1)
        return dist_decrease

    def _reward_dist(self):
        return torch.norm(self.rel_pos, dim=1)

    def _reward_perpendicular_dist(self):
        target_unit_vec = torch.stack([torch.cos(self.commands[:, 2]), torch.sin(self.commands[:, 2])], dim=1)
        dist = self.rel_pos[:, 1] * target_unit_vec[:, 0] - self.rel_pos[:, 0] * target_unit_vec[:, 1]
        return dist ** 2

    def _reward_alignment(self):
        close_enough = torch.norm(self.rel_pos, dim=1) < self.env_cfg["at_target_threshold"] * 10
        cos_rel = torch.cos(self.rel_yaw)
        return cos_rel ** 2 * close_enough

    def _reward_success(self):
        pos_success = torch.norm(self.rel_pos, dim=1) < self.env_cfg["at_target_threshold"]
        orient_success = torch.abs(self.rel_yaw) < 0.15
        return (pos_success & orient_success).float()

    def _reward_at_target(self):
        return torch.norm(self.rel_pos, dim=1) < self.env_cfg["at_target_threshold"]

    def _reward_vel_at_target(self):
        close_to_target = torch.norm(self.rel_pos, dim=1) < self.env_cfg["at_target_threshold"]
        return close_to_target * torch.norm(self.base_lin_vel, dim=1)

    def _reward_smoothness(self):
        return torch.norm(self.actions - self.last_actions, dim=1)

    def _reward_stillness(self):
        return (torch.norm(self.base_lin_vel, dim=1) + 1.5) ** -2
