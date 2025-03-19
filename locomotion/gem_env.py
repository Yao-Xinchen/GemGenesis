import torch
import math
import genesis as gs
from genesis.ext.trimesh.path.packing import visualize
from genesis.utils.geom import xyz_to_quat, quat_to_xyz, transform_by_quat, inv_quat, transform_quat_by_quat
from sympy.physics.units import degree

from gem_ackermann import GemAckermann


def gs_rand_float(lower, upper, shape, device):
    return (upper - lower) * torch.rand(size=shape, device=device) + lower


def gs_rand_sign(shape, device):
    return 2 * torch.randint(2, size=shape, device=device) - 1


r2d = 180.0 / math.pi
d2r = math.pi / 180.0


class GemEnv:
    def __init__(self, num_envs, env_cfg, obs_cfg, reward_cfg, action_cfg, show_viewer=False, device="cuda"):
        self.device = torch.device(device)

        self.locomotion = GemAckermann(wheel_diameter=0.59, wheel_base=1.75, steer_dist_half=0.6, device=device)

        self.zeros = torch.zeros((num_envs, 1), device=self.device, dtype=gs.tc_float)

        self.num_envs = num_envs
        self.num_obs = obs_cfg["num_obs"]
        self.num_privileged_obs = None
        self.num_actions = env_cfg["num_actions"]

        self.target_pos = torch.zeros((num_envs, 3), device=self.device, dtype=gs.tc_float)
        self.target_quat = torch.zeros((num_envs, 4), device=self.device, dtype=gs.tc_float)
        self.target_quat[:, 0] = 1.0
        self.target_euler = torch.zeros((num_envs, 3), device=self.device, dtype=gs.tc_float)

        self.simulate_action_latency = env_cfg["simulate_action_latency"]
        self.dt = 0.01  # run in 100hz
        self.max_episode_length = math.ceil(env_cfg["episode_length_s"] / self.dt)

        self.env_cfg = env_cfg
        self.obs_cfg = obs_cfg
        self.reward_cfg = reward_cfg
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
                    pos=(0.0, 0.0, 0.0),
                    size=(5.5, 2.7, 0.01),
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

        # add obstacles
        obstacle_left_x = gs_rand_float(
            self.env_cfg["obstacle_x_range"][0], self.env_cfg["obstacle_x_range"][1],
            (self.num_envs, 1), self.device
        ) * gs_rand_sign((self.num_envs, 1), self.device)
        obstacle_left_y = gs_rand_float(
            self.env_cfg["obstacle_y_range"][0], self.env_cfg["obstacle_y_range"][1],
            (self.num_envs, 1), self.device
        )
        self.obstacle_left = self.scene.add_entity(
            morph=gs.morphs.Box(
                pos=(1.0, 0.0, 0.0),
                size=(4.5, 1.9, 1.5),
                fixed=True,
                visualization=True,
                collision=True,
            ),
        )

        obstacle_right_x = - obstacle_left_x
        obstacle_right_y = - obstacle_left_y
        self.obstacle_right = self.scene.add_entity(
            morph=gs.morphs.Box(
                pos=(1.0, 0.0, 0.0),
                size=(4.5, 1.9, 1.5),
                fixed=True,
                visualization=True,
                collision=True,
            ),
        )

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
        self.gem = self.scene.add_entity(gs.morphs.URDF(file="assets/gem/urdf/gem.urdf"))

        # build scene
        self.scene.build(n_envs=num_envs)

        self.obstacle_left.set_pos(torch.cat([obstacle_left_x, obstacle_left_y, self.zeros + 0.75], dim=1))
        self.obstacle_right.set_pos(torch.cat([obstacle_right_x, obstacle_right_y, self.zeros + 0.75], dim=1))

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
        self.crash_condition = torch.zeros((self.num_envs,), device=self.device, dtype=gs.tc_int)
        self.rel_pos_world = torch.zeros((self.num_envs, 3), device=self.device, dtype=gs.tc_float)
        self.last_rel_pos_world = torch.zeros_like(self.rel_pos_world)
        self.rel_pos_for_gem = torch.zeros_like(self.rel_pos_world)
        self.rel_pos_for_target = torch.zeros_like(self.rel_pos_world)
        self.base_euler = torch.zeros((self.num_envs, 3), device=self.device, dtype=gs.tc_float)
        self.rel_yaw_cos_square = torch.zeros((self.num_envs,), device=self.device, dtype=gs.tc_float)

        self.actions = torch.zeros((self.num_envs, self.num_actions), device=self.device, dtype=gs.tc_float)
        self.last_actions = torch.zeros_like(self.actions)

        self.base_pos = torch.zeros((self.num_envs, 3), device=self.device, dtype=gs.tc_float)
        self.base_quat = torch.zeros((self.num_envs, 4), device=self.device, dtype=gs.tc_float)
        self.base_lin_vel = torch.zeros((self.num_envs, 3), device=self.device, dtype=gs.tc_float)
        self.base_ang_vel = torch.zeros((self.num_envs, 3), device=self.device, dtype=gs.tc_float)
        self.last_base_pos = torch.zeros_like(self.base_pos)
        self.last_target_pos = torch.zeros_like(self.target_pos)

        self.extras = dict()  # extra information for logging

        self.steering = torch.zeros((self.num_envs,), device=self.device, dtype=gs.tc_float)
        self.speed = torch.zeros((self.num_envs,), device=self.device, dtype=gs.tc_float)
        self.outputs = torch.zeros((self.num_envs, len(joint_idx)), device=self.device, dtype=gs.tc_float)

    def _resample_base(self, envs_idx):
        num = len(envs_idx)
        base_pos = gs_rand_float(
            self.env_cfg["base_x_range"][0], self.env_cfg["base_x_range"][1], (num, 2), self.device
        ) * gs_rand_sign((num, 2), self.device)
        self.base_pos[envs_idx] = torch.cat([base_pos, torch.zeros((num, 1), device=self.device)], dim=1)

        base_yaw = gs_rand_float(-math.pi, math.pi, (num, 1), self.device)
        self.base_quat[envs_idx] = xyz_to_quat(
            torch.cat([
                torch.zeros((num, 2), device=self.device),
                base_yaw * r2d
            ], dim=1)
        )

        self.inv_base_quat = inv_quat(self.base_quat)

        self.base_euler = quat_to_xyz(self.base_quat) * d2r

    def _resample_obstacles(self, envs_idx):
        num = len(envs_idx)
        obstacle_left_x = gs_rand_float(
            self.env_cfg["obstacle_x_range"][0], self.env_cfg["obstacle_x_range"][1],
            (num, 1), self.device
        ) * gs_rand_sign((num, 1), self.device)
        obstacle_left_y = gs_rand_float(
            self.env_cfg["obstacle_y_range"][0], self.env_cfg["obstacle_y_range"][1],
            (num, 1), self.device
        )
        self.obstacle_left.set_pos(
            torch.cat([obstacle_left_x, obstacle_left_y, self.zeros + 0.75], dim=1),
            envs_idx=envs_idx)

        obstacle_right_x = - obstacle_left_x
        obstacle_right_y = - obstacle_left_y
        self.obstacle_right.set_pos(
            torch.cat([obstacle_right_x, obstacle_right_y, self.zeros + 0.75], dim=1),
            envs_idx=envs_idx)

    def step(self, actions):
        self.actions = torch.clip(actions, -self.env_cfg["clip_actions"], self.env_cfg["clip_actions"])

        # control the gem
        self.steering = actions[:, 0] * self.action_cfg["action_scales"]["steering"]
        max_steering = self.action_cfg["action_limits"]["steering_max"]
        self.steering = torch.clip(self.steering, -max_steering, max_steering)
        self.speed = actions[:, 1] * self.action_cfg["action_scales"]["velocity"]
        max_speed = self.action_cfg["action_limits"]["velocity_max"]
        self.speed = torch.clip(self.speed, -max_speed, max_speed)
        self.outputs = self.locomotion.control(steering=self.steering, velocity=self.speed)
        self.gem.control_dofs_position(self.outputs[:, :2], self.pos_idx)
        self.gem.control_dofs_velocity(self.outputs[:, 2:], self.vel_idx)

        # step the scene
        self.scene.step()

        # update buffers
        self.episode_length_buf += 1
        self.last_base_pos[:] = self.base_pos[:]
        self.last_target_pos[:] = self.target_pos[:]
        self.base_pos[:] = self.gem.get_pos()
        self.rel_pos_world = self.target_pos - self.base_pos
        self.last_rel_pos_world = self.target_pos - self.last_base_pos
        rel_2d = torch.cat([self.rel_pos_world[:, :2], torch.zeros((self.num_envs, 1), device=self.device)], dim=1)
        self.rel_pos_for_gem = transform_by_quat(rel_2d, inv_quat(self.base_quat))
        self.rel_pos_for_target = transform_by_quat(rel_2d, inv_quat(self.target_quat))
        self.base_quat[:] = self.gem.get_quat()
        self.inv_base_quat = inv_quat(self.base_quat)
        self.base_euler = quat_to_xyz(self.base_quat) * d2r
        self.rel_yaw_cos_square = torch.cos(self.target_euler[:, 2] - self.base_euler[:, 2]) ** 2
        self.base_lin_vel[:] = transform_by_quat(self.gem.get_vel(), self.inv_base_quat)
        self.base_ang_vel[:] = transform_by_quat(self.gem.get_ang(), self.inv_base_quat)

        self.crash_condition = (
                (torch.abs(self.base_euler[:, 1]) > self.env_cfg["termination_if_pitch_greater_than"])
                | (torch.abs(self.base_euler[:, 0]) > self.env_cfg["termination_if_roll_greater_than"])
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

        norm_long = self.obs_scales["rel_pos_long"]
        norm_short = self.obs_scales["rel_pos_short"]
        long_dist = torch.clip(self.rel_pos_for_gem[:, :2], -norm_long, norm_long) / norm_long
        short_dist = torch.clip(self.rel_pos_for_gem[:, :2], -norm_short, norm_short) / norm_short

        self.obs_buf = torch.cat(
            [
                long_dist,  # 2
                short_dist,  # 2
                self.rel_yaw_cos_square.unsqueeze(1) * self.obs_scales["rel_yaw_cos_square"],  # 1
                torch.norm(self.base_lin_vel, dim=1).unsqueeze(1) * self.obs_scales["lin_vel"],  # 1
                self.base_ang_vel[:, 2].unsqueeze(1) * self.obs_scales["ang_vel"],  # 1
                torch.norm(self.base_euler[:, :2], dim=1).unsqueeze(1) * self.obs_scales["base_euler"],  # 1
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

        self._resample_base(envs_idx)

        # reset gem
        self.rel_pos_world = self.target_pos - self.base_pos
        self.last_rel_pos_world = self.last_target_pos - self.last_base_pos
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

        self.rel_yaw_cos_square = torch.cos(self.target_euler[:, 2] - self.base_euler[:, 2]) ** 2

        # fill extras
        self.extras["episode"] = {}
        for key in self.episode_sums.keys():
            self.extras["episode"]["rew_" + key] = (
                    torch.mean(self.episode_sums[key][envs_idx]).item() / self.env_cfg["episode_length_s"]
            )
            self.episode_sums[key][envs_idx] = 0.0

    def reset(self):
        self.reset_buf[:] = True
        self.reset_idx(torch.arange(self.num_envs, device=self.device))
        return self.obs_buf, None

    # ------------ reward functions----------------
    def _reward_dist(self):
        return torch.log(torch.norm(self.rel_pos_world, dim=1) / 3 + 1.0)

    def _reward_perp_dist(self):
        return torch.log(self.rel_pos_for_target[:, 1] ** 2 + 1.0)

    def _reward_alignment(self):
        close_enough = ((torch.abs(self.rel_pos_for_target[:, 0]) < self.env_cfg["at_target_threshold_x"] * 5.)
                        & (torch.abs(self.rel_pos_for_target[:, 1]) < self.env_cfg["at_target_threshold_y"] * 5.))
        return (self.rel_yaw_cos_square - 0.5) * close_enough

    def _reward_success(self):
        pos_success = ((torch.abs(self.rel_pos_for_target[:, 0]) < self.env_cfg["at_target_threshold_x"])
                       & (torch.abs(self.rel_pos_for_target[:, 1]) < self.env_cfg["at_target_threshold_y"]))
        orient_success = self.rel_yaw_cos_square > 0.8
        return (pos_success & orient_success).float()

    def _reward_at_target(self):
        return ((torch.abs(self.rel_pos_for_target[:, 0]) < self.env_cfg["at_target_threshold_x"])
                & (torch.abs(self.rel_pos_for_target[:, 1]) < self.env_cfg["at_target_threshold_y"]))

    def _reward_vel_at_target(self):
        close_to_target = ((torch.abs(self.rel_pos_for_target[:, 0]) < self.env_cfg["at_target_threshold_x"] * 3.)
                           & (torch.abs(self.rel_pos_for_target[:, 1]) < self.env_cfg["at_target_threshold_y"] * 3.))
        return close_to_target * torch.norm(self.base_lin_vel, dim=1)

    def _reward_smoothness(self):
        return torch.norm(self.actions - self.last_actions, dim=1)

    def _reward_stillness(self):
        return (torch.norm(self.base_lin_vel, dim=1) + 1.5) ** -2

    def _reward_incline(self):
        return torch.norm(self.base_euler[:, :2], dim=1)

    def _reward_collision(self):
        force_left = torch.sum(torch.norm(
            self.obstacle_left.get_links_net_contact_force()  # [num_envs, num_links=1, 3]
            , dim=2
        ), dim=1)
        force_right = torch.sum(torch.norm(
            self.obstacle_right.get_links_net_contact_force()  # [num_envs, num_links=1, 3]
            , dim=2
        ), dim=1)
        return force_left + force_right
