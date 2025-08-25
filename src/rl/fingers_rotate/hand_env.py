import math

import genesis as gs
import numpy as np
import torch
from genesis.sensors import (
    RecordingOptions,
    RigidContactForceGridSensor,
    RigidContactForceSensor,
    RigidContactGridSensor,
    RigidContactSensor,
    RigidNormalTangentialForceGridSensor,
    RigidNormalTangentialForceSensor,
    SensorDataRecorder,
    VideoFileWriter,
)
from genesis.utils.geom import quat_to_xyz, xyz_to_quat
from huggingface_hub import snapshot_download


def gs_rand_float(lower, upper, shape, device):
    return (upper - lower) * torch.rand(size=shape, device=device) + lower


def unscale(x, lower, upper):
    # normalize to -1 to 1
    return (2.0 * x - upper - lower) / (upper - lower)


class InHandRotateEnv:
    def __init__(
        self,
        num_envs,
        env_cfg,
        obs_cfg,
        reward_cfg,
        command_cfg,
        domain_randomization_cfg=None,
        priv_cfg=None,
        show_viewer=False,
        record_video=False,
    ):
        self.device = gs.device

        self.control_hz = 20
        self.dt = 1 / self.control_hz
        self.substeps = 3  # the effective simulation dt is self.dt/3

        self.max_episode_length = math.ceil(env_cfg["episode_length_s"] / self.dt)

        self.env_cfg = env_cfg
        self.obs_cfg = obs_cfg
        self.reward_cfg = reward_cfg
        self.command_cfg = command_cfg
        self.rand_cfg = domain_randomization_cfg
        self.priv_cfg = priv_cfg

        self.num_envs = num_envs
        self.num_obs = obs_cfg["num_obs"]
        self.num_actions = env_cfg["num_actions"]
        self.num_commands = command_cfg["num_commands"]
        self.num_priv_obs = priv_cfg["num_priv_obs"] if priv_cfg is not None else 0

        self.obs_scales = obs_cfg["obs_scales"]
        self.reward_scales = reward_cfg.get("reward_scales", {})

        self.show_viewer = show_viewer
        self.record_video = record_video

        # create scene
        self.scene = gs.Scene(
            sim_options=gs.options.SimOptions(dt=self.dt, substeps=self.substeps),
            profiling_options=gs.options.ProfilingOptions(
                show_FPS=False,
            ),
            viewer_options=gs.options.ViewerOptions(
                camera_pos=(
                    env_cfg["hand_init_pos"][0] + 0.3,
                    env_cfg["hand_init_pos"][1] + 0.3,
                    env_cfg["hand_init_pos"][2] + 0.7,
                ),
                camera_lookat=env_cfg["hand_init_pos"],
                camera_fov=40,
            ),
            vis_options=gs.options.VisOptions(rendered_envs_idx=[0]),
            rigid_options=gs.options.RigidOptions(
                dt=self.dt,
                enable_collision=True,
                enable_joint_limit=True,
                use_gjk_collision=True,
            ),
            show_viewer=self.show_viewer,
        )

        self.camera = None
        self.data_recorder = None
        if self.record_video:
            self.camera = self.scene.add_camera(
                res=(1280, 960),
                pos=(
                    env_cfg["hand_init_pos"][0] + 0.3,
                    env_cfg["hand_init_pos"][1] + 0.3,
                    env_cfg["hand_init_pos"][2] + 0.7,
                ),
                lookat=env_cfg["hand_init_pos"],
                fov=40,
            )
            self.data_recorder = SensorDataRecorder(step_dt=self.dt)
            self.data_recorder.add_sensor(
                self.camera,
                RecordingOptions(handler=VideoFileWriter(filename="hand_eval.mp4", fps=60), hz=60),
            )
            self.data_recorder.start_recording()

        self.scene.add_entity(gs.morphs.Plane())

        # add hand
        asset_path = snapshot_download(
            repo_id="Genesis-Intelligence/assets",
            allow_patterns="allegro_hand/*",
            repo_type="dataset",
        )

        self.hand_init_pos = torch.tensor(self.env_cfg["hand_init_pos"], device=gs.device)
        self.hand_offset_pos = self.hand_init_pos + torch.tensor([0.0, 0.0, 0.05], device=gs.device)
        self.hand_init_euler = torch.tensor(self.env_cfg["hand_init_euler"], device=gs.device)
        self.hand = self.scene.add_entity(
            morph=gs.morphs.URDF(
                file=f"{asset_path}/allegro_hand/allegro_hand_right_glb.urdf",
                pos=self.hand_init_pos.cpu().numpy(),
                euler=self.hand_init_euler.cpu().numpy(),
                fixed=True,
                merge_fixed_links=False,
            ),
            material=gs.materials.Rigid(
                gravity_compensation=1.0,
                friction=self.env_cfg["hand_friction"],
            ),
        )

        # add sensors
        self.sensors = []
        self.sensor_type = None
        self.per_sensor_dim = 1
        if self.env_cfg["tactile_sensor_grid_size"] == (1, 1, 1):
            if self.env_cfg["tactile_sensor_type"] == "boolean":
                self.sensor_type = RigidContactSensor
            elif self.env_cfg["tactile_sensor_type"] == "force_magnitude":
                self.sensor_type = RigidContactForceSensor
            elif self.env_cfg["tactile_sensor_type"] == "force":
                self.sensor_type = RigidContactForceSensor
                self.per_sensor_dim = 3
            elif self.env_cfg["tactile_sensor_type"] == "normal_tangential":
                self.sensor_type = RigidNormalTangentialForceSensor
                self.per_sensor_dim = 4
        else:
            if self.env_cfg["tactile_sensor_type"] == "boolean":
                self.sensor_type = RigidContactGridSensor
            elif self.env_cfg["tactile_sensor_type"] == "force_magnitude":
                self.sensor_type = RigidContactForceGridSensor
            elif self.env_cfg["tactile_sensor_type"] == "force":
                self.sensor_type = RigidContactForceGridSensor
                self.per_sensor_dim = 3
            elif self.env_cfg["tactile_sensor_type"] == "normal_tangential":
                self.sensor_type = RigidNormalTangentialForceGridSensor
                self.per_sensor_dim = 4
            # multiply dim for grid
            self.per_sensor_dim *= np.prod(self.env_cfg["tactile_sensor_grid_size"])

        if self.sensor_type is None:
            raise ValueError(f"Invalid tactile sensor type: {self.env_cfg['tactile_sensor_type']}")

        for link_name in self.env_cfg["tactile_sensors"]:
            kwargs = dict(entity=self.hand, link_idx=self.hand.get_link(link_name).idx)
            if "Grid" in self.env_cfg["tactile_sensor_type"]:
                kwargs["grid_size"] = self.env_cfg["tactile_sensor_grid_size"]
            self.sensors.append(self.sensor_type(**kwargs))
        self.total_tactile_dim = len(self.sensors) * self.per_sensor_dim

        # add object to rotate
        self.obj_base_pos = torch.tensor(self.env_cfg["obj_base_pos"], device=gs.device)
        self.obj_init_euler = torch.tensor(self.env_cfg["obj_init_euler"], device=gs.device)
        self.obj_init_quat = xyz_to_quat(self.obj_init_euler, degrees=True)
        if self.env_cfg["obj_morph"] == "box":
            obj_morph = gs.morphs.Box(
                pos=self.obj_base_pos.cpu().numpy(),
                euler=self.obj_init_euler.cpu().numpy(),
                size=self.env_cfg["obj_size"],
            )
        elif self.env_cfg["obj_morph"] == "cylinder":
            obj_morph = gs.morphs.Cylinder(
                pos=self.obj_base_pos.cpu().numpy(),
                euler=self.obj_init_euler.cpu().numpy(),
                radius=self.env_cfg["obj_size"],
                height=0.05,
            )
        elif self.env_cfg["obj_morph"] == "sphere":
            obj_morph = gs.morphs.Sphere(
                pos=self.obj_base_pos.cpu().numpy(),
                euler=self.obj_init_euler.cpu().numpy(),
                radius=self.env_cfg["obj_size"],
            )
        else:
            obj_morph = gs.morphs.Mesh(
                file=self.env_cfg["obj_morph"],
                scale=self.env_cfg["obj_size"],
                euler=self.obj_init_euler.cpu().numpy(),
                pos=self.obj_base_pos.cpu().numpy(),
            )

        self.obj = self.scene.add_entity(
            obj_morph,
            surface=gs.surfaces.Default(
                color=(1.0, 0.4, 0.0, 0.5),
            ),
            material=gs.materials.Rigid(
                rho=self.env_cfg["obj_density"],
                friction=self.env_cfg["obj_friction"],
            ),
        )

        self.scene.build(n_envs=num_envs)

        self.dofs_min_limit, self.dofs_max_limit = self.hand.get_dofs_limit()

        # PD control parameters
        self.motors_dof_idx = [self.hand.get_joint(name).dof_start for name in self.env_cfg["joint_names"]]
        self.hand.set_dofs_kp([self.env_cfg["kp"]] * self.num_actions, self.motors_dof_idx)
        self.hand.set_dofs_kv([self.env_cfg["kd"]] * self.num_actions, self.motors_dof_idx)
        self.hand.set_dofs_position([self.env_cfg["default_joint_angles"]] * self.num_envs, self.motors_dof_idx)

        # prepare reward functions and multiply reward scales by dt (only if rewards are defined)
        self.reward_functions, self.episode_sums = dict(), dict()
        for name in self.reward_scales.keys():
            self.reward_scales[name] *= self.dt
            if hasattr(self, f"_reward_{name}"):
                self.reward_functions[name] = getattr(self, f"_reward_{name}")
                self.episode_sums[name] = torch.zeros((self.num_envs,), device=gs.device, dtype=gs.tc_float)

        # initialize buffers
        self.obj_pos = torch.zeros((self.num_envs, 3), device=gs.device, dtype=gs.tc_float)
        self.obj_quat = torch.zeros((self.num_envs, 4), device=gs.device, dtype=gs.tc_float)
        self.obj_euler = torch.zeros((self.num_envs, 3), device=gs.device, dtype=gs.tc_float)
        self.obj_lin_vel = torch.zeros((self.num_envs, 3), device=gs.device, dtype=gs.tc_float)
        self.obj_ang_vel = torch.zeros((self.num_envs, 3), device=gs.device, dtype=gs.tc_float)
        self.initial_obj_euler = torch.zeros((self.num_envs, 3), device=gs.device, dtype=gs.tc_float)

        self.obs_buf = torch.zeros((self.num_envs, self.num_obs), device=gs.device, dtype=gs.tc_float)
        self.priv_obs_buf = torch.zeros((self.num_envs, self.num_priv_obs), device=gs.device, dtype=gs.tc_float)
        self.rew_buf = torch.zeros((self.num_envs,), device=gs.device, dtype=gs.tc_float)
        self.reset_buf = torch.ones((self.num_envs,), device=gs.device, dtype=gs.tc_int)
        self.episode_length_buf = torch.zeros((self.num_envs,), device=gs.device, dtype=gs.tc_int)
        self.commands = torch.zeros((self.num_envs, self.num_commands), device=gs.device, dtype=gs.tc_float)
        self.actions = torch.zeros((self.num_envs, self.num_actions), device=gs.device, dtype=gs.tc_float)
        self.last_actions = torch.zeros_like(self.actions)
        self.dof_pos = torch.zeros_like(self.actions)
        self.dof_vel = torch.zeros_like(self.actions)
        self.default_dof_pos = torch.tensor(
            self.env_cfg["default_joint_angles"],
            device=gs.device,
            dtype=gs.tc_float,
        )
        self.hand.control_dofs_position([self.env_cfg["default_joint_angles"]] * self.num_envs, self.motors_dof_idx)

        # tactile sensor data - dynamically sized based on configured sensors
        self.tactile_data = torch.zeros((self.num_envs, self.total_tactile_dim), device=gs.device, dtype=gs.tc_float)

        # Add tracking for reward components
        self.last_obj_euler = torch.zeros((self.num_envs, 3), device=gs.device, dtype=gs.tc_float)
        self.control_forces = torch.zeros((self.num_envs, self.num_actions), device=gs.device, dtype=gs.tc_float)

        self.extras = dict()  # extra information for logging
        self.extras["observations"] = dict()

        self._resample_commands(torch.arange(self.num_envs, device=gs.device))

    def _resample_commands(self, envs_idx):
        """Resample rotation commands (desired angular velocities around x, y, z axes)"""
        axis_choices = torch.tensor(self.command_cfg["obj_rot_axis"], device=gs.device)
        random_indices = torch.randint(0, len(axis_choices), (len(envs_idx),), device=gs.device)
        self.commands[envs_idx, 0] = axis_choices[random_indices].float()

    def step(self, actions):
        # Store last object euler for rotation tracking
        self.last_obj_euler[:] = self.obj_euler[:]

        # Add action noise if domain randomization is enabled
        if self.rand_cfg is not None:
            action_noise_range = self.rand_cfg["action_noise"]
            action_noise = gs_rand_float(action_noise_range[0], action_noise_range[1], actions.shape, self.device)
            actions = actions + action_noise

        self.actions = torch.clip(actions, -self.env_cfg["clip_actions"], self.env_cfg["clip_actions"])
        target_dof_pos = self.actions * self.env_cfg["action_scale"] + self.dof_pos
        target_dof_pos = torch.clip(target_dof_pos, self.dofs_min_limit, self.dofs_max_limit)
        self.hand.control_dofs_position(target_dof_pos, self.motors_dof_idx)

        # Store control forces for reward computation
        self.control_forces[:] = self.hand.get_dofs_control_force(self.motors_dof_idx)

        self.scene.step()

        if self.data_recorder is not None:
            self.data_recorder.step()

        # update buffers
        self.episode_length_buf += 1
        new_obj_pos = self.obj.get_pos()
        # self.obj_lin_vel[:] = self.obj.get_vel()
        self.obj_lin_vel[:] = (new_obj_pos - self.obj_pos) / self.dt
        self.obj_pos[:] = new_obj_pos
        self.obj_quat[:] = self.obj.get_quat()
        self.obj_euler[:] = quat_to_xyz(self.obj_quat, rpy=True, degrees=True)
        self.obj_ang_vel[:] = self.obj.get_ang()
        self.dof_pos[:] = self.hand.get_dofs_position(self.motors_dof_idx)
        self.dof_vel[:] = self.hand.get_dofs_velocity(self.motors_dof_idx)

        # Add noise if domain randomization is enabled
        if self.rand_cfg is not None:
            joint_noise_range = self.rand_cfg["joint_observation_noise"]
            joint_pos_noise = gs_rand_float(joint_noise_range[0], joint_noise_range[1], self.dof_pos.shape, self.device)
            self.dof_pos = self.dof_pos + joint_pos_noise

        # get tactile sensor data
        for i, sensor in enumerate(self.sensors):
            # sensor_data = torch.clamp(
            #     torch.as_tensor(sensor.read()).flatten(start_dim=1),
            #     -self.env_cfg["force_max_clip"],
            #     self.env_cfg["force_max_clip"],
            # )
            sensor_data = sensor.read()
            if self.env_cfg["tactile_sensor_type"] == "force_magnitude":
                sensor_data = torch.norm(sensor_data, dim=-1)
            # Calculate the start and end indices for this sensor's data
            start_idx = i * self.per_sensor_dim
            end_idx = start_idx + self.per_sensor_dim
            self.tactile_data[:, start_idx:end_idx] = torch.as_tensor(sensor_data).flatten(start_dim=1)

        # resample commands
        envs_idx = (
            (self.episode_length_buf % int(self.env_cfg["resampling_time_s"] / self.dt) == 0)
            .nonzero(as_tuple=False)
            .flatten()
        )
        self._resample_commands(envs_idx)

        # # sanity check / debugging
        # obs_nan_mask = torch.isnan(self.obs_buf).any(dim=1)
        # action_nan_mask = torch.isnan(self.actions).any(dim=1)
        # reward_nan_mask = torch.isnan(self.rew_buf)
        # if obs_nan_mask.any() or action_nan_mask.any() or reward_nan_mask.any():
        #     # Find first NaN environment
        #     nan_envs = torch.logical_or(torch.logical_or(obs_nan_mask, action_nan_mask), reward_nan_mask)
        #     first_nan_env = nan_envs.nonzero(as_tuple=False)[0, 0].item()

        #     print(f"\n=== NaN DETECTED in env {first_nan_env} ===")
        #     print(f"Observations: {self.obs_buf[first_nan_env]}")
        #     print(f"Actions: {self.actions[first_nan_env]}")
        #     print(f"Rewards: {self.rew_buf[first_nan_env]}")
        #     print(f"Object pos: {self.obj_pos[first_nan_env]}")
        #     print(f"Object quat: {self.obj_quat[first_nan_env]}")
        #     print(f"DOF pos: {self.dof_pos[first_nan_env]}")
        #     print(f"DOF vel: {self.dof_vel[first_nan_env]}")
        #     print("===============================\n")

        #     # replace all nan values with zeros
        #     self.obs_buf[obs_nan_mask] = 0.0
        #     self.actions[action_nan_mask] = 0.0

        # compute reward (only if reward functions are defined)
        self.rew_buf[:] = 0.0
        for name, reward_func in self.reward_functions.items():
            rew = reward_func() * self.reward_scales[name]
            self.rew_buf += rew
            self.episode_sums[name] += rew

        # terminate if episode length exceeds max_episode_length
        self.reset_buf = self.episode_length_buf > self.max_episode_length
        # terminate if object falls below z
        self.reset_buf |= self.obj_pos[:, 2] < 0.0
        # ------
        # # terminate if nan detected in actions or observations
        # self.reset_buf |= torch.isnan(self.actions).any(dim=1) | torch.isnan(self.obs_buf).any(dim=1)
        # # terminate if object moves too far from starting pos
        # self.reset_buf |= torch.norm(self.obj_pos - self.obj_base_pos, dim=1) > self.env_cfg["max_obj_distance"]
        # # terminate if object rotation deviates too much from target axis
        # obj_nontarget_axis_angle_diff = torch.abs(self.obj_euler - self.initial_obj_euler)
        # target_axis = self.commands[:, 0].to(dtype=torch.int)
        # env_indices = torch.arange(self.num_envs, device=gs.device)
        # obj_nontarget_axis_angle_diff[env_indices, target_axis] = 0.0
        # obj_nontarget_axis_angle_diff = obj_nontarget_axis_angle_diff * np.pi / 180.0
        # self.reset_buf |= (obj_nontarget_axis_angle_diff > self.env_cfg["max_obj_nontarget_axis_angle_diff"]).any(dim=1)

        time_out_idx = (self.episode_length_buf > self.max_episode_length).nonzero(as_tuple=False).flatten()
        self.extras["time_outs"] = torch.zeros_like(self.reset_buf, device=gs.device, dtype=gs.tc_float)
        self.extras["time_outs"][time_out_idx] = 1.0

        self.reset_idx(self.reset_buf.nonzero(as_tuple=False).flatten())

        # add early termination penalty
        self.rew_buf -= self.reward_cfg.get("early_termination_penalty", 0.0) * self.reset_buf.float()

        # compute observations
        self.obs_buf = torch.cat(
            [
                self.commands,
                unscale(self.dof_pos, self.dofs_min_limit, self.dofs_max_limit),
                self.tactile_data * self.obs_scales["tactile"],
            ],
            axis=-1,
        )
        if self.priv_cfg is not None:
            self.priv_obs_buf = torch.cat(
                [
                    self.obj_pos,
                    self.obj_quat,
                    self.obj_ang_vel,
                ],
                axis=-1,
            )
        self.last_actions[:] = self.actions[:]

        self.extras["observations"]["critic"] = self.get_privileged_observations()

        return self.obs_buf, self.rew_buf, self.reset_buf, self.extras

    def cleanup(self):
        if self.data_recorder is not None:
            self.data_recorder.stop_recording()

    def get_observations(self):
        self.extras["observations"]["critic"] = self.get_privileged_observations()
        return self.obs_buf, self.extras

    def get_privileged_observations(self):
        if self.priv_cfg is None:
            return self.obs_buf

        return torch.cat(
            [
                self.obs_buf,
                self.priv_obs_buf,
            ],
            axis=-1,
        )

    def reset_idx(self, envs_idx):
        if len(envs_idx) == 0:
            return

        # apply domain randomization if enabled
        if self.rand_cfg is not None:
            self._apply_domain_randomization(envs_idx)

        # reset dofs
        self.dof_pos[envs_idx] = self.default_dof_pos
        self.hand.set_dofs_position(
            position=self.dof_pos[envs_idx],
            dofs_idx_local=self.motors_dof_idx,
            zero_velocity=True,
            envs_idx=envs_idx,
        )

        self.obj_quat[envs_idx] = self.obj_init_quat.reshape(1, -1)
        self.obj.set_pos(self.obj_pos[envs_idx], zero_velocity=True, envs_idx=envs_idx)
        self.obj.set_quat(self.obj_quat[envs_idx], zero_velocity=True, envs_idx=envs_idx)
        self.initial_obj_euler[envs_idx] = self.obj_init_euler.reshape(1, -1)

        # reset buffers
        self.last_actions[envs_idx] = 0.0
        self.episode_length_buf[envs_idx] = 0
        self.reset_buf[envs_idx] = True

        # reset tracking variables
        self.last_obj_euler[envs_idx] = self.obj_init_euler.reshape(1, -1)

        # fill extras (only if episode sums exist)
        self.extras["episode"] = {}
        for key in self.episode_sums.keys():
            self.extras["episode"]["rew_" + key] = (
                torch.mean(self.episode_sums[key][envs_idx]).item() / self.env_cfg["episode_length_s"]
            )
            self.episode_sums[key][envs_idx] = 0.0

        self._resample_commands(envs_idx)

    def reset(self):
        self.reset_buf[:] = True
        self.reset_idx(torch.arange(self.num_envs, device=gs.device))
        return self.obs_buf, None

    def _apply_domain_randomization(self, envs_idx):
        """Apply domain randomization for the specified environments"""
        if self.rand_cfg is None:
            return

        n_envs = len(envs_idx)

        self.scene._sim.rigid_solver.set_gravity(
            torch.stack(
                [
                    gs_rand_float(self.rand_cfg["gravity_x"][0], self.rand_cfg["gravity_x"][1], (n_envs,), self.device),
                    gs_rand_float(self.rand_cfg["gravity_y"][0], self.rand_cfg["gravity_y"][1], (n_envs,), self.device),
                    gs_rand_float(self.rand_cfg["gravity_z"][0], self.rand_cfg["gravity_z"][1], (n_envs,), self.device),
                ],
                dim=1,
            )
            .cpu()
            .numpy(),
            envs_idx=envs_idx.cpu(),
        )

        self.obj.set_friction_ratio(
            self.rand_cfg["obj_friction_ratio"][0]
            + torch.rand(n_envs, self.obj.n_links) * self.rand_cfg["obj_friction_ratio"][1],
            links_idx_local=[0],
            envs_idx=envs_idx,
        )
        self.obj.set_mass_shift(
            (self.rand_cfg["obj_mass_shift"][0] + torch.rand(n_envs) * self.rand_cfg["obj_mass_shift"][1]).unsqueeze(
                -1
            ),
            links_idx_local=[0],
            envs_idx=envs_idx,
        )

        self.obj.set_COM_shift(
            (self.rand_cfg["obj_com_shift"][0] + torch.rand(n_envs, 3) * self.rand_cfg["obj_com_shift"][1]).unsqueeze(
                -2
            ),
            links_idx_local=[0],
            envs_idx=envs_idx,
        )

        self.obj_pos[envs_idx] = self.obj_base_pos + torch.stack(
            [
                gs_rand_float(
                    self.rand_cfg["obj_pos_shift_x"][0], self.rand_cfg["obj_pos_shift_x"][1], (n_envs,), self.device
                ),
                gs_rand_float(
                    self.rand_cfg["obj_pos_shift_y"][0], self.rand_cfg["obj_pos_shift_y"][1], (n_envs,), self.device
                ),
                gs_rand_float(
                    self.rand_cfg["obj_pos_shift_z"][0], self.rand_cfg["obj_pos_shift_z"][1], (n_envs,), self.device
                ),
            ],
            dim=1,
        )

        # set_dofs_kp does not currently support different gains for each env, so
        # the same gains are set for all resetting environments
        kp_randomized = self.env_cfg["kp"] * (self.rand_cfg["kp_gain"][0] + torch.rand(1) * self.rand_cfg["kp_gain"][1])
        kd_randomized = self.env_cfg["kd"] * (self.rand_cfg["kd_gain"][0] + torch.rand(1) * self.rand_cfg["kd_gain"][1])
        kp_gains = torch.full((len(self.motors_dof_idx),), kp_randomized.item(), device=self.device)
        kd_gains = torch.full((len(self.motors_dof_idx),), kd_randomized.item(), device=self.device)
        self.hand.set_dofs_kp(kp_gains, self.motors_dof_idx, envs_idx=envs_idx)
        self.hand.set_dofs_kv(kd_gains, self.motors_dof_idx, envs_idx=envs_idx)

    ######################## reward functions ########################

    def _reward_delta_rotation(self):
        """Rotation reward: clip(delta_theta, -c1, c1)"""
        # rotation around target axis
        axis = self.commands[:, 0].to(dtype=torch.int)
        env_indices = torch.arange(self.num_envs, device=gs.device)

        current_rot = self.obj_euler[env_indices, axis]
        last_rot = self.last_obj_euler[env_indices, axis]

        # rotation delta (handle angle wrapping)
        delta_theta = current_rot - last_rot
        delta_theta = torch.where(delta_theta > 180, delta_theta - 360, delta_theta)
        delta_theta = torch.where(delta_theta < -180, delta_theta + 360, delta_theta)

        return torch.clamp(delta_theta, -self.reward_cfg["rot_clip"], self.reward_cfg["rot_clip"])

    def _reward_hand_pose(self):
        """Hand pose penalty: -||q_t - q_init||"""
        return -torch.norm(self.dof_pos - self.default_dof_pos, dim=1)
        # diff = self.dof_pos - self.default_dof_pos
        # print("diff")
        # print(diff)
        # mag = torch.norm(diff, dim=1)
        # print("mag")
        # print(mag)
        # return -mag

    def _reward_work(self):
        """Work penalty: -mean(|tau| * |q_dot|)"""
        work = torch.abs(self.control_forces) * torch.abs(self.dof_vel)
        return -torch.mean(work, dim=1)

    def _reward_torque(self):
        """Torque penalty: -||tau||"""
        return -torch.norm(self.control_forces, dim=1)

    def _reward_obj_linvel(self):
        """Object linear velocity penalty: -||v_t||"""
        return -torch.norm(self.obj_lin_vel, dim=1)

    def _reward_alive(self):
        """Alive reward: 1 - episode_length / max_episode_length"""
        return 1.0 - self.episode_length_buf / self.max_episode_length
