from collections import OrderedDict

import numpy as np

import robosuite.utils.transform_utils as T
from robosuite.environments.manipulation.multi_arm_env import MultiArmEnv
from robosuite.models.arenas import TableArena
from robosuite.models.objects import BoxObject
from robosuite.models.tasks import ManipulationTask
from robosuite.utils.observables import Observable, sensor
from robosuite.utils.placement_samplers import SequentialCompositeSampler, UniformRandomSampler
from robosuite.utils.transform_utils import convert_quat


class MultiArmBlockLift(MultiArmEnv):
    """
    A tabletop multi-arm block lifting task with configurable robot placement.

    Supports 1-4 single-arm robots arranged clockwise around a table. For two
    robots, env_configuration supports:
        - "adjacent": neighboring sides
        - "opposed": opposite sides

    You can also pass arm_positions explicitly to control exact sides.
    Valid sides are: "west", "south", "east", "north".
    """

    def __init__(
        self,
        robots,
        env_configuration="default",
        controller_configs=None,
        gripper_types="default",
        base_types="default",
        initialization_noise="default",
        table_full_size=(0.9, 0.9, 0.05),
        table_friction=(1.0, 5e-3, 1e-4),
        arm_positions=None,
        position_radius_scale=1.0,
        num_cubes=3,
        cube_size=0.022,
        cube_spawn_range=(0.18, 0.18),
        use_camera_obs=True,
        use_object_obs=True,
        reward_scale=1.0,
        reward_shaping=False,
        placement_initializer=None,
        has_renderer=False,
        has_offscreen_renderer=True,
        render_camera="frontview",
        render_collision_mesh=False,
        render_visual_mesh=True,
        render_gpu_device_id=-1,
        control_freq=20,
        lite_physics=True,
        horizon=1000,
        ignore_done=False,
        hard_reset=True,
        camera_names="agentview",
        camera_heights=256,
        camera_widths=256,
        camera_depths=False,
        camera_segmentations=None,
        renderer="mjviewer",
        renderer_config=None,
        seed=None,
    ):
        self.table_full_size = table_full_size
        self.table_friction = table_friction
        self.table_offset = np.array((0, 0, 0.8))

        self.arm_positions = None if arm_positions is None else list(arm_positions)
        self.position_radius_scale = position_radius_scale

        self.num_cubes = int(num_cubes)
        if self.num_cubes <= 0:
            raise ValueError(f"num_cubes must be positive, got {num_cubes}")
        self.cube_size = float(cube_size)
        self.cube_spawn_range = np.array(cube_spawn_range, dtype=float)

        self.reward_scale = reward_scale
        self.reward_shaping = reward_shaping
        self.use_object_obs = use_object_obs
        self.placement_initializer = placement_initializer

        self.cubes = []
        self.cube_body_ids = {}

        super().__init__(
            robots=robots,
            env_configuration=env_configuration,
            controller_configs=controller_configs,
            base_types=base_types,
            gripper_types=gripper_types,
            initialization_noise=initialization_noise,
            use_camera_obs=use_camera_obs,
            has_renderer=has_renderer,
            has_offscreen_renderer=has_offscreen_renderer,
            render_camera=render_camera,
            render_collision_mesh=render_collision_mesh,
            render_visual_mesh=render_visual_mesh,
            render_gpu_device_id=render_gpu_device_id,
            control_freq=control_freq,
            lite_physics=lite_physics,
            horizon=horizon,
            ignore_done=ignore_done,
            hard_reset=hard_reset,
            camera_names=camera_names,
            camera_heights=camera_heights,
            camera_widths=camera_widths,
            camera_depths=camera_depths,
            camera_segmentations=camera_segmentations,
            renderer=renderer,
            renderer_config=renderer_config,
            seed=seed,
        )

    def reward(self, action=None):
        reward = 0.0

        if self._check_success():
            reward = 2.0
        elif self.reward_shaping:
            max_reaching = 0.0
            grasping = 0.0
            for cube in self.cubes:
                dists = [
                    self._gripper_to_target(robot.gripper, cube.root_body, target_type="body", return_distance=True)
                    for robot in self.robots
                ]
                max_reaching = max(max_reaching, 1.0 - np.tanh(10.0 * min(dists)))
                if any(self._check_grasp(gripper=robot.gripper, object_geoms=cube) for robot in self.robots):
                    grasping = 0.25
            reward = max_reaching + grasping

        if self.reward_scale is not None:
            reward *= self.reward_scale / 2.0

        return reward

    def _load_model(self):
        super()._load_model()

        placements = self._resolve_arm_positions(self.arm_positions)
        for robot, side in zip(self.robots, placements):
            yaw = self._SIDE_TO_YAW[side]
            rot = np.array((0, 0, yaw))
            rot_mat = T.euler2mat(rot)
            default_xpos = np.array(robot.robot_model.base_xpos_offset["table"](self.table_full_size[0]))
            xpos = rot_mat @ default_xpos
            xpos[:2] *= self.position_radius_scale
            robot.robot_model.set_base_xpos(xpos)
            robot.robot_model.set_base_ori(rot)

        mujoco_arena = TableArena(
            table_full_size=self.table_full_size,
            table_friction=self.table_friction,
            table_offset=self.table_offset,
        )
        mujoco_arena.set_origin([0, 0, 0])

        mujoco_arena.set_camera(
            camera_name="agentview0",
            pos=[0.91, 0.0, 1.54],
            quat=[0.6026570796966553, 0.3698708415031433, 0.36987125873565674, 0.6026568412780762],
        )
        mujoco_arena.set_camera(
            camera_name="agentview1",
            pos=[-0.08, 0.89, 1.54],
            quat=[0.023, 0.032, -0.5, -0.867],
        )

        self.cubes = []
        colors = [
            # [0.9, 0.1, 0.1, 1.0],
            # [0.1, 0.6, 0.9, 1.0],
            [0.1, 0.8, 0.2, 1.0],
            # [0.95, 0.8, 0.1, 1.0],
        ]
        for i in range(self.num_cubes):
            color = colors[i % len(colors)]
            cube = BoxObject(
                name=f"cube_{i}",
                size_min=[self.cube_size, self.cube_size, self.cube_size],
                size_max=[self.cube_size, self.cube_size, self.cube_size],
                rgba=color
            )
            self.cubes.append(cube)

        if self.placement_initializer is None:
            self.placement_initializer = SequentialCompositeSampler(name="ObjectSampler")
            for i in range(self.num_cubes):
                self.placement_initializer.append_sampler(
                    sampler=UniformRandomSampler(
                        name=f"CubeSampler{i}",
                        x_range=[-self.cube_spawn_range[0], self.cube_spawn_range[0]],
                        y_range=[-self.cube_spawn_range[1], self.cube_spawn_range[1]],
                        rotation=None,
                        ensure_object_boundary_in_range=False,
                        ensure_valid_placement=True,
                        reference_pos=self.table_offset,
                        z_offset=0.01,
                    )
                )

        self.placement_initializer.reset()
        if isinstance(self.placement_initializer, SequentialCompositeSampler):
            for i, cube in enumerate(self.cubes):
                self.placement_initializer.add_objects_to_sampler(sampler_name=f"CubeSampler{i}", mujoco_objects=cube)
        else:
            for cube in self.cubes:
                self.placement_initializer.add_objects(cube)

        self.model = ManipulationTask(
            mujoco_arena=mujoco_arena,
            mujoco_robots=[robot.robot_model for robot in self.robots],
            mujoco_objects=self.cubes,
        )

    def _setup_references(self):
        super()._setup_references()
        self.cube_body_ids = {cube.name: self.sim.model.body_name2id(cube.root_body) for cube in self.cubes}

    def _setup_observables(self):
        observables = super()._setup_observables()

        if self.use_object_obs:
            modality = "object"
            sensors = []

            for cube in self.cubes:

                @sensor(modality=modality)
                def cube_pos(obs_cache, cube_key=cube.name):
                    return np.array(self.sim.data.body_xpos[self.cube_body_ids[cube_key]])

                @sensor(modality=modality)
                def cube_quat(obs_cache, cube_key=cube.name):
                    return convert_quat(np.array(self.sim.data.body_xquat[self.cube_body_ids[cube_key]]), to="xyzw")

                cube_pos.__name__ = f"{cube.name}_pos"
                cube_quat.__name__ = f"{cube.name}_quat"
                sensors.extend([cube_pos, cube_quat])

            for s in sensors:
                observables[s.__name__] = Observable(
                    name=s.__name__,
                    sensor=s,
                    sampling_rate=self.control_freq,
                )

        return observables

    def _reset_internal(self):
        super()._reset_internal()

        if not self.deterministic_reset:
            object_placements = self.placement_initializer.sample()
            for obj_pos, obj_quat, obj in object_placements.values():
                self.sim.data.set_joint_qpos(obj.joints[0], np.concatenate([np.array(obj_pos), np.array(obj_quat)]))

    def visualize(self, vis_settings):
        super().visualize(vis_settings=vis_settings)

        if vis_settings["grippers"] and self.cubes:
            target_cube = self.cubes[0]
            self._visualize_gripper_to_target(gripper=self.robots[0].gripper, target=target_cube)

    def _check_success(self):
        table_height = self.model.mujoco_arena.table_offset[2]
        for cube in self.cubes:
            cube_height = self.sim.data.body_xpos[self.cube_body_ids[cube.name]][2]
            if cube_height > table_height + 0.04:
                return True
        return False
