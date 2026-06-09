from collections import OrderedDict

import numpy as np

import robosuite.utils.transform_utils as T
from robosuite.environments.manipulation.multi_arm_block_lift import MultiArmBlockLift
from robosuite.environments.manipulation.multi_arm_env import MultiArmEnv
from robosuite.models.arenas import TableArena
from robosuite.models.objects import BoxObject
from robosuite.models.tasks import ManipulationTask
from robosuite.utils.mjcf_utils import new_site
from robosuite.utils.observables import Observable, sensor
from robosuite.utils.placement_samplers import SequentialCompositeSampler, UniformRandomSampler
from robosuite.utils.transform_utils import convert_quat

# Fixed cube identity: cube k always has color PALETTE[k]. Target zones get
# recolored per episode to encode the (randomized) arm -> color assignment.
PALETTE = [
    ("red", [0.90, 0.10, 0.10, 1.0]),
    ("blue", [0.10, 0.40, 0.90, 1.0]),
    ("green", [0.10, 0.80, 0.20, 1.0]),
    ("yellow", [0.95, 0.80, 0.10, 1.0]),
]


class MultiArmPickPlace(MultiArmEnv):
    """
    Multi-arm, color-conditioned table-clearing (pick-and-place) task.

    Setup (1-4 single-arm robots around a table):
        - There are exactly N cubes (N = number of arms), each a distinct color
          (cube k is PALETTE[k]).
        - Each arm i owns a fixed target zone (a flat colored marker) on its own
          side of the table.
        - Every episode a random permutation assigns arm i a color; arm i's zone
          is recolored to that color. The arm must pick the cube of its assigned
          color and place it into its (matching-colored) zone.
        - Success = every cube resting inside the zone of its matching color
          ("clear the table" by color). The arm->color map varies across episodes.

    The arm->color assignment is encoded in the model XML (zone marker rgba), so
    it is reproduced deterministically when demonstrations are replayed for
    observation extraction.
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
        num_cubes=None,
        cube_size=0.022,
        cube_spawn_range=(0.20, 0.20),
        zone_radius=0.30,
        zone_half_size=0.08,
        place_xy_tol=0.08,
        place_z_tol=0.08,
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

        n_arms = len(robots) if isinstance(robots, (list, tuple)) else 1
        if n_arms > len(PALETTE):
            raise ValueError(f"Only {len(PALETTE)} colors defined; cannot support {n_arms} arms.")
        # one color per arm; there can be MORE cubes than arms (several cubes per
        # color). cube k has color (k % n_colors).
        self.n_colors = n_arms
        self.num_cubes = int(num_cubes) if num_cubes is not None else 2 * n_arms
        if self.num_cubes < n_arms:
            raise ValueError(f"num_cubes ({self.num_cubes}) must be >= num arms ({n_arms}).")
        self.cube_color_ids = [k % self.n_colors for k in range(self.num_cubes)]
        self.cube_size = float(cube_size)
        self.cube_spawn_range = np.array(cube_spawn_range, dtype=float)

        self.zone_radius = float(zone_radius)
        self.zone_half_size = float(zone_half_size)
        self.place_xy_tol = float(place_xy_tol)
        self.place_z_tol = float(place_z_tol)

        self.reward_scale = reward_scale
        self.reward_shaping = reward_shaping
        self.use_object_obs = use_object_obs
        self.placement_initializer = placement_initializer

        self.cubes = []
        self.cube_body_ids = {}
        self.zone_names = []
        self.zone_positions = None        # (N, 3) world positions of zones, by arm index
        self.arm_to_color = None          # arm i -> color id assigned this episode
        self.color_to_zone = None         # color id -> arm/zone index

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

    @property
    def table_top_z(self):
        return float(self.table_offset[2]) + float(self.table_full_size[2]) / 2.0

    def reward(self, action=None):
        placed = self._cubes_placed()
        reward = float(np.sum(placed))  # +1 per correctly placed cube
        if self.reward_scale is not None:
            reward *= self.reward_scale / max(self.num_cubes, 1)
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

        # global top-down bird's-eye + per-arm shoulder cameras (same as block lift)
        mujoco_arena.set_camera(
            camera_name="new_birdview",
            pos=[0.0, 0.0, 2.4],
            quat=[0.7071067690849304, 0.0, 0.0, 0.7071067690849304],
        )
        shoulder_offset_local = np.array([-0.5, 0.85, 1.55])
        look_at_target = self.table_offset + np.array([0.0, 0.0, 0.15])
        for i, side in enumerate(placements):
            yaw = self._SIDE_TO_YAW[side]
            rot_mat = T.euler2mat(np.array((0, 0, yaw)))
            cam_pos = rot_mat @ shoulder_offset_local
            cam_quat = MultiArmBlockLift._lookat_quat(cam_pos, look_at_target)
            mujoco_arena.set_camera(
                camera_name=f"robot{i}_shouldercamera",
                pos=cam_pos.tolist(),
                quat=cam_quat.tolist(),
            )

        # cubes: color of cube k is PALETTE[k % n_colors]
        self.cubes = []
        for k in range(self.num_cubes):
            cube = BoxObject(
                name=f"cube_{k}",
                size_min=[self.cube_size] * 3,
                size_max=[self.cube_size] * 3,
                rgba=PALETTE[self.cube_color_ids[k]][1],
                rng=self.rng,
            )
            self.cubes.append(cube)

        # per-episode random arm -> color assignment, encoded in zone marker colors
        self.arm_to_color = np.array(self.rng.permutation(self.n_colors), dtype=int)
        self.color_to_zone = np.argsort(self.arm_to_color)  # color id -> arm/zone index

        # target zone markers (flat colored boxes) on each arm's side
        self.zone_names = []
        zone_pos_list = []
        for i, side in enumerate(placements):
            yaw = self._SIDE_TO_YAW[side]
            rot_mat = T.euler2mat(np.array((0, 0, yaw)))
            zone_pos = rot_mat @ np.array([-self.zone_radius, 0.0, 0.0]) + self.table_offset
            zone_pos[2] = self.table_top_z + 0.001
            zone_pos_list.append(zone_pos)
            color_id = int(self.arm_to_color[i])
            zone_name = f"zone_{i}"
            self.zone_names.append(zone_name)
            site = new_site(
                name=zone_name,
                pos=zone_pos.tolist(),
                size=[self.zone_half_size, self.zone_half_size, 0.001],
                rgba=PALETTE[color_id][1],
                type="box",
                group="1",
            )
            mujoco_arena.worldbody.append(site)
        self.zone_positions = np.array(zone_pos_list)

        # cube placement sampler: spawn near table centre
        if self.placement_initializer is None:
            self.placement_initializer = SequentialCompositeSampler(name="ObjectSampler")
            for k in range(self.num_cubes):
                self.placement_initializer.append_sampler(
                    sampler=UniformRandomSampler(
                        name=f"CubeSampler{k}",
                        x_range=[-self.cube_spawn_range[0], self.cube_spawn_range[0]],
                        y_range=[-self.cube_spawn_range[1], self.cube_spawn_range[1]],
                        rotation=None,
                        ensure_object_boundary_in_range=False,
                        ensure_valid_placement=True,
                        reference_pos=self.table_offset,
                        z_offset=0.01,
                        rng=self.rng,
                    )
                )
        self.placement_initializer.reset()
        if isinstance(self.placement_initializer, SequentialCompositeSampler):
            for k, cube in enumerate(self.cubes):
                self.placement_initializer.add_objects_to_sampler(sampler_name=f"CubeSampler{k}", mujoco_objects=cube)
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
        # recover zone world positions and the arm->color assignment from the
        # loaded model (robust to XML replay where _load_model isn't re-run)
        zone_pos, arm_to_color = [], []
        palette_rgba = np.array([c[1] for c in PALETTE])
        for i in range(self.n_colors):
            sid = self.sim.model.site_name2id(f"zone_{i}")
            zone_pos.append(np.array(self.sim.model.site_pos[sid]))
            rgba = np.array(self.sim.model.site_rgba[sid])
            color_id = int(np.argmin(np.linalg.norm(palette_rgba - rgba, axis=1)))
            arm_to_color.append(color_id)
        self.zone_positions = np.array(zone_pos)
        self.arm_to_color = np.array(arm_to_color, dtype=int)
        self.color_to_zone = np.argsort(self.arm_to_color)

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

            # per-arm goal: the xy position and color id of that arm's target zone
            for i in range(self.n_colors):

                @sensor(modality=modality)
                def goal_pos(obs_cache, arm_idx=i):
                    return np.array(self.zone_positions[arm_idx])

                @sensor(modality=modality)
                def goal_color(obs_cache, arm_idx=i):
                    return np.array([float(self.arm_to_color[arm_idx])])

                goal_pos.__name__ = f"robot{i}_goal_pos"
                goal_color.__name__ = f"robot{i}_goal_color"
                sensors.extend([goal_pos, goal_color])

            for s in sensors:
                observables[s.__name__] = Observable(name=s.__name__, sensor=s, sampling_rate=self.control_freq)

        return observables

    def _reset_internal(self):
        super()._reset_internal()
        if not self.deterministic_reset:
            object_placements = self.placement_initializer.sample()
            for obj_pos, obj_quat, obj in object_placements.values():
                self.sim.data.set_joint_qpos(obj.joints[0], np.concatenate([np.array(obj_pos), np.array(obj_quat)]))

    def _cubes_placed(self):
        """Boolean array: is cube k resting inside the zone of its matching color?"""
        placed = np.zeros(self.num_cubes, dtype=bool)
        for k in range(self.num_cubes):
            color_id = self.cube_color_ids[k]
            zone_idx = int(self.color_to_zone[color_id])
            target = self.zone_positions[zone_idx]
            cube_pos = self.sim.data.body_xpos[self.cube_body_ids[f"cube_{k}"]]
            xy_ok = np.linalg.norm(cube_pos[:2] - target[:2]) < self.place_xy_tol
            z_ok = cube_pos[2] < self.table_top_z + self.place_z_tol
            placed[k] = xy_ok and z_ok
        return placed

    def _check_success(self):
        return bool(np.all(self._cubes_placed()))
