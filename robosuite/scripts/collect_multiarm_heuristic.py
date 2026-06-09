"""
Scripted heuristic expert that collects multi-arm, color-conditioned
table-clearing (pick-and-place) demonstrations for the ``MultiArmPickPlace``
environment, using robosuite's official ``DataCollectionWrapper`` +
``gather_demonstrations_as_hdf5`` so the output is a standard robomimic-style
``demo.hdf5`` (states + actions only; images are added afterwards with
robomimic's ``dataset_states_to_obs.py``).

Each arm i is assigned a color this episode (env.arm_to_color[i]); it picks the
cube of that color and places it into its own side's target zone
(env.zone_positions[i]). State machine per arm:
    APPROACH -> DESCEND -> GRASP -> LIFT -> TRANSPORT -> PLACE -> RELEASE -> DONE
under the default OSC_POSE *delta* controller (per-arm action =
[dx,dy,dz, drx,dry,drz, gripper], 7-dim).

Example:
    python collect_multiarm_heuristic.py \
        --robots Panda Panda Panda Panda \
        --num_episodes 50 --horizon 600 \
        --directory ~/data/multiarm_pickplace

Then extract image observations (run from the robomimic repo):
    python robomimic/scripts/dataset_states_to_obs.py \
        --dataset <out>/demo.hdf5 --output_name image.hdf5 --done_mode 2 \
        --camera_names new_birdview robot0_eye_in_hand robot1_eye_in_hand \
        robot2_eye_in_hand robot3_eye_in_hand --camera_height 96 --camera_width 96
"""

import argparse
import json
import os
import time

import numpy as np

import robosuite as suite
from robosuite.controllers import load_composite_controller_config
from robosuite.scripts.collect_human_demonstrations import gather_demonstrations_as_hdf5
from robosuite.wrappers import DataCollectionWrapper

POS_GAIN = 8.0           # P gain: position error (m) -> normalized OSC delta action
HOVER_HEIGHT = 0.12      # hover height above a cube before descending
GRASP_Z_OFFSET = 0.004   # target height above cube centre when grasping
CARRY_HEIGHT = 0.20      # carry height above the table top while transporting
PLACE_Z_OFFSET = 0.03    # height above table top when releasing
XY_ALIGN_TOL = 0.035     # xy tolerance to consider aligned above a target
GRASP_DIST_TOL = 0.05    # distance to cube to start closing the gripper
ZONE_XY_TOL = 0.03       # xy tolerance over the zone to start lowering
GRASP_HOLD_STEPS = 8
RELEASE_HOLD_STEPS = 6
GRIPPER_CLOSE = 1.0      # Panda: +1 closes, -1 opens
GRIPPER_OPEN = -1.0

# distinct drop spots within a zone so multiple same-color cubes don't stack
DROP_OFFSETS = [(0.0, 0.0), (0.045, 0.045), (-0.045, -0.045),
                (0.045, -0.045), (-0.045, 0.045)]

# collision avoidance: if two end-effectors come within DANGER_R (xy), the
# lower-priority arm yields by lifting above and backing off. In traffic mode,
# only one arm is allowed to occupy the central pickup workspace at a time; the
# others visibly stage above their own goal zones until the centre is clear.
DANGER_R = 0.22
SHARED_WORKSPACE_R = 0.34
YIELD_LIFT = 0.12        # extra height (above carry) the yielding arm rises to
YIELD_BACKOFF = 0.12     # how far the yielding arm backs away from the intruder


class ArmExpert:
    """
    Clears all cubes of one color to this arm's zone. Each time it is free, it
    selects the easiest-to-grab remaining same-color cube (nearest to the current
    end-effector) and runs a pick-place state machine on it; repeats until none
    of its color remain unplaced.
    """
    APPROACH, DESCEND, GRASP, LIFT, TRANSPORT, PLACE, RELEASE = range(7)

    def __init__(self, arm_idx, cube_indices, zone_pos, table_top_z, base_yaw,
                 place_xy_tol, place_z_tol):
        self.arm_idx = arm_idx
        self.cube_indices = list(cube_indices)  # cubes of this arm's color
        self.zone_pos = np.asarray(zone_pos, dtype=float)
        self.table_top_z = table_top_z
        self.place_xy_tol = place_xy_tol
        self.place_z_tol = place_z_tol
        self.target = None
        self.phase = self.APPROACH
        self.counter = 0
        self.finished = False
        self.placed_count = 0  # how many of this color already dropped (for drop offset)
        # OSC_POSE uses base-frame deltas (input_ref_frame="base"); map a
        # world-frame position error into this arm's base frame via Rz(-yaw).
        c, s = np.cos(-base_yaw), np.sin(-base_yaw)
        self.world_to_base = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])

    def _pos_action(self, target, eef, gripper):
        base_err = self.world_to_base @ (np.asarray(target) - np.asarray(eef))
        pos_delta = np.clip(POS_GAIN * base_err, -1.0, 1.0)
        return np.concatenate([pos_delta, np.zeros(3), [gripper]])

    def _eef(self, obs):
        return np.asarray(obs[f"robot{self.arm_idx}_eef_pos"], dtype=float)

    def _cube_pos(self, obs, k):
        return np.asarray(obs[f"cube_{k}_pos"], dtype=float)

    def _is_placed(self, obs, k):
        c = self._cube_pos(obs, k)
        return (np.linalg.norm(c[:2] - self.zone_pos[:2]) < self.place_xy_tol
                and c[2] < self.table_top_z + self.place_z_tol)

    def _select_target(self, obs):
        """Pick the easiest (nearest) unplaced same-color cube; else finish."""
        eef = self._eef(obs)
        remaining = [(np.linalg.norm(self._cube_pos(obs, k)[:2] - eef[:2]), k)
                     for k in self.cube_indices if not self._is_placed(obs, k)]
        if not remaining:
            self.finished = True
            self.target = None
            return
        self.target = min(remaining)[1]
        self.phase = self.APPROACH
        self.counter = 0

    def act(self, obs):
        eef = self._eef(obs)
        if self.target is None and not self.finished:
            self._select_target(obs)
        if self.finished:
            return self._hold_above_zone(eef, GRIPPER_OPEN)

        cube = self._cube_pos(obs, self.target)
        off = DROP_OFFSETS[self.placed_count % len(DROP_OFFSETS)]
        drop_xy = np.array([self.zone_pos[0] + off[0], self.zone_pos[1] + off[1]])
        gripper = GRIPPER_OPEN
        target = eef.copy()

        if self.phase == self.APPROACH:
            target = np.array([cube[0], cube[1], cube[2] + HOVER_HEIGHT])
            if np.linalg.norm(eef[:2] - cube[:2]) < XY_ALIGN_TOL:
                self.phase = self.DESCEND
        elif self.phase == self.DESCEND:
            target = np.array([cube[0], cube[1], cube[2] + GRASP_Z_OFFSET])
            if np.linalg.norm(eef - target) < GRASP_DIST_TOL:
                self.phase = self.GRASP
                self.counter = 0
        elif self.phase == self.GRASP:
            target = np.array([cube[0], cube[1], cube[2] + GRASP_Z_OFFSET])
            gripper = GRIPPER_CLOSE
            self.counter += 1
            if self.counter >= GRASP_HOLD_STEPS:
                self.phase = self.LIFT
        elif self.phase == self.LIFT:
            target = np.array([cube[0], cube[1], self.table_top_z + CARRY_HEIGHT])
            gripper = GRIPPER_CLOSE
            if eef[2] > self.table_top_z + CARRY_HEIGHT * 0.7:
                self.phase = self.TRANSPORT
        elif self.phase == self.TRANSPORT:
            target = np.array([drop_xy[0], drop_xy[1], self.table_top_z + CARRY_HEIGHT])
            gripper = GRIPPER_CLOSE
            if np.linalg.norm(eef[:2] - drop_xy) < ZONE_XY_TOL:
                self.phase = self.PLACE
        elif self.phase == self.PLACE:
            target = np.array([drop_xy[0], drop_xy[1], self.table_top_z + PLACE_Z_OFFSET])
            gripper = GRIPPER_CLOSE
            if eef[2] < self.table_top_z + PLACE_Z_OFFSET + 0.02:
                self.phase = self.RELEASE
                self.counter = 0
        elif self.phase == self.RELEASE:
            target = np.array([drop_xy[0], drop_xy[1], self.table_top_z + PLACE_Z_OFFSET])
            gripper = GRIPPER_OPEN
            self.counter += 1
            if self.counter >= RELEASE_HOLD_STEPS:
                # done with this cube; choose the next one (or finish)
                self.placed_count += 1
                self.target = None
                return self._hold_above_zone(eef, GRIPPER_OPEN)

        return self._pos_action(target, eef, gripper)

    def _hold_above_zone(self, eef, gripper):
        target = np.array([self.zone_pos[0], self.zone_pos[1], self.table_top_z + CARRY_HEIGHT])
        return self._pos_action(target, eef, gripper)

    def is_carrying(self):
        """Holding a cube (closed gripper, mid pick-place)."""
        return self.target is not None and self.phase in (self.GRASP, self.LIFT, self.TRANSPORT, self.PLACE)

    def priority(self):
        """Higher tuple = higher priority. Carrying arms win; ties broken by
        lower arm index (so exactly one arm yields in any conflict -> no deadlock)."""
        return (1 if self.is_carrying() else 0, -self.arm_idx)

    def yield_act(self, obs, intruder_xy):
        """Visible avoidance: lift above and back away from the intruder, holding
        any carried cube. Phase is NOT advanced, so the task resumes once clear."""
        eef = self._eef(obs)
        away = eef[:2] - np.asarray(intruder_xy)[:2]
        norm = np.linalg.norm(away)
        away = away / norm if norm > 1e-6 else np.array([0.0, 0.0])
        gripper = GRIPPER_CLOSE if self.is_carrying() else GRIPPER_OPEN
        target = np.array([eef[0] + YIELD_BACKOFF * away[0],
                           eef[1] + YIELD_BACKOFF * away[1],
                           self.table_top_z + CARRY_HEIGHT + YIELD_LIFT])
        return self._pos_action(target, eef, gripper)

    def retreat_act(self, obs):
        eef = self._eef(obs)
        gripper = GRIPPER_CLOSE if self.phase in (self.LIFT, self.TRANSPORT, self.PLACE) \
            and self.target is not None else GRIPPER_OPEN
        target = np.array([self.zone_pos[0], self.zone_pos[1], self.table_top_z + CARRY_HEIGHT + 0.06])
        return self._pos_action(target, eef, gripper)

    def stage_act(self, obs):
        """Park above this arm's own drop zone while another arm owns centre."""
        eef = self._eef(obs)
        gripper = GRIPPER_CLOSE if self.is_carrying() else GRIPPER_OPEN
        target = np.array([
            self.zone_pos[0],
            self.zone_pos[1],
            self.table_top_z + CARRY_HEIGHT + YIELD_LIFT,
        ])
        return self._pos_action(target, eef, gripper)

    def hold_act(self, obs):
        eef = self._eef(obs)
        return self._pos_action(eef, eef, GRIPPER_OPEN)

    def ensure_target(self, obs):
        if self.target is None and not self.finished:
            self._select_target(obs)

    def needs_shared_workspace(self, obs):
        """True while this arm needs the central pickup corridor.

        Cubes spawn around the table centre, so simultaneous approach / grasp
        from multiple sides is the common collision case. Keep the reservation
        until a carrying arm has lifted and moved out of the centre.
        """
        self.ensure_target(obs)
        if self.finished or self.target is None:
            return False
        eef = self._eef(obs)
        cube = self._cube_pos(obs, self.target)
        if self.phase in (self.APPROACH, self.DESCEND, self.GRASP, self.LIFT):
            return np.linalg.norm(cube[:2]) < SHARED_WORKSPACE_R
        if self.phase == self.TRANSPORT:
            return np.linalg.norm(eef[:2]) < SHARED_WORKSPACE_R
        return False

    def traffic_priority(self, obs):
        """Higher tuple owns the central workspace."""
        eef = self._eef(obs)
        cube_dist = 0.0
        if self.target is not None:
            cube_dist = np.linalg.norm(eef[:2] - self._cube_pos(obs, self.target)[:2])
        phase_rank = {
            self.TRANSPORT: 5,
            self.LIFT: 4,
            self.GRASP: 3,
            self.DESCEND: 2,
            self.APPROACH: 1,
            self.PLACE: 0,
            self.RELEASE: 0,
        }[self.phase]
        return (1 if self.is_carrying() else 0, phase_rank, -cube_dist, -self.arm_idx)


def build_experts(env):
    """One expert per arm. Arm i is assigned a color (env.arm_to_color[i]) and
    must clear every cube of that color (cube k has color k % n_colors) to its
    own zone (env.zone_positions[i])."""
    n_arms = len(env.robots)
    sides = env._resolve_arm_positions(env.arm_positions)
    experts = []
    for i in range(n_arms):
        color_id = int(env.arm_to_color[i])
        cube_indices = [k for k in range(env.num_cubes) if env.cube_color_ids[k] == color_id]
        base_yaw = env._SIDE_TO_YAW[sides[i]]
        experts.append(ArmExpert(
            arm_idx=i, cube_indices=cube_indices, zone_pos=env.zone_positions[i],
            table_top_z=env.table_top_z, base_yaw=base_yaw,
            place_xy_tol=env.place_xy_tol, place_z_tol=env.place_z_tol))
    return experts


def _shared_workspace_owner(experts, obs, preferred_owner=None):
    candidates = [e for e in experts if e.needs_shared_workspace(obs)]
    if not candidates:
        return None
    if preferred_owner is not None and preferred_owner in candidates:
        return preferred_owner
    return max(candidates, key=lambda e: e.traffic_priority(obs))


def simultaneous_actions(experts, obs, traffic=True, preferred_owner=None):
    """All arms act concurrently, with priority-based collision avoidance: if two
    end-effectors are within DANGER_R (xy), the lower-priority arm yields (lifts +
    backs off) while the higher-priority arm proceeds. In traffic mode, the
    central pickup region is also reserved by one arm at a time."""
    n = len(experts)
    eefs = [np.asarray(obs[f"robot{i}_eef_pos"], dtype=float) for i in range(n)]
    owner = _shared_workspace_owner(experts, obs, preferred_owner=preferred_owner) if traffic else None
    actions = []
    for i, e in enumerate(experts):
        if traffic and owner is not None and e is not owner and e.needs_shared_workspace(obs):
            actions.append(e.stage_act(obs))
            continue
        intruder = None
        best = None
        for j in range(n):
            if j == i:
                continue
            if np.linalg.norm(eefs[i][:2] - eefs[j][:2]) < DANGER_R \
                    and experts[j].priority() > e.priority():
                d = np.linalg.norm(eefs[i][:2] - eefs[j][:2])
                if best is None or d < best:
                    best, intruder = d, eefs[j]
        actions.append(e.yield_act(obs, intruder) if intruder is not None else e.act(obs))
    return np.concatenate(actions)


class TrafficScheduler:
    """Round-robin reservation for the central pickup workspace."""

    def __init__(self, experts):
        self.experts = experts
        self.next_arm = 0
        self.owner = None

    def action(self, obs):
        candidates = [e for e in self.experts if e.needs_shared_workspace(obs)]
        if not candidates:
            self.owner = None
            return simultaneous_actions(self.experts, obs, traffic=True)

        if self.owner not in candidates:
            by_idx = {e.arm_idx: e for e in candidates}
            for offset in range(len(self.experts)):
                idx = (self.next_arm + offset) % len(self.experts)
                if idx in by_idx:
                    self.owner = by_idx[idx]
                    self.next_arm = (idx + 1) % len(self.experts)
                    break

        return simultaneous_actions(self.experts, obs, traffic=True, preferred_owner=self.owner)


def _geom_robot_index(model, geom_id, n_robots):
    name = model.geom_id2name(geom_id)
    if not name:
        return None
    for i in range(n_robots):
        if name.startswith(f"robot{i}_") or name.startswith(f"gripper{i}_"):
            return i
        if name.startswith(f"gripper{i}_right") or name.startswith(f"gripper{i}_left"):
            return i
    return None


def inter_arm_collision(env):
    """Returns (hit, details) for contacts between different robot instances."""
    base_env = getattr(env, "unwrapped", env)
    sim = base_env.sim
    n_robots = len(base_env.robots)
    for cidx in range(sim.data.ncon):
        contact = sim.data.contact[cidx]
        r1 = _geom_robot_index(sim.model, contact.geom1, n_robots)
        r2 = _geom_robot_index(sim.model, contact.geom2, n_robots)
        if r1 is not None and r2 is not None and r1 != r2:
            g1 = sim.model.geom_id2name(contact.geom1)
            g2 = sim.model.geom_id2name(contact.geom2)
            return True, f"robot{r1}:{g1} <-> robot{r2}:{g2}"
    return False, None


def run_episode(env, horizon, mode="traffic", abort_on_collision=True):
    """Roll out the per-arm experts.

    mode="traffic" (default): arms run concurrently except for the central pickup
    workspace, which is reserved by one arm while others stage visibly. mode=
    "simultaneous": every arm advances its own pick-place with only local eef
    avoidance. mode="sequential": advance one arm at a time while others retreat.
    """
    obs = env.reset()
    experts = build_experts(getattr(env, "unwrapped", env))
    scheduler = TrafficScheduler(experts)
    n = len(experts)
    active, steps_on_active = 0, 0
    max_steps_per_arm = max(1, (horizon - 20) // n)

    success = False
    for _ in range(horizon):
        if mode in ("traffic", "simultaneous"):
            action = scheduler.action(obs) if mode == "traffic" else simultaneous_actions(experts, obs, traffic=False)
        else:
            acts = [experts[i].act(obs) if i == active else experts[i].retreat_act(obs)
                    for i in range(n)]
            steps_on_active += 1
            if active < n and (experts[active].finished
                               or steps_on_active >= max_steps_per_arm):
                active += 1
                steps_on_active = 0
            action = np.concatenate(acts)
        obs, _, _, _ = env.step(action)

        if abort_on_collision:
            collided, details = inter_arm_collision(env)
            if collided:
                print(f"  aborting episode: inter-arm collision detected ({details})")
                return False

        if env._check_success():
            success = True
            for _ in range(10):  # hold still so the success state is robustly logged
                actions = [e.hold_act(obs) for e in experts]
                obs, _, _, _ = env.step(np.concatenate(actions))
                if abort_on_collision:
                    collided, details = inter_arm_collision(env)
                    if collided:
                        print(f"  aborting episode: inter-arm collision detected ({details})")
                        return False
            break
    return success


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--robots", nargs="+", type=str,
                        default=["Panda", "Panda"],
                        help="start with 2 arms (opposed); scale up once data looks clean")
    parser.add_argument("--num_episodes", type=int, default=50)
    parser.add_argument("--num_cubes", type=int, default=4,
                        help="total cubes (>= num arms); colors = num arms, several cubes per color")
    parser.add_argument("--horizon", type=int, default=700)
    parser.add_argument("--mode", type=str, default="traffic",
                        choices=["traffic", "simultaneous", "sequential"],
                        help="traffic-managed concurrent motion (default), raw simultaneous, or one arm at a time")
    parser.add_argument("--abort-on-collision", action=argparse.BooleanOptionalAction, default=True,
                        help="discard episodes with any detected inter-arm contact")
    parser.add_argument("--controller", type=str, default=None,
                        help="default = robot's default composite controller (OSC_POSE delta)")
    parser.add_argument("--camera", nargs="*", type=str, default="new_birdview")
    parser.add_argument("--directory", type=str,
                        default=os.path.join(suite.models.assets_root, "demonstrations_multiarm_pickplace"))
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    np.random.seed(args.seed)

    controller_config = load_composite_controller_config(
        controller=args.controller, robot=args.robots[0])
    config = {
        "env_name": "MultiArmPickPlace",
        "robots": args.robots,
        "controller_configs": controller_config,
    }
    env = suite.make(
        **config,
        has_renderer=args.render,
        has_offscreen_renderer=False,
        render_camera=args.camera,
        use_camera_obs=False,
        use_object_obs=True,
        reward_shaping=True,
        control_freq=20,
        ignore_done=True,
        seed=args.seed,
        num_cubes=args.num_cubes,
    )
    env_info = json.dumps(config)

    tmp_directory = "/tmp/multiarm_pp_{}".format(str(time.time()).replace(".", "_"))
    env = DataCollectionWrapper(env, tmp_directory, flush_freq=args.horizon + 50)

    n_success = 0
    for ep in range(args.num_episodes):
        ok = run_episode(env, args.horizon, mode=args.mode, abort_on_collision=args.abort_on_collision)
        n_success += int(ok)
        print(f"[ep {ep + 1}/{args.num_episodes}] success={ok} (cumulative {n_success}/{ep + 1})")
    env.close()

    os.makedirs(args.directory, exist_ok=True)
    t1, t2 = str(time.time()).split(".")
    out_dir = os.path.join(args.directory, f"{t1}_{t2}")
    os.makedirs(out_dir)
    gather_demonstrations_as_hdf5(tmp_directory, out_dir, env_info)
    print(f"\nCollected {n_success}/{args.num_episodes} successful demos.")
    print(f"demo.hdf5 written under: {out_dir}")


if __name__ == "__main__":
    main()
