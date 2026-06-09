"""
Render a video of the MultiArmPickPlace scene driven by the heuristic expert, so
the scene design (colored cubes, per-arm target zones, cameras) can be reviewed.

Records new_birdview (global top-down BEV) plus each arm's eye-in-hand camera,
tiled side by side, for one episode.

Example:
    MUJOCO_GL=egl python -m robosuite.scripts.render_multiarm_pickplace \
        --robots Panda Panda --out /path/to/scene.mp4
"""

import argparse
import os

import numpy as np
import cv2

import robosuite as suite
from robosuite.controllers import load_composite_controller_config
from robosuite.scripts.collect_multiarm_heuristic import build_experts, simultaneous_actions, TrafficScheduler


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--robots", nargs="+", type=str, default=["Panda", "Panda"])
    parser.add_argument("--num_cubes", type=int, default=4)
    parser.add_argument("--horizon", type=int, default=700)
    parser.add_argument("--bev_size", type=int, default=384)
    parser.add_argument("--eye_size", type=int, default=192)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--fps", type=int, default=20)
    parser.add_argument("--mode", type=str, default="traffic",
                        choices=["traffic", "simultaneous"],
                        help="traffic-managed concurrent motion (default) or raw simultaneous")
    parser.add_argument("--out", type=str, required=True)
    args = parser.parse_args()

    n_arms = len(args.robots)
    cam_names = ["new_birdview"] + [f"robot{i}_eye_in_hand" for i in range(n_arms)]
    cam_h = [args.bev_size] + [args.eye_size] * n_arms
    cam_w = [args.bev_size] + [args.eye_size] * n_arms

    cc = load_composite_controller_config(controller=None, robot=args.robots[0])
    env = suite.make(
        env_name="MultiArmPickPlace",
        robots=args.robots,
        controller_configs=cc,
        has_renderer=False,
        has_offscreen_renderer=True,
        use_camera_obs=True,
        use_object_obs=True,
        camera_names=cam_names,
        camera_heights=cam_h,
        camera_widths=cam_w,
        control_freq=20,
        ignore_done=True,
        seed=args.seed,
        num_cubes=args.num_cubes,
    )
    obs = env.reset()
    experts = build_experts(env)
    scheduler = TrafficScheduler(experts)
    print("arm_to_color:", env.arm_to_color.tolist())

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    writer = None

    def append_frame(frame):
        nonlocal writer
        if writer is None:
            h, w = frame.shape[:2]
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(args.out, fourcc, args.fps, (w, h))
            if not writer.isOpened():
                raise RuntimeError(f"Could not open video writer for {args.out}")
        writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

    def pad_to_height(img, h):
        if img.shape[0] == h:
            return img
        pad = np.zeros((h - img.shape[0], img.shape[1], 3), dtype=img.dtype)
        return np.concatenate([img, pad], axis=0)

    def grab_frame(obs):
        # robosuite camera obs are stored upside down -> flip vertically
        bev = obs["new_birdview_image"][::-1]
        eyes = [obs[f"robot{i}_eye_in_hand_image"][::-1] for i in range(n_arms)]
        h = bev.shape[0]
        tiles = [bev] + [pad_to_height(e, h) for e in eyes]
        return np.concatenate(tiles, axis=1)

    done_logged = False
    for t in range(args.horizon):
        action = scheduler.action(obs) if args.mode == "traffic" else simultaneous_actions(experts, obs, traffic=False)
        obs, _, _, _ = env.step(action)
        append_frame(grab_frame(obs))

        if env._check_success():
            if not done_logged:
                print(f"success at step {t}")
                done_logged = True
            for _ in range(15):
                actions = [e.hold_act(obs) for e in experts]
                obs, _, _, _ = env.step(np.concatenate(actions))
                append_frame(grab_frame(obs))
            break

    if writer is not None:
        writer.release()
    print("placed:", env._cubes_placed().tolist())
    env.close()
    print("video saved to:", args.out)


if __name__ == "__main__":
    main()
