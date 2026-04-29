"""
Record video of agent episodes with the imageio library.
This script uses offscreen rendering.

Example:
    $ python demo_video_recording.py --environment Lift --robots Panda
"""
import os
import argparse

import imageio
import numpy as np

import robosuite.macros as macros
from robosuite import make

# Set the image convention to opencv so that the images are automatically rendered "right side up" when using imageio
# (which uses opencv convention)
macros.IMAGE_CONVENTION = "opencv"

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--environment", type=str, default="Stack")
    parser.add_argument("--robots", nargs="+", type=str, default="Panda", help="Which robot(s) to use in the env")
    parser.add_argument("--camera", nargs="+", default=["agentview"], help="Name(s) of camera(s) to render")
    parser.add_argument("--video_path", type=str, default="video.mp4")
    parser.add_argument("--root", type=str, default=".", help="Root directory to save video to")
    parser.add_argument("--timesteps", type=int, default=500)
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--skip_frame", type=int, default=1)
    args = parser.parse_args()

    args.root = os.path.join(args.root, args.environment + "_" + "_".join(args.robots))
    if not os.path.exists(args.root):
        os.makedirs(args.root)

    camera_names = list(args.camera)
    if len(camera_names) == 1:
        camera_heights = args.height
        camera_widths = args.width
    else:
        camera_heights = [args.height for _ in camera_names]
        camera_widths = [args.width for _ in camera_names]

    video_paths = {camera: os.path.join(args.root, camera + ".mp4") for camera in camera_names}

    # initialize an environment with offscreen renderer
    env = make(
        args.environment,
        args.robots,
        has_renderer=False,
        ignore_done=True,
        use_camera_obs=True,
        use_object_obs=False,
        camera_names=camera_names,
        camera_heights=camera_heights,
        camera_widths=camera_widths,
        hard_reset=False,
    )

    obs = env.reset()
    ndim = env.action_dim

    # create one video writer per camera
    writers = {camera: imageio.get_writer(path, fps=20) for camera, path in video_paths.items()}

    for i in range(args.timesteps):

        # run a uniformly random agent
        action = 0.5 * np.random.randn(ndim)
        obs, reward, done, info = env.step(action)

        # dump a frame from every K frames
        if i % args.skip_frame == 0:
            for camera in camera_names:
                frame = obs[camera + "_image"]
                writers[camera].append_data(frame)
            print("Saving frame #{}".format(i))

        if done:
            break

    for writer in writers.values():
        writer.close()

    for camera, path in video_paths.items():
        print(f"Video saved to {path} ({camera})")
