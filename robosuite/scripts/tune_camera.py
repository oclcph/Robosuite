"""
Convenience script to tune a camera view in a mujoco environment.
Allows keyboard presses to move a camera around in the viewer, and
then prints the final position and quaternion you should set
for your camera in the mujoco XML file.
"""

import argparse
import time
import xml.etree.ElementTree as ET
from collections import deque
from threading import Lock

import numpy as np
from pynput.keyboard import Controller, Key, Listener

import robosuite
import robosuite.utils.transform_utils as T
from robosuite.utils.camera_utils import CameraMover
from robosuite.utils.mjcf_utils import find_elements, find_parent

# some settings
DELTA_POS_KEY_PRESS = 0.05  # delta camera position per key press
DELTA_ROT_KEY_PRESS = 1  # delta camera angle per key press


class KeyboardHandler:
    def __init__(self, camera_mover):
        """
        Store internal state here.

        Args:
            camera_mover (CameraMover): Playback camera class
        cam_body_id (int): id corresponding to parent body of camera element
        """
        self.camera_mover = camera_mover
        self.running = True
        self._cmd_queue = deque(maxlen=256)
        self._queue_lock = Lock()

        # make a thread to listen to keyboard and register our callback functions
        self.listener = Listener(on_press=self.on_press, on_release=self.on_release)

        # start listening
        self.listener.start()

    def on_press(self, key):
        """
        Key handler for key presses.

        Args:
            key (int): keycode corresponding to the key that was pressed
        """

        try:
            if key == Key.esc:
                self.running = False
                return False

            # controls for moving rotation
            if key == Key.up:
                # rotate up
                self._push_command(("rotate", [1.0, 0.0, 0.0], DELTA_ROT_KEY_PRESS))
            elif key == Key.down:
                # rotate down
                self._push_command(("rotate", [-1.0, 0.0, 0.0], DELTA_ROT_KEY_PRESS))
            elif key == Key.left:
                # rotate left
                self._push_command(("rotate", [0.0, 1.0, 0.0], DELTA_ROT_KEY_PRESS))
            elif key == Key.right:
                # rotate right
                self._push_command(("rotate", [0.0, -1.0, 0.0], DELTA_ROT_KEY_PRESS))

            # controls for moving position
            elif key.char == "w":
                # move forward
                self._push_command(("move", [0.0, 0.0, -1.0], DELTA_POS_KEY_PRESS))
            elif key.char == "s":
                # move backward
                self._push_command(("move", [0.0, 0.0, 1.0], DELTA_POS_KEY_PRESS))
            elif key.char == "a":
                # move left
                self._push_command(("move", [-1.0, 0.0, 0.0], DELTA_POS_KEY_PRESS))
            elif key.char == "d":
                # move right
                self._push_command(("move", [1.0, 0.0, 0.0], DELTA_POS_KEY_PRESS))
            elif key.char == "r":
                # move up
                self._push_command(("move", [0.0, 1.0, 0.0], DELTA_POS_KEY_PRESS))
            elif key.char == "f":
                # move down
                self._push_command(("move", [0.0, -1.0, 0.0], DELTA_POS_KEY_PRESS))
            elif key.char == ".":
                # rotate counterclockwise
                self._push_command(("rotate", [0.0, 0.0, 1.0], DELTA_ROT_KEY_PRESS))
            elif key.char == "/":
                # rotate clockwise
                self._push_command(("rotate", [0.0, 0.0, -1.0], DELTA_ROT_KEY_PRESS))
            elif key.char == "q":
                self.running = False
                return False

        except AttributeError as e:
            pass

    def on_release(self, key):
        """
        Key handler for key releases.

        Args:
            key: [NOT USED]
        """
        pass

    def stop(self):
        """Stops keyboard listener thread cleanly."""
        self.running = False
        if self.listener is not None:
            self.listener.stop()
            self.listener.join()

    def _push_command(self, cmd):
        with self._queue_lock:
            self._cmd_queue.append(cmd)

    def drain_commands(self):
        """Returns a snapshot of queued camera commands and clears the queue."""
        with self._queue_lock:
            cmds = list(self._cmd_queue)
            self._cmd_queue.clear()
        return cmds


def print_command(char, info):
    """
    Prints out the command + relevant info entered by user

    Args:
        char (str): Command entered
        info (str): Any additional info to print
    """
    char += " " * (10 - len(char))
    print("{}\t{}".format(char, info))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", type=str, default="Lift")
    parser.add_argument("--robots", nargs="+", type=str, default=["Sawyer"], help="Which robot(s) to use in the env")
    parser.add_argument(
        "--camera",
        type=str,
        default=None,
        help="Camera name or full <camera ... /> XML tag to tune. If omitted, the script asks interactively.",
    )
    parser.add_argument(
        "--table-full-size",
        nargs=3,
        type=float,
        default=None,
        metavar=("X", "Y", "Z"),
        help="Table full size passed to the environment, e.g. --table-full-size 1.2 1.2 0.05",
    )
    parser.add_argument("--num-cubes", type=int, default=None, help="Number of cubes for MultiArmBlockLift.")
    parser.add_argument("--cube-size", type=float, default=None, help="Half-size of each cube for MultiArmBlockLift.")
    parser.add_argument(
        "--cube-spawn-range",
        nargs=2,
        type=float,
        default=None,
        metavar=("X", "Y"),
        help="Uniform spawn range around table center for MultiArmBlockLift cubes.",
    )
    parser.add_argument(
        "--arm-positions",
        nargs="+",
        type=str,
        default=None,
        choices=["west", "south", "east", "north"],
        help="Explicit robot sides for MultiArmBlockLift, in robot order.",
    )
    args = parser.parse_args()

    print("\nWelcome to the camera tuning script! You will be able to tune a camera view")
    print("by moving it around using your keyboard. The controls are printed below.")

    print("")
    print_command("Keys", "Command")
    print_command("w-s", "zoom the camera in/out")
    print_command("a-d", "pan the camera left/right")
    print_command("r-f", "pan the camera up/down")
    print_command("arrow keys", "rotate the camera to change view direction")
    print_command(".-/", "rotate the camera view without changing view direction")
    print_command("q / esc", "quit and close safely")
    print("")

    # read camera XML tag from CLI or user input
    inp = args.camera
    if inp is None:
        inp = input(
            "\nPlease paste a camera name below \n"
            "OR xml tag below (e.g. <camera ... />) \n"
            "OR leave blank for an example:\n"
        )

    if len(inp) == 0:
        if args.env != "Lift":
            raise Exception("ERROR: env must be Lift to run default example.")
        print("\nUsing an example tag corresponding to the frontview camera.")
        print("This xml tag was copied from robosuite/models/assets/arenas/table_arena.xml")
        inp = '<camera mode="fixed" name="frontview" pos="1.6 0 1.45" quat="0.56 0.43 0.43 0.56"/>'

    # remember the tag and infer some properties
    from_tag = "<" in inp
    notify_str = (
        "NOTE: using the following xml tag:\n"
        if from_tag
        else "NOTE: using the following camera (initialized at default sim location)\n"
    )

    print(notify_str)
    print("{}\n".format(inp))

    cam_tree = ET.fromstring(inp) if from_tag else ET.Element("camera", attrib={"name": inp})
    CAMERA_NAME = cam_tree.get("name")

    env_kwargs = {}
    if args.table_full_size is not None:
        env_kwargs["table_full_size"] = tuple(args.table_full_size)
    if args.num_cubes is not None:
        env_kwargs["num_cubes"] = args.num_cubes
    if args.cube_size is not None:
        env_kwargs["cube_size"] = args.cube_size
    if args.cube_spawn_range is not None:
        env_kwargs["cube_spawn_range"] = tuple(args.cube_spawn_range)
    if args.arm_positions is not None:
        env_kwargs["arm_positions"] = args.arm_positions

    # make the environment
    env = robosuite.make(
        args.env,
        robots=args.robots,
        has_renderer=True,
        has_offscreen_renderer=False,
        ignore_done=True,
        use_camera_obs=False,
        control_freq=100,
        **env_kwargs,
    )
    env.reset()

    # Create the camera mover
    camera_mover = CameraMover(
        env=env,
        camera=CAMERA_NAME,
    )

    # Make sure we're using the camera that we're modifying
    camera_id = env.sim.model.camera_name2id(CAMERA_NAME)
    env.viewer.set_camera(camera_id=camera_id)

    # Infer initial camera pose
    if from_tag:
        initial_file_camera_pos = np.array(cam_tree.get("pos").split(" ")).astype(float)
        initial_file_camera_quat = T.convert_quat(np.array(cam_tree.get("quat").split(" ")).astype(float), to="xyzw")
        # Set these values as well
        camera_mover.set_camera_pose(pos=initial_file_camera_pos, quat=initial_file_camera_quat)
        # Optionally set fov if specified
        cam_fov = cam_tree.get("fovy", None)
        if cam_fov is not None:
            env.sim.model.cam_fovy[camera_id] = float(cam_fov)
    else:
        initial_file_camera_pos, initial_file_camera_quat = camera_mover.get_camera_pose()
    # Define initial file camera pose
    initial_file_camera_pose = T.make_pose(initial_file_camera_pos, T.quat2mat(initial_file_camera_quat))

    # remember difference between camera pose in initial tag and absolute camera pose in world
    initial_world_camera_pos, initial_world_camera_quat = camera_mover.get_camera_pose()
    initial_world_camera_pose = T.make_pose(initial_world_camera_pos, T.quat2mat(initial_world_camera_quat))
    world_in_file = initial_file_camera_pose.dot(T.pose_inv(initial_world_camera_pose))

    # register callbacks to handle key presses in the viewer
    key_handler = KeyboardHandler(camera_mover=camera_mover)

    # just spin to let user interact with window
    spin_count = 0
    try:
        while key_handler.running:
            # Apply queued key commands from listener thread in the main sim thread.
            for cmd_type, vec, scale in key_handler.drain_commands():
                if cmd_type == "move":
                    camera_mover.move_camera(direction=vec, scale=scale)
                elif cmd_type == "rotate":
                    camera_mover.rotate_camera(point=None, axis=vec, angle=scale)

            action = np.zeros(env.action_dim)
            obs, reward, done, _ = env.step(action)

            # If viewer window closes externally, render() may fail in backend bindings.
            try:
                env.render()
            except Exception:
                break

            spin_count += 1
            if spin_count % 500 == 0:
                # convert from world coordinates to file coordinates (xml subtree)
                camera_pos, camera_quat = camera_mover.get_camera_pose()
                world_camera_pose = T.make_pose(camera_pos, T.quat2mat(camera_quat))
                file_camera_pose = world_in_file.dot(world_camera_pose)
                # TODO: Figure out why numba causes black screen of death (specifically, during mat2pose --> mat2quat call below)
                camera_pos, camera_quat = T.mat2pose(file_camera_pose)
                camera_quat = T.convert_quat(camera_quat, to="wxyz")

                print("\n\ncurrent camera tag you should copy")
                cam_tree.set("pos", "{} {} {}".format(camera_pos[0], camera_pos[1], camera_pos[2]))
                cam_tree.set("quat", "{} {} {} {}".format(camera_quat[0], camera_quat[1], camera_quat[2], camera_quat[3]))
                print(ET.tostring(cam_tree, encoding="utf8").decode("utf8"))
    except KeyboardInterrupt:
        pass
    finally:
        key_handler.stop()
        env.close()
