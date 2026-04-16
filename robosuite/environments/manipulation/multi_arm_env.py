import numpy as np

from robosuite.environments.manipulation.manipulation_env import ManipulationEnv
from robosuite.utils.robot_utils import check_bimanual


class MultiArmEnv(ManipulationEnv):
    """
    A manipulation environment intended for 1 to 4 single-arm robots.

    This class only handles multi-arm configuration checks and common arm
    placement resolution. Task-specific logic should be implemented in subclasses.
    """

    _VALID_SIDES = ("west", "south", "east", "north")
    _SIDE_TO_YAW = {
        "west": 0.0,
        "south": np.pi / 2,
        "east": np.pi,
        "north": -np.pi / 2,
    }

    def _check_robot_configuration(self, robots):
        """
        Sanity check to make sure the inputted robots and configuration are acceptable.

        Args:
            robots (str or list of str): Robots to instantiate within this env
        """
        super()._check_robot_configuration(robots)

        robots = robots if isinstance(robots, (list, tuple)) else [robots]
        if not 1 <= len(robots) <= 4:
            raise ValueError("MultiArmEnv supports 1 to 4 robots.")

        for robot in robots:
            if check_bimanual(robot):
                raise ValueError("MultiArmEnv currently supports only single-arm robots.")

        valid_configs = {"default", "clockwise", "custom"}
        if len(robots) == 2:
            valid_configs.update({"adjacent", "opposed"})

        if self.env_configuration not in valid_configs:
            raise ValueError(
                f"Invalid env_configuration '{self.env_configuration}'. "
                f"Supported values for {len(robots)} robot(s): {sorted(valid_configs)}"
            )

    def _resolve_arm_positions(self, arm_positions=None):
        """
        Resolves clockwise side assignments for all robots.

        Args:
            arm_positions (None or list of str): Explicit side names for each robot.
                If None, defaults are derived from env_configuration.

        Returns:
            list[str]: Side assignment for each robot in robot order.
        """
        requested_positions = arm_positions
        if requested_positions is None:
            requested_positions = getattr(self, "arm_positions", None)

        if requested_positions is not None:
            if len(requested_positions) != self.num_robots:
                raise ValueError(
                    f"arm_positions length ({len(requested_positions)}) must match number of robots ({self.num_robots})."
                )
            positions = [p.lower() for p in requested_positions]
        else:
            if self.num_robots == 1:
                positions = ["west"]
            elif self.num_robots == 2:
                config = "adjacent" if self.env_configuration in ("default", "clockwise") else self.env_configuration
                if config == "adjacent":
                    positions = ["west", "south"]
                elif config == "opposed":
                    positions = ["west", "east"]
                else:
                    raise ValueError(
                        "For two robots, env_configuration must be 'adjacent' or 'opposed' unless arm_positions is set."
                    )
            elif self.num_robots == 3:
                positions = ["west", "south", "east"]
            else:
                positions = ["west", "south", "east", "north"]

        invalid = [p for p in positions if p not in self._VALID_SIDES]
        if invalid:
            raise ValueError(f"Invalid arm side(s): {invalid}. Valid values: {self._VALID_SIDES}")

        if len(set(positions)) != len(positions):
            raise ValueError(f"arm_positions contains duplicates: {positions}")

        return positions
