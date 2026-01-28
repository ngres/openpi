from openpi_client.runtime import environment as _environment
from lerobot.robots import Robot


class LeRobotEnvironment(_environment.Environment):
    """Adapter for a LeRobot robot."""

    def __init__(
        self,
        robot: Robot,
    ) -> None:
        self._robot = robot
        self.observation_features = robot.observation_features
        self._action_features = robot.action_features

    def reset(self) -> None:
        """Reset the environment to its initial state.

        This will be called once before starting each episode.
        """
        return

    def is_episode_complete(self) -> bool:
        """Allow the environment to signal that the episode is complete.

        This will be called after each step. It should return `True` if the episode is
        complete (either successfully or unsuccessfully), and `False` otherwise.
        """
        return False

    def get_observation(self) -> dict:
        """Query the environment for the current state."""
        obs_dict = self._robot.get_observation()
        obs = {
            "state": [],
            "images": {},
        }
        for feature in self.observation_features:
            if isinstance(obs_dict[feature], tuple):
                obs["images"][feature] = obs_dict[feature]
            else:
                obs["state"].append(obs_dict[feature])
        assert len(obs["images"]) >= 1  # at least one image needs to be included in the robot observation
        return obs

    def apply_action(self, action: dict) -> None:
        """Take an action in the environment."""
        action_tensor = action["actions"]
        action_dict = {}
        if len(action_tensor) != len(self._action_features):
            raise ValueError(
                f"Action tensor has different length ({len(action_tensor)}) than action features ({len(self._action_features)})."
            )
        for idx, feature in enumerate(self._action_features):
            action_dict[feature] = action_tensor[idx]
        self._robot.send_action(action_dict)
