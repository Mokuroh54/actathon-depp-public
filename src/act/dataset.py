"""LeRobot dataset helpers."""
from lerobot.utils.feature_utils import hw_to_dataset_features


def make_features(robot):
    return {
        **hw_to_dataset_features(robot.action_features, "action"),
        **hw_to_dataset_features(robot.observation_features, "observation"),
    }
