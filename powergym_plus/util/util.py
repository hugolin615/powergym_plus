import argparse
import os
import glob

from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.common.policies import BasePolicy, MultiInputActorCriticPolicy
from stable_baselines3 import A2C, DDPG, DQN, PPO, SAC, TD3
from stable_baselines3.ppo import MlpPolicy
from powergym_plus.util.sb3_policy import MyActorCriticPolicy
from powergym_plus.util.sb3_policy import ActorCriticGnnPolicy

ALGOS: dict[str, type[BaseAlgorithm]] = {
    "ppo": PPO,
}

POLICY: dict[str, type[BasePolicy]] = {
    "mlp": MlpPolicy,
    "custom": MyActorCriticPolicy,
    "multi": MultiInputActorCriticPolicy,
    "graph": ActorCriticGnnPolicy,
}

def get_latest_run_id(log_path: str, env_name: str) -> int:
    """
    Returns the latest run number for the given log name and log path,
    by finding the greatest number in the directories.

    :param log_path: path to log folder
    :param env_name:
    :return: latest run number
    """
    max_run_id = 0
    for path in glob.glob(os.path.join(log_path, env_name + "_[0-9]*")):
        run_id = path.split("_")[-1]
        path_without_run_id = path[: -len(run_id) - 1]
        if path_without_run_id.endswith(env_name) and run_id.isdigit() and int(run_id) > max_run_id:
            max_run_id = int(run_id)
    return max_run_id

