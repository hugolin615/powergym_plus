import gymnasium as gym
import numpy as np
import os
import random
import argparse
import sys
from datetime import datetime
import yaml

from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

from powergym_plus.env.env_register import make_env
from powergym_plus.util.util import get_latest_run_id
from powergym_plus.util.util import ALGOS, POLICY
from stable_baselines3.common.callbacks import BaseCallback
from powergym_plus.env.env import Wrap_Obs


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--algo", help="RL Algorithm", default="ppo", type=str, required=False)
    parser.add_argument("--policy", help="Policy used by algorith", default="custom", type=str, required=False)
    parser.add_argument('--env_name', default='13Bus', type = str)
    parser.add_argument('--total_steps', default = 24000, type = int, help = 'maximum number of training steps')
    parser.add_argument('--seed', default = 12345, type = int, help = 'random seed')
    parser.add_argument('--eval_freq', default = 5, type = int, help = 'Evaluate the agent every n episodes (NOT steps) if negative, no evaluation') # at the current implementation, all evaluation load profiles will be evaluated
    parser.add_argument("-f", "--log-folder", help="Log folder", type=str, default="/home/hugo/experiment/AI_Power/powergym_plus/powergym_plus/logs")
    parser.add_argument("--device", help="PyTorch device to be use (ex: cpu, cuda...)", default="auto", type=str)
    # the following is the arguments for PPO algorithms passed directly to SB3 PPO policy API
    parser.add_argument('--ppo_batch_size', default = 24, type = int, help = 'Batch size used for PPO policy')
    parser.add_argument('--wrap_obs', default = 'flat', type = str, help = 'Let user decide how to wrap observations')
    parser.add_argument('--no_observe_load', action = 'store_true', help = 'dont use load profiles in observation')
    args = parser.parse_args()
    return args

def train(args):

    random.seed(args.seed)
    set_random_seed(args.seed)

    user_info = {}
    if args.no_observe_load:
        user_info = {'observe_load': False}
    else:
        user_info = {'observe_load': True}
  
    print(f'train: {args.wrap_obs}')
    if args.wrap_obs == 'flat':
        user_info['wrap_obs'] = Wrap_Obs.FLAT
    elif args.wrap_obs == 'no wraps':
        user_info['wrap_obs'] = Wrap_Obs.NO_WRAP
    elif args.wrap_obs == 'graph':
        user_info['wrap_obs'] = Wrap_Obs.GRAPH
    else:
        raise NotImplementedError("Subclasses must implement this method")

    print(f'train: {user_info}')
    env = make_env(args.env_name, user_info = user_info)

    num_load_profiles = env.get_num_profiles()
    ep_len = env.get_ep_len()
    train_load_profiles = range(int(num_load_profiles / 2)) 
    eval_load_profiles = range(int(num_load_profiles / 2) + 1, num_load_profiles)
   
    vec_env = DummyVecEnv([lambda: env])

    assert args.algo in ALGOS
    assert args.policy in POLICY

    # Initialize PPO agent

    print(f'train {ALGOS[args.algo]}, {POLICY[args.policy]}')
    model = ALGOS[args.algo](
        policy = POLICY[args.policy],
        env = vec_env,
        n_steps = 5 * ep_len,
        batch_size = ep_len,
        verbose = 1
    )

    # Train the agent

    print(f'current circuit has {num_load_profiles} load profiles and episode length of {ep_len}')
    print(f'training profiles: {train_load_profiles}; testing profiles: {eval_load_profiles}')

    obs, info = env.reset()
    action = env.random_action()
    action, reward, done, truncated, info = env.step(action)
    print(f'obs: {obs}, action: {action}, reward: {reward}')

    # teesting on graphs
    print(env.topology, env.topology.nodes, env.topology.edges)

    
if __name__ == "__main__":
    args = parse_arguments()
    print(f"{args}")
    train(args)


