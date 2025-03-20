# Copyright 2021 Siemens Corporation
# SPDX-License-Identifier: MIT

"""Random agent to probe enviroment
"""
import numpy as np
import dss as opendss
import glob
from core_open_dss.env_register import make_env
from env.agent_env import ActionSpace
import argparse
import random
import itertools
import sys, os
import multiprocessing as mp
from stable_baselines3 import PPO
import matplotlib.pyplot as plt

def parse_arguments():
    parser = argparse.ArgumentParser(description='Argument Parser')
    parser.add_argument('--env_name', default='13Bus')
    parser.add_argument('--seed', type=int, default=123456, metavar='N',
                         help='random seed')
    parser.add_argument('--num_steps', type=int, default=1000, metavar='N',
                         help='maximum number of steps')
    parser.add_argument('--num_workers', type=int, default=3, metavar='N', 
                         help='number of parallel processes')
    parser.add_argument('--use_plot', type=lambda x: str(x).lower()=='true', default=False)
    parser.add_argument('--do_testing', type=lambda x: str(x).lower()=='true', default=False)
    parser.add_argument('--mode', type=str, default='single',
                        help="running mode, random, parallele, episodic or dss")
    args = parser.parse_args()
    return args

def seeding(seed):
    np.random.seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)


def run_random_agent(args, load_profile_idx=0, worker_idx=None, use_plot=False, print_step=False):
    """Run a agent that takes arbitrary control actions
    Args:
        test_action (str): Test env with fixed or random actions 
        steps (int, optional): Number of steps to run the env
        use_plot (bool, optional): Plot outcome of random agent
    """
    
    cwd = os.getcwd()
    
    # get environment
    env= make_env(args.env_name, profile_id= load_profile_idx, action_type= "Reduced", obs_type="graph", worker_idx=worker_idx)
    #env.seed(args.seed + 0 if worker_idx is None else worker_idx) #123456
    print(env)

    if print_step:
        print('This system has {} capacitors, {} regulators and {} batteries'.format(env.cap_num, env.reg_num, env.bat_num))
        print('reg, bat action nums: ', env.reg_act_num, env.bat_act_num)
        print('-'*80)

    
    obs = env.reset()
    reward = []

    episode_reward = 0.0
    episode_length = 24

    #for j in range(epsidoe_len):
    for i in range(episode_length):
        print(f"for i: {i} {'-'*80}")
        dis_action = env.random_action()
        obs, r, done, trunacted, info = env.step(dis_action)
        print("action:", dis_action, "obs: ", obs, "rewards", r)
        episode_reward += r
        reward.append(r)
        print(f'load_profile: {load_profile_idx}, episode_reward: {r}')
        break
    print(reward)

if __name__ == '__main__':
    print('DSS-python version:', opendss.__version__)
    args = parse_arguments()
    #print(args)
    seeding(args.seed)
    if args.mode.lower() ==  'single':
        run_random_agent(args, worker_idx=None, use_plot=args.use_plot, print_step=True)
    else:
        raise NotImplementedError("Running mode not implemented")
    
