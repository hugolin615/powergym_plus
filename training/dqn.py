import sys, os
from pathlib import Path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from custom_policy_and_fe.single_policy import CombinedGCN
from custom_policy_and_fe.custom_feature_extractor import GNNFeatureExtractor
from stable_baselines3 import DQN
from stable_baselines3.common.utils import get_schedule_fn
import dss as opendss
import argparse
from core_open_dss.env_register import make_env
import time


def parse_arguments():
    parser = argparse.ArgumentParser(description='Argument Parser')
    parser.add_argument(
        '--policy', type=str,
        choices=['MLP', 'GNN_FE', 'GNN_Policy'], 
        default='MLP', 
        help='Choose a policy: MLP, GNN_FE, or GNN_Policy'
    )
    parser.add_argument('--env_name', type=str,
        choices=['13Bus', '34Bus', '123Bus', '8500-Node'], 
        default='13Bus', 
        help='Choose a architecture: 13Bus, 34Bus, 123Bus, 8500-Node'
    )
    parser.add_argument(
        '--total_training_step', type=int,  
        default=100000, 
        help='Choose a training step'
    )
    parser.add_argument(
        '--buffer_size', type=int,  
        default=10000, 
        help='Choose a buffer size'
    )
    parser.add_argument(
        '--learning_starts', type=int, 
        default=1000, 
        help='Choose a buffer size'
    )
    parser.add_argument(
        '--action_type', type=str,
        choices=['Reduced','Original'],
        default='Reduced',
        help='Chose an action type'
    )
    parser.add_argument(
        '--obs_type', type=str,
        choices=['graph','non_graph'],
        default='graph',
        help='For GNN use graph, for other cases use non_graph'
    )
    parser.add_argument(
        '--load_profile_id', type=int,
        default= 0,
        help='Choose a load profile. There are 72 load profile for 13 bus, 15 load profile for 34 bus'
    )
    parser.add_argument(
        '--seed', type=int,
        default=123456, 
        help='random seed'
    )
    parser.add_argument(
        '--worker_idx', type=int,
        default=None,
        help='Set worker_id'
    )
    args = parser.parse_args()
    return args

def run_agent(env, args):
    if args.policy == "GNN_FE":
        policy_kwargs = dict(features_extractor_class=GNNFeatureExtractor, features_extractor_kwargs=dict(embed_dim=64),)
    elif args.policy == "GNN_Policy":
        policy_kwargs = dict(features_extractor_class=CombinedGCN, features_extractor_kwargs=dict(num_actions=env.action_space.n, embed_dim=64))
    elif args.policy == "MLP":
        policy_kwargs = None
    
    name = f"{args.env_name}_{args.policy}_ts_{args.total_training_step}_bs_{args.buffer_size}_ls_{args.learning_starts}"

    cwd = os.getcwd()
    print(cwd)
    parent_directory = os.path.abspath(os.path.join(cwd, "..", ".."))
    print(parent_directory)
    tensorboard_log_dir = os.path.join(parent_directory, "data", "dqn_tensorboard")
    saved_model_dir  = os.path.join(parent_directory, "saved_models", name)
    print(tensorboard_log_dir)
    print(saved_model_dir)

    # Ensure directories exist
    os.makedirs(tensorboard_log_dir, exist_ok=True)
    os.makedirs(os.path.dirname(saved_model_dir), exist_ok=True)

    if args.obs_type == "graph":
        policy = "MultiInputPolicy"
    else:
        policy = "MlpPolicy"

    #initialize the environment
    model = DQN(policy = policy, env = env, policy_kwargs= policy_kwargs, device = "cpu" ,verbose=1, buffer_size= args.buffer_size, learning_starts= args.learning_starts, 
                tensorboard_log = tensorboard_log_dir)
    
    #train the model
    model.learn(total_timesteps=args.total_training_step, tb_log_name = name)
    
    # Save the trained model
    model.save(saved_model_dir)
    
if __name__=="__main__":
    print('DSS-python version:', opendss.__version__)
    start_time = time.time()
    args = parse_arguments()
    env = make_env(args.env_name, profile_id=args.load_profile_id, action_type=args.action_type, obs_type=args.obs_type, worker_idx = args.worker_idx )
    agent = run_agent(env, args)
    end_time = time.time()
    print(f"Total time for simulation: {end_time - start_time}")

