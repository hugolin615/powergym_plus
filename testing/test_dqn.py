import sys, os
import numpy as np
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
# Now import your modules
from stable_baselines3 import DQN
import dss as opendss
from core_open_dss.env_register import make_env
from random_agent import parse_arguments, seeding
from custom_policy_and_fe.custom_feature_extractor import GNNFeatureExtractor
from custom_policy_and_fe.single_policy import CombinedGCN
from stable_baselines3.common.vec_env import DummyVecEnv
import time
import argparse


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
    parser.add_argument(
        '--test', type=str,
        choices=['single_load_profile','all_load_profile'], 
        default = 'single_load_profile',
        help = 'Select the type of test'
    )
    parser.add_argument(
        '--saved_training_file', type=str,
        required=True, 
        help = 'Required: The name of the training file that you saved and want to load for this testing'
    )
    args = parser.parse_args()
    return args

def model_path(filename):
    cwd = os.getcwd()
    parent_directory = os.path.abspath(os.path.join(cwd, "..", ".."))
    trained_model_path = os.path.join(parent_directory, "saved_models", filename)
    return trained_model_path


def test_agent(env, args, filename, print_step = True):
    # Convert the environment into vec so it is compatible with SB3
    env = DummyVecEnv([lambda: env])

    if args.policy == "GNN_FE":
        policy_kwargs = dict(policy=GNNFeatureExtractor)
    elif args.policy == "GNN_Policy":
        policy_kwargs = dict(policy=CombinedGCN)
    elif args.policy == "MLP":
        policy_kwargs = None
    
    model_path_ = model_path(filename)
    print(model_path_)


    saved_model = DQN.load(model_path_, custom_objects=policy_kwargs)

    test_model = DQN("MultiInputPolicy", env, policy_kwargs=saved_model.policy_kwargs, verbose=1)

    # Load the policy state_dict from model_A into model_B
    test_model.policy.load_state_dict(saved_model.policy.state_dict())
    test_model.policy.optimizer.load_state_dict(saved_model.policy.optimizer.state_dict())

    print("Policy weights transferred from model_A to model_B.")

    obs = env.reset()

    # If obs is a numpy array with a single string element, try to access the actual observation
    if isinstance(obs, np.ndarray) and obs.shape == (1,) and isinstance(obs[0], str):
        print("Attempting to access actual observation...")
        try:
            actual_obs = getattr(env.envs[0], obs[0])
            print("Actual observation:", actual_obs)
            obs = actual_obs
        except AttributeError:
            print(f"Unable to find attribute '{obs[0]}' in the environment")

    num_episodes = 24
    episode_rewards = np.zeros(num_episodes)
    total_reward = 0

    for episode in range(num_episodes):
        done = False
        action, _ = test_model.predict(obs)
        obs, reward, done, info = env.step(action)
        
        # If obs is a numpy array with a single string element, try to access the actual observation
        if isinstance(obs, np.ndarray) and obs.shape == (1,) and isinstance(obs[0], str):
            try:
                actual_obs = getattr(env.envs[0], obs[0])
                obs = actual_obs
            except AttributeError:
                print(f"Unable to find attribute '{obs[0]}' in the environment")
                break
        
        total_reward += reward[0]

        if print_step:
            print(f"Step - Obs type: {type(obs)}, Action: {action}, Reward: {reward}, Done: {done}")

        episode_rewards[episode] = reward[0]
        print(f"Episode {episode + 1}: Total Reward: {total_reward}")
        # obs = env.reset()

    print(f"Total reward for load profile : {args.load_profile_id} : Total Reward: {np.sum(episode_rewards)}")

    # del model
    env.close()
    return episode_rewards

if __name__ == "__main__":
    print('DSS-python version:', opendss.__version__)
    start = time.time()
    args = parse_arguments()
    saved_model_name = args.saved_training_file
    #saved_model_name = '34_dqn_fe_graph_p_mlp_300000.zip' #change this name according to the saved file name that you want to use for testing
    num_episode = 24
    if args.test == "single_load_profile":
        env = make_env(args.env_name, profile_id=args.load_profile_id, action_type=args.action_type, obs_type=args.obs_type, worker_idx = args.worker_idx )
        episode_reward = test_agent(env, args, saved_model_name)
        data = episode_reward
    elif args.test == "all_load_profile":
        if args.env_name == "13Bus":
            load_profile = 72
        elif args.env_name == "34Bus":
            load_profile = 15
        elif args.env_name == "123Bus":
            load_profile = 15
        data = np.zeros((load_profile,num_episode))    
        for i in range(0, load_profile):
            env = make_env(args.env_name, profile_id=args.load_profile_id, action_type=args.action_type, obs_type=args.obs_type, worker_idx = args.worker_idx )
            episode_reward = test_agent(env, args, saved_model_name)
            data[i] = episode_reward
    end_time = time.time()
    file_name = f"reward_e_{args.env_name}"
    cwd = os.getcwd()
    print(cwd)
    parent_directory = os.path.abspath(os.path.join(cwd, "..", ".."))
    print(parent_directory)
    save_reward = os.path.join(parent_directory, "reward_data", file_name)
    np.save(save_reward, data)




















