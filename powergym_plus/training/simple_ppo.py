import gymnasium as gym
import numpy as np
import os
import random

from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

from powergym_plus.env.env_register import make_env

env_name = '13Bus'
seed_val = 1

random.seed(seed_val)
set_random_seed(seed_val)

def evaluate(model, num_episodes=100, eval_range = range(0,1), deterministic=True):
    """
    Evaluate a RL agent
    :param model: (BaseRLModel object) the RL Agent
    :param num_episodes: (int) number of episodes to evaluate it
    :return: (float) Mean reward for the last num_episodes
    """
    # This function will only work for a single Environment
    vec_env = model.get_env()
    all_episode_rewards = []

    for i in range(num_episodes):
        print(f'-------- episode: {i} -----------')
        episode_rewards = []
        done = False
        #obs = vec_env.reset(load_profile_idx = eval_profile_idx)
        eval_profile_idx = eval_range.start + i % len(eval_range)
        obs, info = vec_env.envs[0].reset(load_profile_idx = eval_profile_idx)
        while not done:
            # _states are only useful when using LSTM policies
            action, _states = model.predict(obs, deterministic=deterministic)
            # here, action, rewards and dones are arrays
            # because we are using vectorized env
            # also note that the step only returns a 4-tuple, as the env that is returned
            # by model.get_env() is an sb3 vecenv that wraps the >v0.26 API
            # obs, reward, done, info = vec_env.step(action)
            obs, reward, done, truncated, info = vec_env.envs[0].step(action)
            # print(f'action: {action}, obs: {obs}, reward: {reward}')
            episode_rewards.append(reward)
        #print(f'episode_rewards: {episode_rewards}, {sum(episode_rewards)}')
        all_episode_rewards.append(sum(episode_rewards))

    mean_episode_reward = np.mean(all_episode_rewards)
    std_reward = np.std(all_episode_rewards)
    print("Mean reward:", mean_episode_reward, "standard deviation:", std_reward ,"Num episodes:", num_episodes)

    return mean_episode_reward, std_reward

class ResetEachEpisodeAndEvalCallback(BaseCallback):
    def __init__(self, 
                 eval_every=5,
                 eval_episodes=3,
                 save_path="./best_model",
                 log_path="./eval_log.txt",
                 train_range = range(0, 1), # range of load profile idx for training
                 eval_range = range(0, 1), # range of load profile idx for evaluation
                 verbose=1):
        super().__init__(verbose)
        self.eval_every = eval_every
        self.eval_episodes = eval_episodes
        self.save_path = save_path
        self.log_path = log_path
        self.episode_count = 0
        self.best_mean_reward = -np.inf
        self.train_range = train_range
        self.eval_range = eval_range

        os.makedirs(self.save_path, exist_ok=True)

        # Open log file (append mode)
        self.log_file = open(self.log_path, "a")

    def _on_step(self) -> bool:
        done_array = self.locals["dones"]

        # print(f"HL DEBUG: {type(self.training_env)}")

        for done in done_array:
            if done:
                self.episode_count += 1

               
                # 📊 Evaluation every N episodes
                
                if self.episode_count % self.eval_every == 0:
                    mean_reward = self._evaluate_policy()
                    
                    # 📝 Log to file
                    self.log_file.write(f"Episode {self.episode_count}, Avg Reward: {mean_reward:.2f}\n")
                    self.log_file.flush()

                    if self.verbose:
                        print(f"📊 Eval at episode {self.episode_count}: Avg reward = {mean_reward:.2f}")

                    # 💾 Save best model
                    if mean_reward > self.best_mean_reward:
                        self.best_mean_reward = mean_reward
                        save_file = os.path.join(self.save_path, "best_model")
                        self.model.save(save_file)
                        if self.verbose:
                            print(f"💾 Best model saved at episode {self.episode_count} with reward {mean_reward:.2f}")
                # 🔁 Always reset environment after each episode
                #if self.verbose:
                #    print(f"\n🔄 Manual reset after episode {self.episode_count}")
                training_profile_idx = random.choice(self.train_range)
                # self.training_env.reset(seed = None, load_profile_idx = training_profile_idx)
                self.training_env.envs[0].reset(load_profile_idx = training_profile_idx) # only work for single env


        return True

    def _evaluate_policy(self):
        rewards = []
        #eval_env = self.training_env.envs[0]  # first sub-env
        eval_env = self.training_env # first sub-env

        # eval_profile_idx = self.eval_range.start

        for i in range(self.eval_episodes):
            eval_profile_idx = self.eval_range.start + i % len(self.eval_range)
            obs, info = eval_env.envs[0].reset(load_profile_idx = eval_profile_idx)
            done = False
            total_reward = 0.0

            while not done:
                action, _ = self.model.predict(obs, deterministic=True)
                # obs, reward, done, info = eval_env.step(action)
                obs, reward, done, truncated, info = eval_env.envs[0].step(action)
                # print(f'action: {action}, obs: {obs}, reward: {reward}')
                total_reward += reward

            rewards.append(total_reward)
            # eval_profile_idx = (eval_profile_idx + 1) % len(self.eval_range) # it is possible that eval_episodes are longer than all evaluations profiles

        return np.mean(rewards)

    def _on_training_end(self):
        self.log_file.close()

env = make_env(env_name)

num_load_profiles = env.get_num_profiles()
ep_len = env.get_ep_len()
train_load_profiles = range(int(num_load_profiles / 2)) 
eval_load_profiles = range(int(num_load_profiles / 2) + 1, num_load_profiles)

total_steps = 1500 * ep_len

vec_env = DummyVecEnv([lambda: env])

# Initialize PPO agent
model = PPO(
    policy = "MlpPolicy",
    env = vec_env,
    n_steps = 5 * ep_len,
    batch_size = ep_len,
    verbose = 1
)

# Train the agent

print(f'current circuit has {num_load_profiles} load profiles and episode length of {ep_len}')
print(f'training profiles: {train_load_profiles}; testing profiles: {eval_load_profiles}')

callback = ResetEachEpisodeAndEvalCallback(
    eval_every = 5,
    eval_episodes = len(eval_load_profiles),
    save_path = "./checkpoints",
    log_path = "./eval_results.txt",
    train_range = train_load_profiles,
    eval_range = eval_load_profiles
)

model.learn(total_timesteps = total_steps, callback = callback)

mean_ep_reward, std_ep_reward = evaluate(model, num_episodes = len(eval_load_profiles), eval_range = eval_load_profiles)

# eval_env = make_env(env_name)

#mean_reward, std_reward = evaluate_policy(model, eval_env, n_eval_episodes=100)
#print(f'mean_reward: {mean_reward}, std_reward: {std_reward}')

