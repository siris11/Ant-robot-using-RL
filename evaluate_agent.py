import gym
import pybullet, pybullet_envs
import torch as th

from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy

# Create environment
env = gym.make('AntBulletEnv-v0')
env.render(mode="human")

# Set the maximum episode length (increase as needed)
max_episode_steps = 2000
env._max_episode_steps = max_episode_steps

policy_kwargs = dict(activation_fn=th.nn.LeakyReLU, net_arch=[512, 512])
# Instantiate the agent
model = PPO('MlpPolicy', env, learning_rate=0.0003, policy_kwargs=policy_kwargs, verbose=1, device='cpu')

del model  # delete trained model to demonstrate loading
# Load the trained agent
model = PPO.load("ppo_Ant")

# Evaluate the agent
obs = env.reset()
for i in range(100):
    dones = False
    game_score = 0
    steps = 0
    while not dones:
        action, _states = model.predict(obs, deterministic=True)
        obs, rewards, dones, info = env.step(action)
        game_score += rewards
        steps += 1
        env.render()
    print("game", i, "steps", steps, "game score %.3f" % game_score)
    obs = env.reset()
