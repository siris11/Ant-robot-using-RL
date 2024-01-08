import gym
import pybullet_envs, pybullet
import torch as th
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy

def train_agent(env_name, max_iterations=8000, target_mean_reward=2000):
    # Create environment
    env = gym.make(env_name)
    env.render(mode="human")

    # Define policy architecture
    policy_kwargs = dict(activation_fn=th.nn.LeakyReLU, net_arch=[512, 512])

    # Instantiate the agent
    model = PPO('MlpPolicy', env, learning_rate=0.0003, policy_kwargs=policy_kwargs, verbose=1, device='cpu')

    # Train the agent
    for i in range(max_iterations):
        print("Training iteration", i)
        model.learn(total_timesteps=10000)
        model.save("ppo_Ant2")

        mean_reward, std_reward = evaluate_policy(model, model.get_env(), n_eval_episodes=10)
        print("Mean reward:", mean_reward)

        if mean_reward >= target_mean_reward:
            print("***Agent trained with average reward", mean_reward)
            break

    del model  # Delete trained model to demonstrate loading

if __name__ == "__main__":
    train_agent('AntBulletEnv-v0')
