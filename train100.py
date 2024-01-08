import gym
import pybullet as p
import pybullet_envs
import pybullet_data
import torch as th
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy

class CustomAntBulletEnv(gym.Env):
    def __init__(self):
        self.robot = None
        self.obstacle = None
        # ... (other initialization)
        self.connect_to_physics_server()  # Establish connection to the physics server
        self.load_obstacle()  # Load the obstacle

    def connect_to_physics_server(self):
        p.connect(p.GUI)  # Connect to the physics server using the graphical interface
        p.setRealTimeSimulation(5)

    def load_obstacle(self):
        # Load the obstacle from the URDF file
        self.obstacle = p.loadURDF('obstacles.urdf')  # Adjust position as needed


    def step(self, action):
        # Perform action in the environment
        # Get observations, rewards, etc.

        # Check for collision between robot and obstacle
        contacts = p.getContactPoints(self.robot, self.obstacle)
        collision_detected = len(contacts) > 0

        if collision_detected:
            # Penalize the agent for hitting the obstacle
            penalty = -1  # You can adjust the penalty value
            reward += penalty
            done = True  # You might want to end the episode on collision

        # Return observations, rewards, etc.
        return observations, reward, done, info

    

def train_agent(env_name, max_iterations=8000, target_mean_reward=2000):
    # Create the custom environment
    env = CustomAntBulletEnv()

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

    # Keep the simulation running for additional observation
    input("Press Enter to continue...")
    del model  # Delete trained model to demonstrate loading

if __name__ == "__main__":
    train_agent('AntBulletEnv-v0')
