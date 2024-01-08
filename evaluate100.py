import gym
import pybullet as p
import pybullet_envs
import torch as th
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy

class CustomAntBulletEnv(gym.Env):
    def __init__(self):
        self.physics_client = p.connect(p.GUI)  # Connect to the physics server using the graphical interface
        self.env = gym.make('AntBulletEnv-v0')
        self.obstacle = None
        self.load_obstacle()

    def load_obstacle(self):
        # Load the obstacle from the URDF file
        self.obstacle = p.loadURDF('obstacles.urdf')  # Adjust position as needed

    def step(self, action):
        # Get the initial position of the robot
        initial_pos = self.env.robot.body_xyz  # Assuming 'robot' is the agent

        # Perform action in the environment
        obs, reward, done, info = self.env.step(action)

        # Get the current position of the robot after taking the action
        current_pos = self.env.robot.body_xyz

        # Check for collision between the robot and the obstacle
        contacts = p.getContactPoints(self.env.robot.body, self.obstacle)
        collision_detected = len(contacts) > 0

        # Reward or penalize the agent upon collision with the obstacle
        if collision_detected:
            # Penalize the agent for colliding with the obstacle
            reward -= 1  # Example penalty value; adjust as needed
            done = True  # End the episode on collision

        return obs, reward, done, info

    def reset(self):
        return self.env.reset()

    def render(self):
        return self.env.render()

# Rest of the code remains the same


# Instantiate the environment
env = CustomAntBulletEnv()
env.render(mode="human")

# Set the maximum episode length (increase as needed)
max_episode_steps = 2000
env.env._max_episode_steps = max_episode_steps

# Define policy architecture
policy_kwargs = dict(activation_fn=th.nn.LeakyReLU, net_arch=[512, 512])

# Instantiate the agent
model = PPO('MlpPolicy', env, learning_rate=0.0003, policy_kwargs=policy_kwargs, verbose=1, device='cpu')

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
