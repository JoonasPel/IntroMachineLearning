# Load OpenAI Gym and other necessary packages
import gym
import random
import numpy as np
import time

# Environment
env = gym.make("Taxi-v3").env

# Training parameters for Q learning
alpha = 1.0  # Learning rate .WAS 0.9
gamma = 0.9  # Future reward discount factor
num_of_episodes = 500
num_of_steps = 500  # per each episode

# Q tables for rewards
# Q_reward = -100000*numpy.ones((500,6)) # All same
Q_reward = 1 * np.random.random((500, 6))  # Random

# Training w/ random sampling of actions
print("Training...")
for episode in range(0, num_of_episodes):
    old_state = env.reset()
    for step in range(0, num_of_steps):
        random_action = random.randrange(0, 6)
        new_state, reward, done, info = env.step(random_action)
        Q_reward[old_state][random_action] = reward + gamma * np.amax(Q_reward[new_state])
        old_state = new_state
        if done:
            break

# Testing
print("Testing starts")
time.sleep(1)

state = env.reset()
tot_reward = 0
for t in range(50):
    action = np.argmax(Q_reward[state, :])
    state, reward, done, info = env.step(action)
    tot_reward += reward
    env.render()
    time.sleep(1)
    if done:
        print("Total reward %d" % tot_reward)
        break
