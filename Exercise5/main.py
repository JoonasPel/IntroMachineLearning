# Load OpenAI Gym and other necessary packages
import gym
import random
import numpy
import time

# Environment
env = gym.make("Taxi-v3")

# Training parameters for Q learning
alpha = 0.9 # Learning rate
gamma = 0.9 # Future reward discount factor
num_of_episodes = 1000
num_of_steps = 500 # per each episode

# Q tables for rewards
#Q_reward = -100000*numpy.ones((500,6)) # All same
Q_reward = -100000*numpy.random.random((500, 6)) # Random

# Training w/ random sampling of actions
# YOU WRITE YOUR CODE HERE

# Testing
state = env.reset()
tot_reward = 0
for t in range(50):
    action = numpy.argmax(Q_reward[state,:])
    state, reward, done, info = env.step(action)
    tot_reward += reward
    env.render()
    time.sleep(1)
    if done:
        print("Total reward %d" %tot_reward)
        break