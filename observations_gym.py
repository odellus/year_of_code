#! /usr/bin/env python
import gym
import numpy as np
from matplotlib import pyplot as plt
import time


env = gym.make('Boxing-ram-v0')

img = env.render(mode='rgb_array')
observation = env.reset()
ih, iw, n_channels = img.shape

n_episodes = 20
n_tsteps = 100
n_images = n_episodes * n_tsteps

images = np.zeros((n_images, ih, iw, n_channels))
obs = []
rewards = []
actions = []

for i_episode in range(n_episodes):
    observation = env.reset()
    for t in range(100):
        # Get the image from the screen. This will be part of observed quantity.
        img = env.render(mode='rgb_array')
        images[n_tsteps*i_episode + t,:,:,:] = img

        # We're only randomly sampling right now. Later this will be
        # replace by the argmax of a final softmax layer from a CNN of the image
        # combined in some way with the observation given directly by gym.
        action = env.action_space.sample()

        # Step forward the environment with agent taking the sampled action.
        # Collect new observations
        observation, reward, done, info = env.step(action)
        obs.append(observation)
        rewards.append(reward)
        actions.append(action)

        if done:
            print("Episode finished after {} timesteps".format(t+1))

plt.ion()
for k in range(n_images):
    plt.imshow(images[k,:,:,:])
    time.sleep(1.)
