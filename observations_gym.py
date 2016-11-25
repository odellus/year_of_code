#! /usr/bin/env python
import gym
import numpy as np
from matplotlib import pyplot as plt
import time

def random_sample(environment):
    env = gym.make(environment)

    img = env.render(mode='rgb_array')
    observation = env.reset()
    n_obs = observation.shape[0]
    ih, iw, n_channels = img.shape

    n_episodes = 20
    n_timesteps = 100
    n_images = n_episodes * n_timesteps

    # Create empty arrays to store information from environment.
    images = np.zeros((n_images, ih, iw, n_channels))
    obs = np.zeros((n_images, n_obs))
    rewards = np.zeros((n_images,))
    actions = np.zeros((n_images,))


    for i_episode in range(n_episodes):
        observation = env.reset()
        for t in range(n_timesteps):
            # Keeps track of the total number of observations, rewards, etc...
            k = n_timesteps*i_episode + t
            # Get the image from the screen. This will be part of observed quantity.
            img = env.render(mode='rgb_array')
            images[k,:,:,:] = img

            # We're only randomly sampling right now. Later this will be
            # replace by the argmax of a final softmax layer from a CNN of the image
            # combined in some way with the observation given directly by gym.
            action = env.action_space.sample()

            # Step forward the environment with agent taking the sampled action.
            # Collect new observations
            observation, reward, done, info = env.step(action)
            obs[k,:] = observation
            rewards[k] = reward
            actions[k] = action

            if done:
                print("Episode finished after {} timesteps".format(t+1))
                break

    return obs, rewards, actions, images

def test_random_sample():
    obs, rewards, actions, images = random_sample('Boxing-ram-v0')

    print(images.shape)
    rando = np.random.randint(0,len(images))
    plt.imshow(images[rando,:,:,:])
    plt.show()

if __name__ == "__main__":
    test_random_sample()
