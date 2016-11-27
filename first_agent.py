#! /usr/bin/env python3
# -*- coding: utf-8

import gym
import universe

env = gym.make('flashgames.DuskDrive-v0')
observation_n = env.reset()

while True:
    action_n = [[('KeyEvent', 'ArrowUp', True)] for ob in observation_n] # your agent here
    observation_n, reward_n, done_n, info = env.step(action_n)
    env.render()
