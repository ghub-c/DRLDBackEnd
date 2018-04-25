# -*- coding: utf-8 -*-

import numpy as np
import gym

import gym_airsim.envs
import gym_airsim

import argparse

from keras.models import Model, Sequential
from keras.layers import Input, Reshape, Dense, Activation, Flatten, Conv2D, concatenate
from keras.optimizers import Adam
from keras.callbacks import TensorBoard


from rl.agents.dqn import DQNAgent
from rl.policy import LinearAnnealedPolicy, EpsGreedyQPolicy
from rl.memory import SequentialMemory
from rl.processors import MultiInputProcessor

model = Sequential()
model.add()