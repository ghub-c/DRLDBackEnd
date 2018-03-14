# -*- coding: utf-8 -*-

import numpy as np
import gym

import gym_airsim.envs
import gym_airsim


import argparse

from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten, Conv2D
from keras.optimizers import Adam


from rl.agents.dqn import DQNAgent
from rl.policy import LinearAnnealedPolicy, EpsGreedyQPolicy
from rl.memory import SequentialMemory

from callbacks import *


parser = argparse.ArgumentParser()
parser.add_argument('--mode', choices=['train', 'test'], default='train')
parser.add_argument('--env-name', type=str, default='AirSimEnv-v42')
parser.add_argument('--weights', type=str, default=None)
args = parser.parse_args()

# Get the environment and extract the number of actions.
env = gym.make(args.env_name)
np.random.seed(123)
env.seed(123)
nb_actions = env.action_space.n


# Next, we build our model. We use the same model that was described by Mnih et al. (2015).
INPUT_SHAPE = (30, 100)
WINDOW_LENGTH = 1
# Next, we build our model. We use the same model that was described by Mnih et al. (2015).
input_shape = (WINDOW_LENGTH,) + INPUT_SHAPE


model = Sequential()
model.add(Conv2D(32, (4, 4), strides=(4, 4) ,activation='relu', input_shape=input_shape, data_format = "channels_first"))
model.add(Conv2D(64, (3, 3), strides=(2, 2),  activation='relu'))
model.add(Conv2D(64, (1, 1), strides=(1, 1),  activation='relu'))
model.add(Flatten())
model.add(Dense(512))
model.add(Activation('relu'))
model.add(Dense(nb_actions))
model.add(Activation('linear'))
print(model.summary())


train = False

# Finally, we configure and compile our agent. You can use every built-in Keras optimizer and
# even the metrics!
memory = SequentialMemory(limit=25000, window_length=WINDOW_LENGTH)                        #reduce memmory


# Select a policy. We use eps-greedy action selection, which means that a random action is selected
# with probability eps. We anneal eps from 1.0 to 0.1 over the course of 1M steps. This is done so that
# the agent initially explores the environment (high eps) and then gradually sticks to what it knows
# (low eps). We also set a dedicated eps value that is used during testing. Note that we set it to 0.05c
# so that the agent still performs some random actions. This ensures that the agent cannot get stuck.
policy = LinearAnnealedPolicy(EpsGreedyQPolicy(), attr='eps', value_max=1., value_min=.1, value_test=0.0,
                              nb_steps=25000)

dqn = DQNAgent(model=model, nb_actions=nb_actions, memory=memory, nb_steps_warmup=50, 
               enable_double_dqn=True, 
               enable_dueling_network=False, dueling_type='avg', 
               target_model_update=1e-2, policy=policy, gamma=.99)

dqn.compile(Adam(lr=0.00025), metrics=['mae'])


if train:
    # Okay, now it's time to learn something! We visualize the training here for show, but this
    # slows down training quite a lot. You can always safely abort the training prematurely using
    # Ctrl + C.
    
    
    log_filename = 'dqn_{}_log.json'.format(args.env_name)
    callbacks = [FileLogger(log_filename, interval=10)]
    callbacks += [TrainEpisodeLogger()]
    dqn.fit(env, callbacks=callbacks, nb_steps=75000, visualize=False, verbose=0, log_interval=100)
    
    
    # After training is done, we save the final weights.
    dqn.save_weights('dqn_{}_weights.h5f'.format(args.env_name), overwrite=True)

else:

    dqn.load_weights('checkpoint_reward_180.15931910639895.h5f')
    dqn.test(env, nb_episodes=100, visualize=False)
    