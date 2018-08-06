# -*- coding: utf-8 -*-

import numpy as np
import gym

import gym_airsim.envs
import gym_airsim

import argparse

from keras.models import Model, Sequential
from keras.layers import Input, Reshape, Dense, Flatten, Conv2D, concatenate
from keras.optimizers import Adam
from keras.callbacks import TensorBoard
from keras.utils import plot_model

from callbacks import *
from rl.agents.dqn import DQNAgent
from rl.policy import LinearAnnealedPolicy, EpsGreedyQPolicy
from rl.memory import SequentialMemory
from rl.processors import MultiInputProcessor
import os
os.environ["PATH"] += os.pathsep + 'C:/Program Files (x86)/Graphviz2.38/bin/'

parser = argparse.ArgumentParser()
parser.add_argument('--mode', choices=['train', 'test'], default='train')
parser.add_argument('--env-name', type=str, default='AirSimEnv-v42')
parser.add_argument('--weights', type=str, default=None)
args = parser.parse_args()


env = gym.make(args.env_name)
np.random.seed(123)
env.seed(123)
nb_actions = env.action_space.n


#Obtaining shapes from Gym environment
img_shape = env.simage.shape
vel_shape = env.stotalvelocity.shape
dst_shape = env.stotaldistance.shape
geo_shape = env.stotalgeofence.shape

#Keras-rl interprets an extra dimension at axis=0
#added on to our observations, so we need to take it into account
img_kshape = (1,) + img_shape

#Sequential model for convolutional layers applied to image
image_model = Sequential()
image_model.add(Conv2D(32, (4, 4), strides=(4, 4) ,activation='relu', input_shape=img_kshape, data_format = "channels_first"))
image_model.add(Conv2D(64, (3, 3), strides=(2, 2),  activation='relu'))
image_model.add(Flatten())

#Input and output of the Sequential model
image_input = Input(img_kshape)
encoded_image = image_model(image_input)

#Inputs and reshaped tensors for concatenate after with the image
velocity_input = Input((1,) + vel_shape)
distance_input = Input((1,) + dst_shape)
geofence_input = Input((1,) + geo_shape)
vel = Reshape(vel_shape)(velocity_input)
dst = Reshape(dst_shape)(distance_input)
geo = Reshape(geo_shape)(geofence_input)


#Concatenation of image, position, distance and geofence values.
#3 dense layers of 256 units
denses = concatenate([encoded_image, vel, dst, geo])
denses = Dense(256, activation='relu')(denses)
denses = Dense(256, activation='relu')(denses)
denses = Dense(256, activation='relu')(denses)
#Last dense layer with nb_actions for the output
predictions = Dense(nb_actions, kernel_initializer='zeros', activation='linear')(denses)

model = Model(
        inputs=[image_input, velocity_input, distance_input, geofence_input],
        outputs=predictions
        )


train = True



# Finally, we configure and compile our agent. You can use every built-in Keras optimizer and
# even the metrics!
memory = SequentialMemory(limit=100000, window_length=1)                        #reduce memmory

processor = MultiInputProcessor(nb_inputs=4)

# Select a policy. We use eps-greedy action selection, which means that a random action is selected
# with probability eps. We anneal eps from 1.0 to 0.1 over the course of 1M steps. This is done so that
# the agent initially explores the environment (high eps) and then gradually sticks to what it knows
# (low eps). We also set a dedicated eps value that is used during testing. Note that we set it to 0.05c
# so that the agent still performs some random actions. This ensures that the agent cannot get stuck.
policy = LinearAnnealedPolicy(EpsGreedyQPolicy(), attr='eps', value_max=1., value_min=.1, value_test=0.0,
                              nb_steps=100000)

dqn = DQNAgent(model=model, processor=processor, nb_actions=nb_actions, memory=memory, nb_steps_warmup=50, 
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
    #tb_log_dir = 'logs/tmp'
    #callbacks = [TensorBoard(log_dir=tb_log_dir, histogram_freq=0)]
    dqn.fit(env, callbacks = callbacks, nb_steps=250000, visualize=False, verbose=0, log_interval=100)
    
    
    # After training is done, we save the final weights.
    dqn.save_weights('dqn_weights.h5f'.format(args.env_name), overwrite=True)

else:

    dqn.load_weights('dqn_weights.h5f')
    dqn.test(env, nb_episodes=100, visualize=False)


