# -*- coding: utf-8 -*-
from flask import Flask, request
import numpy as np
import gym

import gym_airsim
import gym_airsim.envs

import argparse

from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten, Conv2D
from keras.optimizers import Adam


from rl.agents.dqn import DQNAgent
from rl.policy import LinearAnnealedPolicy, EpsGreedyQPolicy
from rl.memory import SequentialMemory

from callbacks import FileLogger

app = Flask(__name__)


@app.route("/test", methods = ['POST'])
def test():
    
    data = request.data
    print(data)
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['train', 'test'], default='train')
    parser.add_argument('--env-name', type=str, default='AirSimEnv-v42')
    parser.add_argument('--weights', type=str, default=None)
    args = parser.parse_args()
    
  
    env = gym.make(args.env_name)
    np.random.seed(123)
    env.seed(123)
    nb_actions = env.action_space.n
    
    
    INPUT_SHAPE = (30, 100)
    WINDOW_LENGTH = 1
    
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
    
    
    
    memory = SequentialMemory(limit=15000, window_length=WINDOW_LENGTH)                     
    
 
    policy = LinearAnnealedPolicy(EpsGreedyQPolicy(), attr='eps', value_max=1., value_min=.1, value_test=0.0,
                                  nb_steps=15000)
    
    dqn = DQNAgent(model=model, nb_actions=nb_actions, memory=memory, nb_steps_warmup=50, 
                   enable_double_dqn=True, 
                   enable_dueling_network=False, dueling_type='avg', 
                   target_model_update=1e-2, policy=policy, gamma=.99)
    
    dqn.compile(Adam(lr=0.00025), metrics=['mae'])
    
    dqn.load_weights('dqn_AirSimEnv-v42_weights.h5f')
    dqn.test(env, nb_episodes=10, visualize=False)
    
    return 'Tested'

@app.route("/train")
def train():
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['train', 'test'], default='train')
    parser.add_argument('--env-name', type=str, default='AirSimEnv-v42')
    parser.add_argument('--weights', type=str, default=None)
    args = parser.parse_args()
    
    env = gym.make(args.env_name)
    np.random.seed(123)
    env.seed(123)
    nb_actions = env.action_space.n
    
    
    INPUT_SHAPE = (30, 100)
    WINDOW_LENGTH = 1
    
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
    
    

    memory = SequentialMemory(limit=15000, window_length=WINDOW_LENGTH)                       
    
    
    
    policy = LinearAnnealedPolicy(EpsGreedyQPolicy(), attr='eps', value_max=1., value_min=.1, value_test=0.0,
                                  nb_steps=15000)
    
    dqn = DQNAgent(model=model, nb_actions=nb_actions, memory=memory, nb_steps_warmup=50, 
                   enable_double_dqn=True, 
                   enable_dueling_network=False, dueling_type='avg', 
                   target_model_update=1e-2, policy=policy, gamma=.99)
    
    dqn.compile(Adam(lr=0.00025), metrics=['mae'])
    
 
    
    log_filename = 'dqn_{}_log.json'.format(args.env_name)
    callbacks = [FileLogger(log_filename, interval=100)]
    dqn.fit(env, callbacks=callbacks, nb_steps=45000, visualize=False, verbose=2, log_interval=100)
    
  
    dqn.save_weights('dqn_{}_weights.h5f'.format(args.env_name), overwrite=True)

    return 'Trained'

if __name__ == "__main__":
    app.run()
