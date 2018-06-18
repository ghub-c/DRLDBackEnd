import logging
import numpy as np

import gym
import sys
from gym import spaces
from gym.utils import seeding
from gym.spaces import Box
from gym.spaces.box import Box

from gym_airsim.envs.myAirSimClient import *
        
from AirSimClient import *

logger = logging.getLogger(__name__)


class AirSimEnv(gym.Env):

    airgym = None
        
    def __init__(self):
        
        
        
        self.simage = np.zeros((30, 100), dtype=np.uint8)
        self.sposition = np.zeros((2,), dtype=np.float32)
        self.sdistance = np.zeros((3,), dtype=np.float32)
        self.sgeofence = np.zeros((4,), dtype=np.float32)
       
        
        self.action_space = spaces.Discrete(3)
		
        self.goal = 	[137.5, -48.7]
        
        self.episodeN = 0
        self.stepN = 0 
        
        self.allLogs = { 'reward':[0] }
        self.allLogs['distance'] = [145.87]
        self.allLogs['track'] = [-2]
        self.allLogs['action'] = [1]


        self._seed()
        
        global airgym
        airgym = myAirSimClient()
        
        
    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]
    
    def state(self):
        
        return self.simage, self.sposition, self.sdistance, self.sgeofence
        
    def computeReward(self, now, track_now):
	
		# test if getPosition works here liek that
		# get exact coordiantes of the tip
      
        distance_now = np.sqrt(np.power((self.goal[0]-now.x_val),2) + np.power((self.goal[1]-now.y_val),2))
        
        distance_before = self.allLogs['distance'][-1]
              
        r = -1
        
        r = r + (distance_before - distance_now)
            
        return r, distance_now
		
    
    def _step(self, action):
        assert self.action_space.contains(action), "%r (%s) invalid"%(action, type(action))
        
        self.addToLog('action', action)
        
        self.stepN += 1

        collided = airgym.take_action(action)
        
        now = airgym.getPosition()
        track = airgym.goal_direction(self.goal, now) 

        if collided == True:
            done = True
            reward = -100.0
            distance = np.sqrt(np.power((self.goal[0]-now.x_val),2) + np.power((self.goal[1]-now.y_val),2))
       
        else: 
            done = False
            reward, distance = self.computeReward(now, track)
        
        # Youuuuu made it
        if distance < 3:
            landed = airgym.arrived()
            if landed == True:
                done = True
                reward = 100.0
            
                with open("reached.txt", "a") as myfile:
                    myfile.write(str(self.episodeN) + ", ")
            
        
        self.addToLog('reward', reward)
        rewardSum = np.sum(self.allLogs['reward'])
        self.addToLog('distance', distance)
        self.addToLog('track', track)      
            
        # Terminate the episode on large cumulative amount penalties, 
        # since drone probably got into an unexpected loop of some sort
        if rewardSum < -300:
            done = True
       
        sys.stdout.write("\r\x1b[K{}/{}==>reward/depth: {:.1f}/{:.1f}   \t {:.0f}  {:.0f}".format(self.episodeN, self.stepN, reward, rewardSum, track, action))
        sys.stdout.flush()
        
        info = {"x_pos" : now.x_val, "y_pos" : now.y_val}
        
        self.simage = airgym.getScreenDepthVis(track)
        self.sposition = airgym.mapPosition()
        self.sdistance = airgym.mapDistance(self.goal)
        self.sgeofence = airgym.mapGeofence()
        
        state = self.state()
        return state, reward, done, info

    def addToLog (self, key, value):
        if key not in self.allLogs:
            self.allLogs[key] = []
        self.allLogs[key].append(value)
        
    def _reset(self):
        """
        Resets the state of the environment and returns an initial observation.
        
        # Returns
            observation (object): The initial observation of the space. Initial reward is assumed to be 0.
        """
        airgym.AirSim_reset()
        
        totalrewards = np.sum(self.allLogs['reward'])
        with open("rewards.txt", "a") as myfile:
            myfile.write(str(totalrewards) + ", ")
            
        self.stepN = 0
        self.episodeN += 1
        
        self.allLogs = { 'reward': [0] }
        self.allLogs['distance'] = [145.87]
        self.allLogs['track'] = [-2]
        self.allLogs['action'] = [1]
        
        
        now = airgym.getPosition()
        track = airgym.goal_direction(self.goal, now)
        
        self.simage = airgym.getScreenDepthVis(track)
        self.sposition = airgym.mapPosition()
        self.sdistance = airgym.mapDistance(self.goal)
        self.sgeofence = airgym.mapGeofence()
        
        state = self.state()
        
        return state