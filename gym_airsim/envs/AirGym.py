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
        
        
        
        self.simage = np.zeros((20, 100), dtype=np.uint8)
        self.svelocity = np.zeros((3,), dtype=np.float32)
        self.sdistance = np.zeros((3,), dtype=np.float32)
        self.sgeofence = np.zeros((6,), dtype=np.float32)
       
        
        self.action_space = spaces.Discrete(6)
		
        self.goal = 	[137.5, -48.7]
        self.distance = np.sqrt(np.power((self.goal[0]),2) + np.power((self.goal[1]),2))
        
        self.episodeN = 0
        self.stepN = 0 
        
        self.allLogs = { 'reward':[0] }
        self.allLogs['distance'] = [self.distance]
        self.allLogs['track'] = [-2]
        self.allLogs['action'] = [1]

        self._seed()
        
        global airgym
        airgym = myAirSimClient()
        
        
    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]
    
    def computeDistance(self, goal):
        
        distance = np.sqrt(np.power((self.goal[0]),2) + np.power((self.goal[1]),2))
        
        return distance
        
    
    def state(self):
        
        return self.simage, self.svelocity, self.sdistance, self.sgeofence
    
        
    def computeReward(self, now):
	
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

        if collided == True:
            done = True
            reward = -100.0
            distance = np.sqrt(np.power((self.goal[0]-now.x_val),2) + np.power((self.goal[1]-now.y_val),2))
       
        else: 
            done = False
            reward, distance = self.computeReward(now)
        
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
        print(distance)
        self.addToLog('distance', distance)  
        
        # Terminate the episode on large cumulative amount penalties, 
        # since drone probably got into an unexpected loop of some sort
        if rewardSum < -300:
            done = True
       
        sys.stdout.write("\r\x1b[K{}/{}==>reward/depth: {:.1f}/{:.1f}   \t  {:.0f}".format(self.episodeN, self.stepN, reward, rewardSum, action))
        sys.stdout.flush()
        
        info = {"x_pos" : now.x_val, "y_pos" : now.y_val}
        
        self.simage = airgym.getScreenDepthVis()
        self.svelocity = airgym.mapVelocity()
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
        
        
        arr = np.array([[137.5, -48.7], [59.1, -15.1], [-62.3, -7.35], [123, 77.3]])
        probs = [.25, .25, .25, .25]
        indicies = np.random.choice(len(arr), 1, p=probs)
        array = (arr[indicies])
        list = (array.tolist())
        self.goal = [item for sublist in list for item in sublist]

        self.stepN = 0
        self.episodeN += 1
        
        distance = np.sqrt(np.power((self.goal[0]),2) + np.power((self.goal[1]),2))
        self.allLogs = { 'reward': [0] }
        self.allLogs['distance'] = [distance]
        self.allLogs['action'] = [1]
        
        print(self.goal)
        print(distance)
        self.simage = airgym.getScreenDepthVis()
        self.svelocity = airgym.mapVelocity()
        self.sdistance = airgym.mapDistance(self.goal)
        self.sgeofence = airgym.mapGeofence()
        
        state = self.state()
        
        return state