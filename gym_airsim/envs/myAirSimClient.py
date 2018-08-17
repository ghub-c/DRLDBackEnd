import numpy as np
from operator import itemgetter
import time
import math
import cv2
from pylab import array, uint8 
from PIL import Image


from AirSimClient import *


class myAirSimClient(MultirotorClient):

    def __init__(self):        
        self.img1 = None
        self.img2 = None

        MultirotorClient.__init__(self)
        MultirotorClient.confirmConnection(self)
        self.enableApiControl(True)
        self.armDisarm(True)
    
        self.home_pos = self.getPosition()
    
        self.home_ori = self.getOrientation()
        
        #Define your geofence inside the map
        
        self.minx = -100
        self.maxx = 150
        self.miny = -70
        self.maxy = 100
        self.minz = -2
        self.maxz = -20
        '''
        self.minx = -50
        self.maxx = 180
        self.miny = -90
        self.maxy = 30
        self.minz = -2
        self.maxz = -20
        '''
        self.z = -4
        
    def movement(self, speed_x, speed_y, speed_z, duration):
        
        pitch, roll, yaw  = self.getPitchRollYaw()
        vel = self.getVelocity()
        drivetrain = DrivetrainType.ForwardOnly
        yaw_mode = YawMode(is_rate= False, yaw_or_rate = 0)

        self.moveByVelocity(vx = vel.x_val + speed_x,
                            vy = vel.y_val + speed_y,
                            vz = vel.z_val + speed_z,
                            duration = duration,
                            drivetrain = drivetrain,
                            yaw_mode = yaw_mode)
        

    
    def take_action(self, action):
        
		 #check if copter is on level cause sometimes he goes up without a reason

        start = time.time()
        duration = 1 
        
        outside = self.geofence(self.minx, self.maxx, 
                                self.miny, self.maxy,
                                self.minz, self.maxz)
        
        if action == 0:
            
            self.movement(0.5, 0, 0, duration)
    
        elif action == 1:
         
            self.movement(-0.5, 0, 0, duration)
                
        elif action == 2:
            
            self.movement(0, 0.5, 0, duration)
            
                
        elif action == 3:
                    
            self.movement(0, -0.5, 0, duration)
            
        elif action == 4:
                    
            self.movement(0, 0, 0.5, duration)
                
        elif action == 5:
                    
            self.movement(0, 0, -0.5, duration)      
        
        while duration > time.time() - start:
                if self.getCollisionInfo().has_collided == True:
                    return True    
                if outside == True:
                    return True
                
        return False
    
    def geofence(self, minx, maxx, miny, maxy, minz, maxz):
        
        outside = False
        
        if (self.getPosition().x_val < minx) or (self.getPosition().x_val > maxx):
                    return True
        if (self.getPosition().y_val < miny) or (self.getPosition().y_val > maxy):
                    return True
        if (self.getPosition().z_val > minz) or (self.getPosition().z_val < maxz):
                    return True
                
        return outside
    
    def arrived(self):
        
        landed = self.moveToZ(0, 1)
    
        if landed == True:
            return landed
        
        if (self.getPosition().z_val > -1):
            return True
        
    def goal_direction(self, goal, pos):
        
        pitch, roll, yaw  = self.getPitchRollYaw()
        yaw = math.degrees(yaw) 
        
        pos_angle = math.atan2(goal[1] - pos.y_val, goal[0]- pos.x_val)
        pos_angle = math.degrees(pos_angle) % 360

        track = math.radians(pos_angle - yaw)  
        
        return ((math.degrees(track) - 180) % 360) - 180   
    
    '''
    def mapPosition(self):
        
        xval = self.getPosition().x_val
        yval = self.getPosition().y_val
        
        position = np.array([xval, yval])
        
        return position
    '''
    def mapVelocity(self):
        
        vel = self.getVelocity()
        
        velocity = np.array([vel.x_val, vel.y_val, vel.z_val])
        
        return velocity
    
    def mapGeofence(self):
        
        xpos = self.getPosition().x_val
        ypos = self.getPosition().y_val
        zpos = self.getPosition().z_val
        
        geox1 = self.maxx - xpos
        geox2 = self.minx - xpos
        geoy1 = self.maxy - ypos
        geoy2 = self.miny - ypos
        geoz1 = self.maxz - zpos
        geoz2 = self.minz - zpos
        
        geofence = np.array([geox1, geox2, geoy1, geoy2, geoz1, geoz2])
        
        return geofence
    
    def mapDistance(self, goal):
        
        x = [0]
        y = [1]
        goalx = itemgetter(*x)(goal)
        goaly = itemgetter(*y)(goal)
        xdistance = (goalx - (self.getPosition().x_val))
        ydistance = (goaly - (self.getPosition().y_val))
        meandistance = np.sqrt(np.power((goalx -self.getPosition().x_val),2) + np.power((goaly - self.getPosition().y_val),2))
        
        distances = np.array([xdistance, ydistance, meandistance])
        
        return distances
    
    def getScreenDepthVis(self):

        responses = self.simGetImages([ImageRequest(0, AirSimImageType.DepthPerspective, True, False)])
        img1d = np.array(responses[0].image_data_float, dtype=np.float)
        img1d = 255/np.maximum(np.ones(img1d.size), img1d)
        img2d = np.reshape(img1d, (responses[0].height, responses[0].width))
        
        
        image = np.invert(np.array(Image.fromarray(img2d.astype(np.uint8), mode='L')))
        
        factor = 10
        maxIntensity = 255.0 # depends on dtype of image data
        
        
        # Decrease intensity such that dark pixels become much darker, bright pixels become slightly dark 
        newImage1 = (maxIntensity)*(image/maxIntensity)**factor
        newImage1 = array(newImage1,dtype=uint8)
        
        
        small = cv2.resize(newImage1, (0,0), fx=0.39, fy=0.38)
        
       
        cut = small[20:40,:]
        
        '''
        info_section = np.zeros((10,cut.shape[1]),dtype=np.uint8) + 255
       
        info_section[9,:] = 0
        
        line = np.int((((track - -180) * (100 - 0)) / (180 - -180)) + 0)
        
        if line != (0 or 100):
            info_section[:,line-1:line+2]  = 0
        elif line == 0:
            info_section[:,0:3]  = 0
        elif line == 100:
            info_section[:,info_section.shape[1]-3:info_section.shape[1]]  = 0
           
        total = np.concatenate((info_section, cut), axis=0)
        '''
        
        #cv2.imshow("Test", total)
        #cv2.waitKey(0)

        return cut


    def AirSim_reset(self):
        
        self.reset()
        time.sleep(0.2)
        self.enableApiControl(True)
        self.armDisarm(True)
        time.sleep(1)
        self.moveToZ(self.z, 3) 
        time.sleep(3)
