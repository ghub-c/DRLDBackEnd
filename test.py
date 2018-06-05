import math
import numpy as np
from AirSimClient import *


client = MultirotorClient()
client.confirmConnection()
client.enableApiControl(True)
client.armDisarm(True)
#client.takeoff()
#client.hover()
client.moveToZ(-6, 5)
duration = 5
speed_x = -3
speed_y= 3

pitch, roll, yaw  = client.getPitchRollYaw()
vel = client.getVelocity()
vx = math.cos(yaw) * speed_x - math.sin(yaw) * speed_y
vy = math.sin(yaw) * speed_x + math.cos(yaw) * speed_y

drivetrain = DrivetrainType.ForwardOnly
yaw_mode = YawMode(is_rate= False, yaw_or_rate = 0)

client.moveByVelocityZ(vx = (vx +vel.x_val)/2 ,
                       vy = (vy +vel.y_val)/2 , #do this to try and smooth the movement
                       z = -6,
                       duration = duration + 5,
                       drivetrain = drivetrain,
                       yaw_mode = yaw_mode)
