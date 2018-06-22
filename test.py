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
duration = 1
speed_x = 3
speed_y= 3

pitch, roll, yaw  = client.getPitchRollYaw()
vel = client.getVelocity()
vx = math.cos(yaw) * speed_x - math.sin(yaw) * speed_y
vy = math.sin(yaw) * speed_x + math.cos(yaw) * speed_y

drivetrain = DrivetrainType.ForwardOnly
yaw_mode = YawMode(is_rate= False, yaw_or_rate = 0)
print(vel.z_val)

client.moveByVelocity(vx = 0 ,
                      vy = 0.5 ,
                      vz = 0,
                      duration = duration + 5,
                      drivetrain = drivetrain,
                      yaw_mode = yaw_mode)
print(vel.z_val)