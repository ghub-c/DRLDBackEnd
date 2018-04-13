from AirSimClient import *

client = MultirotorClient()
client.confirmConnection()
client.enableApiControl(True)
client.armDisarm(True)
print("Connected")

initialZ=-4
#Move drone forward until it collides
client.moveToZ(initialZ, 3)
client.moveByVelocity(10,0,0.3,18)
'''
# connect to the AirSim simulator 
client = CarClient()
client.confirmConnection()
client.enableApiControl(True)
car_controls = CarControls()
car_state = client.getCarState()
print (car_state.kinematics_true)
'''