from AirSimClient import *

# connect to the AirSim simulator 
client = CarClient()
client.confirmConnection()
client.enableApiControl(True)
car_controls = CarControls()
car_state = client.getCarState()
print (car_state.kinematics_true)

