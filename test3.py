import numpy as np


arr = np.array([[1, 1], [2, 2], [3, 3], [4, 4]])
probs = [.25, .25, .25, .25]
indicies = np.random.choice(len(arr), 1, p=probs)

array = (arr[indicies])
list = (array.tolist())
goal = [item for sublist in list for item in sublist]

print(goal)