from operator import itemgetter

goal = [6.83, -95.41]

pos = [1]

valor = itemgetter(*pos)(goal)

print(valor)