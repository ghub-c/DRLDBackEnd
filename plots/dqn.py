# -*- coding: utf-8 -*-
import pandas as pd

df = pd.DataFrame({
        "dqn":[1]*5,
        "DQN":[3069, 739.5, 3359, 6012, 1629],
        "Atari Games":["Alien","Amidar","Assault","Asterix","Asteroids"],
        "human":[2]*5,
        "Human":[6875, 1676, 1496, 8503, 13157]})

ax = df[["Atari Games","DQN","Human"]].plot(x='Atari Games', kind='bar', color=["g","b"], rot=45)

ax.legen(["1","2"])
