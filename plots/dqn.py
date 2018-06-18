# -*- coding: utf-8 -*-
import pandas as pd

df = pd.DataFrame({
        "dqn":[1]*10,
        "DQN":[3069, 739.5, 3359, 6012, 1629, 85641, 429.7, 26300, 6846, 42.4],
        "Atari Games":["Video Pinball","Boxing","Breakout","Star Gunner","Robotank", "Atlantis",
                       "Bank Heist", "Battle Zone", "Beam Rider", "Bowling"],
        "human":[2]*10,
        "Human":[6875, 1676, 1496, 8503, 13157, 29028, 734.4, 37800, 5775, 154.8]})

ax = df[["Atari Games","DQN","Human"]].plot(x='Atari Games', kind='bar', color=["g","b"], rot=45)

