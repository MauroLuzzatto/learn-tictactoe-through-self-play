import numpy as np


class Player:
    def __init__(self, color, episodes: int):
        self.color = color
        self.reward_array = np.zeros(episodes)
        self.reset_reward()
        self.name = f"Player {color}"

    def reset_reward(self):
        self.reward = 0

    def add_reward(self, new_reward):
        self.reward += new_reward

    def save_reward(self, episode):
        self.reward_array[episode] = self.reward
