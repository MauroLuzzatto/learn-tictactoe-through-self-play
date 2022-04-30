# -*- coding: utf-8 -*-
"""
Created on Wed Jun  5 15:22:09 2019

@author: MauroLuzzatto


Description:

Implementation of environment and a q-learning algorithm
that learns to play TicTacToe through self-play

"""

import time

import gym
import numpy as np
from tqdm import tqdm

from Qagent import Qagent
from utils import (
    create_state_dictionary,
    load_qtable,
    play_tictactoe,
    reshape_state,
    save_qtable,
)

state_dict = create_state_dictionary()


# init the enviornment
env = gym.envs.make("TTT-v0", small=-1, large=10)

state_size = env.observation_space.n
action_size = env.action_space.n

# player1 = 1
# player2 = 2


learning_parameters = {"learning_rate": 1.0, "gamma": 0.9}

exploration_parameters = {
    "epsilon": 1.0,
    "max_epsilon": 1.0,
    "min_epsilon": 0.0,
    "decay_rate": 0.000001,
}

# set training parameters
episodes = 2_000  # 10**6 * 2
max_steps = 9

# name of the qtable when saved
name = "qtable"
load = False
save = True
test = True

num_test_games = 1

# player1_reward_array = np.zeros(episodes)
# player2_reward_array = np.zeros(episodes)

qagent = Qagent(state_size, action_size, learning_parameters, exploration_parameters)


class Game(object):
    def __init__(self):
        self.name = None

    def load():
        try:
            qagent.qtable = load_qtable(name)
            print("{}.npy loaded!".format(name))
        except:
            print("qtable could not be loaded!")

    def learn_to_play():
        pass

    def print_progress():
        pass

    def save():
        save_qtable(qagent.qtable, name)
        qtable = qagent.qtable

    def play():
        pass


if load:
    try:
        qagent.qtable = load_qtable(name)
        print("{}.npy loaded!".format(name))
    except:
        print("qtable could not be loaded!")


class Player:
    def __init__(self, color, episodes: int):
        self.color = color
        self.reward_array = np.zeros(episodes)
        self.reset_reward()

    def reset_reward(self):
        self.reward = 0

    def add_reward(self, new_reward):
        self.reward += new_reward

    def save_reward(self, episode):
        self.reward_array[episode] = self.reward


# TODO: Track the actions taken over time while playing,  9*8*7*6*5*4*3*2*1

# start the training
start_time = time.time()

player_1 = Player(color=1, episodes=episodes)
player_2 = Player(color=2, episodes=episodes)

for episode in tqdm(range(episodes)):
    state = env.reset()
    state = state_dict[reshape_state(state)]

    action_space = np.arange(9)

    player_1.reset_reward()
    player_2.reset_reward()

    # change start of players, randomly change the order players to start the game
    start = np.random.randint(2)  # integer either 0 or 1

    for _step in range(start, max_steps + start):
        # alternate the moves of the players
        if _step % 2 == 0:

            # player 1
            action = qagent.get_action(state, action_space)

            # remove action from the action space
            action_space = action_space[action_space != action]

            new_state, reward, done, _ = env.step((action, player_1.color))
            new_state = state_dict[reshape_state(new_state)]

            qagent.qtable[state, action] = qagent.update_qtable(
                state, new_state, action, reward, done
            )
            # new state
            state = new_state
            player_1.add_reward(reward)

        else:

            # player 2
            action = qagent.get_action(state, action_space)
            # remove action from the action space
            action_space = action_space[action_space != action]

            new_state, reward, done, _ = env.step((action, player_2.color))
            new_state = state_dict[reshape_state(new_state)]

            qagent.qtable[state, action] = qagent.update_qtable(
                state, new_state, action, reward, done
            )

            # new state
            state = new_state
            player_2.add_reward(reward)

        # stopping criterion
        if done == True:
            break

    # reduce epsilon for exporation-exploitation tradeoff
    qagent.update_epsilon(episode)

    player_1.save_reward(episode)
    player_2.save_reward(episode)

    if episode % 1_0000 == 0:

        # def print_progress():
        print("episode: {}, epsilon: {}".format(episode, round(qagent.epsilon, 2)))
        print(
            "elapsed time [min]: {}, done [%]: {}".format(
                round((time.time() - start_time) / 60.0, 2), episode / episodes * 100
            )
        )
        print(np.sum(qagent.qtable))


if save:
    save_qtable(qagent.qtable, name)
    qtable = qagent.qtable

# test the algorithm with playing against it
if test:
    play_tictactoe(env, qtable, max_steps, state_dict)
