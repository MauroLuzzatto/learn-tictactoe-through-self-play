"""
Implementation of environment and a q-learning algorithm
that learns to play TicTacToe through self-play
"""

import time

import gym
import numpy as np
from tqdm import tqdm
import gym_TicTacToe

from qagent import Qagent
from player import Player
from utils import (
    create_state_dictionary,
    load_qtable,
    play_tictactoe,
    reshape_state,
    save_qtable,
)


env = gym.envs.make("TTT-v0", small=-1, large=10)

state_dict = create_state_dictionary()
state_size = env.observation_space.n
action_size = env.action_space.n


# set training parameters
episodes = 90_000  # 10**6 * 2
max_steps = 9

# name of the qtable when saved
load = False
save = True
test = True

num_test_games = 1

learning_parameters = {"learning_rate": 1.0, "gamma": 0.9}

exploration_parameters = {
    "max_epsilon": 1.0,
    "min_epsilon": 0.0,
    "decay_rate": 0.000005,
}

name = f"qtable_{episodes}"
folder = "tables"


qagent = Qagent(state_size, action_size, learning_parameters, exploration_parameters)


if load:
    qagent.qtable = load_qtable(folder, name)


def play(player, state, action_space):
    """_summary_

    Args:
        player (_type_): _description_
        state (_type_): _description_
        action_space (_type_): _description_

    Returns:
        _type_: _description_
    """
    action = qagent.get_action(state, action_space)

    # remove action from the action space
    action_space = action_space[action_space != action]

    new_state, reward, done, _ = env.step((action, player.color))
    new_state = state_dict[reshape_state(new_state)]

    qagent.qtable[state, action] = qagent.update_qtable(
        state, new_state, action, reward, done
    )
    # new state
    state = new_state
    player.add_reward(reward)
    return state, action_space, done


start_time = time.time()

player_1 = Player(color=1, episodes=episodes)
player_2 = Player(color=2, episodes=episodes)

for episode in tqdm(range(episodes)):
    state = env.reset()
    state = state_dict[reshape_state(state)]

    action_space = np.arange(9)

    player_1.reset_reward()
    player_2.reset_reward()

    # change start of players, randomly change the order players to start the game,
    # integer either 0 or 1
    start = np.random.randint(2)

    for _step in range(start, max_steps + start):

        # alternate the moves of the players
        if _step % 2 == 0:
            state, action_space, done = play(player_1, state, action_space)
        else:
            state, action_space, done = play(player_2, state, action_space)

        if done == True:
            break

    # reduce epsilon for exporation-exploitation tradeoff
    qagent.update_epsilon(episode)
    player_1.save_reward(episode)
    player_2.save_reward(episode)

    if episode % 1_0000 == 0:

        sum_q_table = np.sum(qagent.qtable)
        time_passed = round((time.time() - start_time) / 60.0, 2)

        print(
            f"episode: {episode}, \
            epsilon: {round(qagent.epsilon, 2)}, \
            sum q-table: {sum_q_table}, \
            elapsed time [min]: {time_passed},  \
            done [%]: {episode / episodes * 100} \
            "
        )


if save:
    save_qtable(qagent.qtable, folder, name)

# test the algorithm with playing against it
if test:
    play_tictactoe(env, qagent.qtable, max_steps, state_dict)
