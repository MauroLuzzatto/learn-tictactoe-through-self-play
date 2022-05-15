import os
from collections import Counter
from itertools import product

import matplotlib.pyplot as plt
import numpy as np


def create_state_dictionary():
    """
    create a dictionary, that encodes the game postions (3x3) into a state number (int)
    Returns:
      state_dict (dict): key = game position, value = state number
    """
    state_number = 0
    state_dict = {}

    # create all digit combinations with 0,1,3 for 9 digit number
    for game_position in set(product(set(range(3)), repeat=9)):
        # count the digits per tuple
        count_digits = Counter(game_position)
        # remove all board situation, which are not possible
        if abs(count_digits[1] - count_digits[2]) <= 1:
            state_dict[game_position] = state_number
            state_number += 1
    print(f"Number of legal states: {state_number}")
    return state_dict


def reshape_state(state):
    """
    transfrom the 3x3 board numpy array into a flattend tuple
    Args:
        state (array): 3x3 numpy array, representing the board postions = state
    Returns:
        state (tuple): the flattened numy array converted into a tuple
    """
    return tuple(state.reshape(1, -1)[0])


def create_plot(*players):
    """
    plot the rewards of the player 1 and 2 versus the number of training episode in self-play
    Args:
        players List[Player]: rewards over training episoded player1
    """
    plt.figure(figsize=(10, 5))
    plt.title("reward over time")
    for player in players:
        plt.plot(
            range(len(player.reward_array)),
            player.reward_array,
            label=f"Reward {player.name}",
        )
    plt.legend()
    plt.grid()
    plt.show()


def save_qtable(qtable, folder, name="qtable"):
    """
    save the qtable
    """
    np.save(os.path.join(folder, f"{name}.npy"), qtable)
    print(f"{name}.npy saved!")


def load_qtable(folder, name="qtable"):
    """
    load the qtable
    """
    try:
        qtable = np.load(os.path.join(folder, f"{name}.npy"))
        print(f"{name}.npy loaded!")
    except:
        print(f"qtable '{name}' could not be loaded!")
    return qtable
