import gym
import gym_TicTacToe

from src.play_tictactoe import play_tictactoe
from src.utils import create_state_dictionary, load_qtable


# initialize the tictactoe environment
env = gym.envs.make("TTT-v0", small=-1, large=10)
state_dict = create_state_dictionary()


episodes = 1_000_000
name = f"qtable_{episodes}"
folder = "tables"

qtable = load_qtable(folder, name)

play_tictactoe(env, qtable, state_dict, num_test_games=3)
