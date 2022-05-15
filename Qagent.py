import random

import numpy as np


class Qagent(object):
    """
    Implementation of a Q-learning Algorithm
    """

    def __init__(
        self,
        state_size: int,
        action_size: int,
        learning_parameters: dict,
        exploration_parameters: dict,
    ):
        """
        initialize the q-learning agent
        Args:
          state_size (int): ..
          action_size (int): ..
          learning_parameters (dict):
          exploration_parameters (dict):

        """
        self.qtable = np.zeros((state_size, action_size))

        self.learning_rate = learning_parameters["learning_rate"]
        self.gamma = learning_parameters["gamma"]

        self.epsilon = exploration_parameters["max_epsilon"]
        self.max_epsilon = exploration_parameters["max_epsilon"]
        self.min_epsilon = exploration_parameters["min_epsilon"]
        self.decay_rate = exploration_parameters["decay_rate"]

    def update_qtable(
        self, state: int, new_state: int, action: int, reward: int, done: bool
    ) -> np.array:
        """
        update the q-table: Q(s,a) = Q(s,a) + lr  * [R(s,a) + gamma * max Q(s',a') - Q (s,a)]

        Args:
          state (int): current state of the environment
          new_state (int): new state of the environment
          action (int): current action taken by agent
          reward (int): current reward received from env
          done (boolean): variable indicating if env is done

        Returns:
          qtable (array): the qtable containing a value for every state (y-axis) and action (x-axis)
        """
        return self.qtable[state, action] + self.learning_rate * (
            reward
            + self.gamma * np.max(self.qtable[new_state, :]) * (1 - done)
            - self.qtable[state, action]
        )

    def update_epsilon(self, episode: int, mode: str = "exponential") -> None:
        """reduce epsilon, exponential decay

        Args:
            episode (int): _description_
            mode (str, optional): _description_. Defaults to "exponential_decay".
        """
        if mode == "exponential":
            self.epsilon = self.min_epsilon + (
                self.max_epsilon - self.min_epsilon
            ) * np.exp(-self.decay_rate * episode)

        elif mode == "linear":
            pass

            # self.epsilon = self.min_epsilon + (self.max_epsilon - self.min_epsilon) * (
            #     -self.decay_rate * episode
            # )

    def get_action(self, state: int, action_space: np.array) -> int:
        """select action e-greedy
        get rank of max value (min rank) from the action_space

        Args:
          state (int): current state of the environment/agent
          action_space (array): array with legal actions

        Returns:
          action (int): action that the agent will take in the next step
        """
        if random.uniform(0, 1) >= self.epsilon:
            # exploitation, max value for given state
            ranks = self.qtable[state, :].argsort().argsort()
            action = np.where(ranks == np.min(ranks[action_space]))[0][0]

        else:
            # exploration, random choice
            action = np.random.choice(action_space)
        return action
