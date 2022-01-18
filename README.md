# Q-Learning algorithm that learns to play TicTacToe through self-play


# Q-Learning Algorithm
The image below describes the Q-Learning Algorithm, which is an oﬀ-policy Temporal-Difference control algorithm:

<!---
![Q-Learning](/Sutton_Barto.png)
-->
<img src="/images/Sutton_Barto.png" alt="TicTacToe Environment" width="600"/>

[Source](http://incompleteideas.net/book/the-book-2nd.html): **Image taken from Richard S. Sutton and Andrew G. Barto, Reinforcement Learning: An Introduction, Second edition, 2014/2015, page 158**

## Legend
- Q: action-value function
- s: state
- s': next state
- a: action
- r: reward
- alpha: learning rate
- gamma: discount factor

## Learning Parameters
- learning_rate = 1.0
- gamma = 0.9

## Exploration-Exploitation Parameters
- epsilon = 1.
- max_epsilon = 1.
- min_epsilon = 0.0
- decay_rate = 0.000001


In the training loop the agent learns to play TicTacToe through self-play. After each step taken the Q-table of the agent is updated based on the received reward.

## Training Settings
- episodes: number of games played by the players
- max_steps: number of maximal steps per game


## Training Helper Functions
- **create_state_dictionary**: create state encoding dictionary
- **reshape_state**: reshape the state array into a tuple
- **create_plot**: plot the training progress reward versus episodes for both player
- **save_qtable**: save the Q-table
- **load_qtable**: load the Q-table
- **test_self_play_learning**: test the trained agent with playing against it
