# Q-Learning algorithm that learns to play TicTacToe through self-play

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
