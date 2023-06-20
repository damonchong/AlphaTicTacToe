# AlphaTicTacToe
Learning Reinforcement Learning (RL) with Monte Carlo Tree Search (MCTS) via TicTacToe

## What is it?
-----------
AlphaTicTacToe is my attempt at RL and understanding MCTS. 

## Why AlphaTicTacToe?
-----------
A few years ago, I was inspired by AlphaGo and wanted to learn more about RL and deep learning in general. In particular, I was motivated by how RL can be combined with MCTS as well as how play-outs can help improve RL/MCTS. I tried searching for working code on MCTS, RL as well as TicTacToe. The aim was to learn RL/MCTS via a simple game like TicTacToe. There were pieces of Python code here and there but I couldn't really find any good working examples. I decided back then to try to get something working. Lately, with some free time, I decided to put this online for sharing.


## How do you start the game?
You will need to install Python v3.7 (using v3.7.6) with Tensorflow v1.15 (using v1.15.2) and Keras v2.2.4. There maybe other related packages that needs to be installed as well. Simply run python TicTacToe.py to get the console application fired up. When you see the console prompt as below, play in either game modes: MCTS game mode (select 1) or NN game mode (select 2). Optionally, you can re-train the NN (or an experience replay) if the program had been configured for incremental NN updates. This may help to improve the NN's performance.

```
Enter 0 to exit, 1 to play against MCTS, 2 to play against DQN or 9 to re-train the NN:
```


## In-depth settings and notes
To see RL in action, set the number of simulations lower in the TOTAL_SIMULATIONS variable in the file TicTacToe.py e.g. 50000 and ensure that the variable INCREMENTAL_NN is set as True. At the game prompt, 1st try playing in NN game mode and notice how badly it played. Exit (enter 0) and go into MCTS game mode (select 1) and play the same moves that you did previously. MCTS should be able to achieve a draw if it played optimally. Once, it does, exit and go back to NN game mode again. If RL works, the NN should now play optimally against the same moves from you.

## Who to approach?
This was a learning experience so if you have any issues, queries or comments regarding my attempt, you can drop me an email at c_h_o_n_g_d_a_m_o_n@g-m-a-i-l.c0m (remove the underscores and figure out the domain) although no guarantee I will respond fast.