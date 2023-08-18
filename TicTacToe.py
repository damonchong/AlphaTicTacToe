"""
    AlphaTicTacToe: Inspired by DeepMind's AlphaGo
    Copyright (C) 2023, Damon Chong

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <https://www.gnu.org/licenses/>.
"""
from ADQN import TANH_ACTIVATION
from DQCNN import DQCNN
from DQDNN import DQDNN
from GameContext import gen_train_data, tuple_states_to_int_array, return_train_data_list
from GamePath import GamePath, symmetric_rotations
from GameState import INITIAL_GAME_STATE
from random import choice

import ADQN
import Utilities
import numpy as np
import time

# Below numbers are estimated simulations to run to achieve optimal play with different approaches. If using MCTS alone,
# 235000 simulations is needed and also achieves a NN that plays optimally. If combining MCTS with play-outs then a
# lower number is needed i.e 70000. All figures are just approximations as expected when dealing with randomness.
OPTIMAL_MCTS_COUNT = 235000
OPTIMAL_MCTS_WITH_PLAYOUT_COUNT = 70000

TOTAL_SIMULATIONS = OPTIMAL_MCTS_WITH_PLAYOUT_COUNT
# Both parameters sets the explore and exploit depths during play-out mode.
EXPLORE_DEPTHS = [1000, 1000, 500, 200, 200, 50, 30, 4, 2, 1]
EXPLOIT_DEPTHS = [1000, 1000, 500, 200, 200, 50, 30, 4, 2, 1]
# If set to False, no further updates will be made to the NN after initial training. Strongly advised to set to True
# especially if TOTAL_SIMULATIONS < OPTIMAL_MCTS_COUNT so that the data generated from MCTS game mode can train and
# enhance the NN performance.
INCREMENTAL_NN = True
# 2 options, either use dense NN or convolutional NN.
_agent = DQCNN()  # DQDNN()

# For TANH activation
def create_nn_check_for_tanh():
  c_list = list()
  tup = ([0, 0, 0, 0, 1, 0, 0, 0, 0], [1, 3, 5, 7])
  c_list.append(tup)
  tup = ([0, 0, 1, 0, 0, 0, 0, 0, 0], [0, 1, 3, 5, 6, 7, 8])
  c_list.append(tup)
  # tup = ([0, 0, 0, 0, 0, 0, 1, 0, 0], [0, 1, 2, 3, 5, 7, 8])
  # c_list.append(tup)
  tup = ([0, 0, 0, 0, 0, 0, 0, 0, 1], [0, 1, 2, 3, 5, 6, 7])
  c_list.append(tup)
  # tup = ([1, 0, 0, 0, 0, 0, 0, 0, 0], [1, 2, 3, 5, 6, 7, 8])
  # c_list.append(tup)
  tup = ([0, 1, 0, 0, -1, 0, 0, 0, 1], [6, 7])
  c_list.append(tup)
  tup = ([0, 0, 0, 0, -1, 1, 1, 0, 0], [0, 3])
  c_list.append(tup)
  tup = ([1, 0, 0, 0, -1, 0, 0, 1, 0], [1, 2])
  c_list.append(tup)
  tup = ([0, 0, 1, 1, -1, 0, 0, 0, 0], [5, 8])
  c_list.append(tup)
  tup = ([1, 0, 0, 0, -1, 0, 0, 0, 1], [2, 6])
  c_list.append(tup)
  tup = ([0, 0, 1, 0, -1, 0, 1, 0, 0], [0, 8])
  c_list.append(tup)
  # tup = ([0, 1, 0, 0, -1, 0, 1, 0, 0], [7, 8])
  # c_list.append(tup)
  tup = ([-1, 0, 0, 0, 1, 0, 0, 0, 1], [1, 3, 5, 7])
  c_list.append(tup)
  tup = ([0, 0, -1, 0, 1, 0, 1, 0, 0], [1, 3, 5, 7])
  c_list.append(tup)
  # tup = ([1, 0, 0, 0, 1, 0, 0, 0, -1], [1, 3, 5, 7])
  # c_list.append(tup)
  # tup = ([0, 0, 1, 0, 1, 0, -1, 0, 0], [1, 3, 5, 7])
  # c_list.append(tup)
  # tup = ([0, 0, 0, 1, -1, 0, 0, 1, 0], [1, 2, 5])
  # c_list.append(tup)
  # tup = ([0, 1, 0, 1, -1, 0, 0, 0, 0], [5, 7, 8])
  # c_list.append(tup)
  # tup = ([0, 1, 0, 0, -1, 1, 0, 0, 0], [3, 6, 7])
  # c_list.append(tup)
  # tup = ([0, 0, 0, 0, -1, 1, 0, 1, 0], [0, 1, 3])
  # c_list.append(tup)
  # tup = ([0, 0, 0, 1, -1, 0, 0, 0, 1], [2, 5])
  # c_list.append(tup)
  return c_list


# For RELU activation
def create_nn_check_for_relu_and_sigmoid():
  c_list = list()
  tup = ([0, 0, 0, 0, 1, 0, 0, 0, 0], [1, 3, 5, 7])
  c_list.append(tup)
  tup = ([0, 0, 1, 0, 0, 0, 0, 0, 0], [0, 1, 3, 5, 6, 7, 8])
  c_list.append(tup)
  # tup = ([0, 0, 0, 0, 0, 0, 1, 0, 0], [0, 1, 2, 3, 5, 7, 8])
  # c_list.append(tup)
  tup = ([0, 0, 0, 0, 0, 0, 0, 0, 1], [0, 1, 2, 3, 5, 6, 7])
  c_list.append(tup)
  # tup = ([1, 0, 0, 0, 0, 0, 0, 0, 0], [1, 2, 3, 5, 6, 7, 8])
  # c_list.append(tup)
  tup = ([0, 1, 0, 0, 2, 0, 0, 0, 1], [6, 7])
  c_list.append(tup)
  tup = ([0, 0, 0, 0, 2, 1, 1, 0, 0], [0, 3])
  c_list.append(tup)
  tup = ([1, 0, 0, 0, 2, 0, 0, 1, 0], [1, 2])
  c_list.append(tup)
  tup = ([0, 0, 1, 1, 2, 0, 0, 0, 0], [5, 8])
  c_list.append(tup)
  tup = ([1, 0, 0, 0, 2, 0, 0, 0, 1], [2, 6])
  c_list.append(tup)
  tup = ([0, 0, 1, 0, 2, 0, 1, 0, 0], [0, 8])
  c_list.append(tup)
  # tup = ([0, 1, 0, 0, 2, 0, 1, 0, 0], [7, 8])
  # c_list.append(tup)
  tup = ([2, 0, 0, 0, 1, 0, 0, 0, 1], [1, 3, 5, 7])
  c_list.append(tup)
  tup = ([0, 0, 2, 0, 1, 0, 1, 0, 0], [1, 3, 5, 7])
  c_list.append(tup)
  # tup = ([1, 0, 0, 0, 1, 0, 0, 0, 2], [1, 3, 5, 7])
  # c_list.append(tup)
  # tup = ([0, 0, 1, 0, 1, 0, 2, 0, 0], [1, 3, 5, 7])
  # c_list.append(tup)
  # tup = ([0, 0, 0, 1, 2, 0, 0, 1, 0], [1, 2, 5])
  # c_list.append(tup)
  # tup = ([0, 1, 0, 1, 2, 0, 0, 0, 0], [5, 7, 8])
  # c_list.append(tup)
  # tup = ([0, 1, 0, 0, 2, 1, 0, 0, 0], [3, 6, 7])
  # c_list.append(tup)
  # tup = ([0, 0, 0, 0, 2, 1, 0, 1, 0], [0, 1, 3])
  # c_list.append(tup)
  # tup = ([0, 0, 0, 1, 2, 0, 0, 0, 1], [2, 5])
  # c_list.append(tup)
  return c_list


# One of the problem with training NN is determining when the amount of training is just right i.e avoid over-training
# or under-training. We are using this check list as a proxy to determine when we can stop training. Noticed that too
# much checks caused problem for the NN during training sometimes i.e. it runs indefinitely. Perhaps, the checks are
# made redundant by optimization within the NN.
_check_list = create_nn_check_for_tanh() if ADQN.ACTIVATION_MODE == TANH_ACTIVATION else \
              create_nn_check_for_relu_and_sigmoid()


def _dqn_move(game_board):
  states = game_board.states
  curr_arr = tuple_states_to_int_array(states)

  rfc_state = np.asarray(curr_arr).reshape(_agent.input_shape)
  prediction = _agent.model.predict(rfc_state)
  idx = prediction[0, :].argsort()[::-1]

  for next_move in idx:
    if game_board.parent is None or game_board.states[next_move] is None:
      break
  else:
    next_move = None

  if next_move is None:
    print("No valid move found with DQN! Randomly picking a move.")
    # Final option, do a random pick
    empty_spots = [i for i, value in enumerate(game_board.states) if value is None]
    next_move = choice(empty_spots)

  return next_move + 1


def _mcts_move(game_board, path):
  if game_board.terminal:
    raise RuntimeError(f"method called on terminal node: {game_board.state_path}")

  if game_board == INITIAL_GAME_STATE:
    status = 0
  elif game_board.traversals == 0:
    status = -1
    if game_board.state_path:
      print(f"Current path: {game_board.state_path} had not been explored.")
  else:
    wild_child = 0
    for child in game_board.children:
      if not child.fully_explored:
        wild_child += 1

    if wild_child > 0:
      # As long as there are 1 or more subsequent routes not explored in MCTS, the probabilities of a good move is
      # affected.
      status = -2
      print(f"{wild_child} of the child path(s) for path: {game_board.state_path} had not been explored.")
    else:
      status = 0

  if status >= 0:
    next_board = game_board.choose()
    return next_board.tup_state[1]
  else:
    # Oops, could not ascertain any good move, simulate via play outs then.
    path.play_out(EXPLORE_DEPTHS[len(path.game_path)], EXPLOIT_DEPTHS[len(path.game_path)])
    next_board = game_board.choose()
    return next_board.tup_state[1]


def gen_single_train_data(state_list):
  winning_lists = [state_list]
  winning_lists.extend(_rotational_propagate(state_list, 1))

  train_list = list()
  for path_list in winning_lists:
    if path_list[0] != INITIAL_GAME_STATE:
      raise ValueError("Invalid state!")

    train_list.extend(return_train_data_list(path_list))

  for tup in train_list:
    adhoc_incremental_NN_train(*tup)

  _agent.sync_models()


def adhoc_incremental_NN_train(init_arr, next_move, reward, next_arr, terminal):

  # This approach is iffy especially with low simulation count. Sometimes, we have to go back to the MCTS game mode
  # and play a number of times to improve the learning. In general, higher simulation count equal and above the
  # OPTIMAL_MCTS_WITH_PLAYOUT_COUNT tends to produce better results.
  tup = (init_arr, next_move, reward, next_arr, terminal)
  for _ in range(3):
    _agent.remember(tup)
    _agent.train_short_memory(init_arr, next_move, reward, next_arr, terminal)


def _manual_move(game_board):
  try:
    while True:
      move_choice = input("Your move (enter a number from 1 to 9 or 0 to abort): ")
      next_move = int(move_choice)
      if 0 == next_move:
        return -1  # aborting

      if 0 < next_move < 10:
        return_move = player_move(game_board, next_move)
        if return_move is None:
          print(f"Illegal move: {next_move}! Please try again.")
        else:
          return return_move
      else:
        print(f"Illegal move: {move_choice}! Please try again.")
        continue

  except (ValueError, TypeError) as err:  # Handle exception
    print(err)


class Player:
  def __init__(self, flag, auto, option):
    self.flag = flag
    self.auto = auto
    self.option = option

  def move(self, game_board, path):
    if self.auto:
      if self.option == 1:
        return _mcts_move(game_board, path)
      else:
        return _dqn_move(game_board)
    else:
      return _manual_move(game_board)


def simulate():
  notify_once = False

  print(f"Start of simulation: {time.perf_counter()}")
  for i in range(1, TOTAL_SIMULATIONS + 1):
    path = GamePath()
    path.add(INITIAL_GAME_STATE)

    # Simulate by exploring and exploiting game paths
    reward = path.simulate()
    path.back_propagate(reward)
    if reward != 0.5:
      _rotational_propagate(path.game_path, reward)

    if not notify_once and INITIAL_GAME_STATE.fully_explored:
      print(f"All states fully explored....{i} simulations ran!")
      notify_once = True

  train_dqn()
  print(f"End of simulation: {time.perf_counter()}")
  print(f"Completed all {i} simulations!")


def _rotational_propagate(state_list, reward):
  rotated_paths = symmetric_rotations(state_list)
  for path_list in rotated_paths:
    path = GamePath()
    for rot_state in path_list:
      path.add(rot_state)

    if not rot_state.terminal:
      raise RuntimeError(f"Expecting a terminal last state in the rotated path!")

    path.back_propagate(reward)

  return rotated_paths


def train_dqn(reset_data=True):
  if reset_data:
    _agent.clear_train_data()
    prep_data()

  _agent.replay(True)
  _agent.replay(True)

  if INCREMENTAL_NN:
    _agent.replay(True, _check_list)
  else:
    cnt = 2
    while not _agent.replay(True, _check_list):
      cnt += 1
      print(f"Didn't pass NN check....number of replay: {cnt}")
      if cnt > 5:
        print(f"{cnt} times of replay....failed!")
        break


def prep_data():
  _train_list = gen_train_data(INITIAL_GAME_STATE)

  for init_arr, next_move, reward, next_arr, terminal in _train_list:
    _agent.remember((init_arr, next_move, reward, next_arr, terminal))


def is_valid(gc):
  valid_options = ['0', '1', '2', '9']
  if gc in valid_options:
    return True
  else:
    print(f"Unrecognized option: {gc}!")
    return False


def print_new_tictactoe():
  print(" 1 | 2 | 3 ")
  print(u'\u2500'u'\u2500'u'\u2500'u'\u2500'u'\u2500'u'\u2500'u'\u2500'u'\u2500'u'\u2500'u'\u2500'u'\u2500'u'\u2500')
  print(" 4 | 5 | 6 ")
  print(u'\u2500'u'\u2500'u'\u2500'u'\u2500'u'\u2500'u'\u2500'u'\u2500'u'\u2500'u'\u2500'u'\u2500'u'\u2500'u'\u2500')
  print(" 7 | 8 | 9 ")
  print("Please made your move, select 1 to 9 to occupy the positions.")
  print("")
  print("   |   |   ")
  print(u'\u2500'u'\u2500'u'\u2500'u'\u2500'u'\u2500'u'\u2500'u'\u2500'u'\u2500'u'\u2500'u'\u2500'u'\u2500'u'\u2500')
  print("   |   |   ")
  print(u'\u2500'u'\u2500'u'\u2500'u'\u2500'u'\u2500'u'\u2500'u'\u2500'u'\u2500'u'\u2500'u'\u2500'u'\u2500'u'\u2500')
  print("   |   |   ")


def player_move(game_board, next_move):
  if game_board.parent is None:  # Initial game state
    return next_move
  elif game_board.states[next_move - 1] is None:
    return next_move
  else:
    return None


def display_flag(flag):
  if flag is None:
    return " "
  if flag:
    return "O"
  else:
    return "X"


def display_board(states):
  print("           ")
  print(f" {display_flag(states[0])} | {display_flag(states[1])} | {display_flag(states[2])} ")
  print(u'\u2500'u'\u2500'u'\u2500'u'\u2500'u'\u2500'u'\u2500'u'\u2500'u'\u2500'u'\u2500'u'\u2500'u'\u2500'u'\u2500')
  print(f" {display_flag(states[3])} | {display_flag(states[4])} | {display_flag(states[5])} ")
  print(u'\u2500'u'\u2500'u'\u2500'u'\u2500'u'\u2500'u'\u2500'u'\u2500'u'\u2500'u'\u2500'u'\u2500'u'\u2500'u'\u2500')
  print(f" {display_flag(states[6])} | {display_flag(states[7])} | {display_flag(states[8])} ")


def update_game_move(game_board, next_move, path, side2move, first_to_move, option):
  next_board = game_board.choose(next_move)
  if next_board is None:
    print(f"Illegal move, please try again!")
    side2move = False if side2move else True
    return game_board, path, side2move, first_to_move

  if not next_board.terminal:
    next_board.expand()
  path.add(next_board)
  display_board(next_board.states)

  if next_board.terminal:
    if next_board.winner is not None:
      if len(path.game_path) % 2 == 0:
        print("O has won and X loses!")
      else:
        print("O has lost and X wins!")
    else:
      print("It's a draw!")

    if INCREMENTAL_NN:
      # To train the NN to avoid same mistake
      if option == 1:
        gen_single_train_data(path.game_path)
      # else it's NN that made the mistake. Since, we are using MCTS to train the NN, we can't update unless
      # it is originating from MCTS. On a side note, AlphaGo Zero appears to be able to use pure RL to train itself,
      # that's cool.

    next_board = INITIAL_GAME_STATE
    path = GamePath()
    path.add(next_board)
    print("")
    print_new_tictactoe()
    first_to_move = not first_to_move
    side2move = not first_to_move

  return next_board, path, side2move, first_to_move


def interactive_play(option):
  game_board = INITIAL_GAME_STATE
  game_board.expand()
  human_player = Player(True, False, option)
  comp_player = Player(False, True, option)
  print_new_tictactoe()
  path = GamePath()
  path.add(game_board)
  side2move = first_to_move = True

  while True:
    if human_player.flag == side2move:
      next_player = human_player
    else:
      next_player = comp_player

    next_move = next_player.move(game_board, path)

    if -1 == next_move:
      print("Game aborted by player!")
      print("")
      break

    game_board, path, side2move, first_to_move = update_game_move(game_board, next_move, path, side2move,
                                                                  first_to_move, option)
    side2move = not side2move


if __name__ == "__main__":
  simulate()
  print("Welcome to a game of Tic Tac Toe.")
  game_option = None
  user_choice = 1

  """
  You can choose to play in either game modes: MCTS game mode (select 1) or NN game mode (select 2). Optionally, you can 
  re-train the NN (or an experience replay) if the program had been configured for incremental NN updates. This may help 
  to improve the NN's performance.
  """
  while user_choice:
    while Utilities.is_blank(game_option) or not is_valid(game_option):
      if INCREMENTAL_NN:
        game_option = input("Enter 0 to exit, 1 to play against MCTS, 2 to play against DQN or 9 to re-train the NN: ")
      else:
        game_option = input("Enter 0 to exit, 1 to play against MCTS or 2 to play against DQN: ")

    user_choice = int(game_option)
    game_option = None
    print(f"Your selected choice: {user_choice}")

    if user_choice == 0:
      print("Ending program, good-bye!")
    elif user_choice == 9:
      train_dqn(False)
    else:
      interactive_play(user_choice)


