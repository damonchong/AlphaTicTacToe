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
from ADQNN import RELU_ACTIVATION, SIGMOID_ACTIVATION, TANH_ACTIVATION
from typing import Any, Callable

import ADQNN

# Key: Path string. Value: GameState instance.
AllPath2State = dict()


def tuple_states_to_int_array(tup):
  if ADQNN.ACTIVATION_MODE == TANH_ACTIVATION:
    bool2int: Callable[[Any], int] = lambda f: (1 if f is True else (-1 if f is False else 0))
  elif ADQNN.ACTIVATION_MODE == RELU_ACTIVATION:
    bool2int: Callable[[Any], int] = lambda f: (1 if f is True else (2 if f is False else 0))
  elif ADQNN.ACTIVATION_MODE == SIGMOID_ACTIVATION:
    bool2int: Callable[[Any], int] = lambda f: (1 if f is True else (2 if f is False else 0))
  else:
    raise ValueError(f"Unsupported activation: {ADQNN.ACTIVATION_MODE}!")
  arr = [bool2int(tup[idx]) for idx in range(9)]
  return arr


def convert_to_tuple(p_ts, c_state, reward):
  init_arr = tuple_states_to_int_array(p_ts)
  next_arr = tuple_states_to_int_array(c_state.states)
  return init_arr, c_state.tup_state[1] - 1, reward, next_arr, c_state.terminal


def gen_train_data(init_state):
  train_list = list()
  children = [init_state]
  sequence = list()

  while children:
    parent = children.pop(0)
    if parent.terminal or not parent.children:
      continue

    sequence.append(parent)
    fav_child = parent.choose()
    sequence.append(fav_child)
    for child in parent.children:
      if child != fav_child:
        children.append(child)

    while not fav_child.terminal and fav_child.children:
      fav_child = fav_child.choose()
      sequence.append(fav_child)

    train_list.extend(return_train_data_list(sequence))
    sequence.clear()

  return train_list


def return_train_data_list(sequence):
  train_list = list()

  size = len(sequence) - 1
  winner = sequence[-1].winner

  while size > 0:
    c_state = sequence[size]
    p_state = c_state.parent
    p_ts = p_state.states

    train_list.append(convert_to_tuple(p_ts, c_state, 1))

    if winner is not None:
      size -= 2
    else:
      size -= 1

  return train_list

def return_train_data_list_v2(sequence):
  train_list = list()

  winner = sequence[-1].winner
  if winner is not None:
    reward = 1
  else:
    reward = 0

  size = len(sequence) - 1

  while size > 0:
    c_state = sequence[size]
    p_state = c_state.parent
    p_ts = p_state.states

    train_list.append(convert_to_tuple(p_ts, c_state, reward))
    if reward != 0:
      reward *= -1

    size -= 1

  return train_list
