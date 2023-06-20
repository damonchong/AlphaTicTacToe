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
from GameContext import (
  AllPath2State
)
from GameState import INITIAL_GAME_STATE, GameState
from abc import ABC, abstractmethod
from collections import namedtuple
from random import choice

import Utilities


def symmetric_rotations(game_path):
  rot_cw_90 = rotate_clockwise(game_path)
  rot_cw_180 = rotate_clockwise(rot_cw_90)
  rot_cw_270 = rotate_clockwise(rot_cw_180)
  return [rot_cw_90, rot_cw_180, rot_cw_270]


def rotate_clockwise(game_path):
  gs = game_path[0]
  path_rotated = [gs]
  for state in game_path[1:]:
    new_pos = Utilities.clockwise_rotate(state.tup_state[1])
    tup_state = (state.tup_state[0], new_pos)
    new_state = GameState(tup_state, gs)
    if new_state.state_path in AllPath2State:
      new_state = AllPath2State[new_state.state_path]
      if gs is not new_state.parent:
        raise RuntimeError(f"Parent is not the same!")
    else:
      AllPath2State[new_state.state_path] = new_state

    path_rotated.append(new_state)
    gs = new_state

  return path_rotated


class Path(ABC):

  @abstractmethod
  def add(self, gs):
    raise NotImplementedError

  @abstractmethod
  def explore(self):
    raise NotImplementedError

  @abstractmethod
  def play_out(self, explore_cycles, exploit_cycles):
    raise NotImplementedError

  @abstractmethod
  def exploit(self):
    raise NotImplementedError

  @abstractmethod
  def branch(self):
    raise NotImplementedError

  @abstractmethod
  def simulate(self):
    raise NotImplementedError

  @abstractmethod
  def back_propagate(self, reward):
    raise NotImplementedError


_newPath = namedtuple("GamePath", "")


class GamePath(_newPath, Path):
  def __init__(self):
    super().__init__()
    self.game_path = []

  def __new__(cls):
    self = super(GamePath, cls).__new__(cls)
    return self

  def add(self, gs):
    if gs.tup_state[0] != 0:
      move_index = len(self.game_path)
      if gs.tup_state[0] != move_index:
        raise ValueError(f"Invalid game state for adding! Current move index in path: {move_index}"
                         f" but move index of game state to be added: {gs.tup_state[0]}!")

    if gs.state_path in AllPath2State:
      next_state = AllPath2State[gs.state_path]
      if gs is not next_state:
        raise RuntimeError(f"Unable to find game state: {gs.state_path} in all paths!")
    else:
      AllPath2State[gs.state_path] = gs

    self.game_path.append(gs)

  def explore(self):
    if len(self.game_path) == 0:
      raise RuntimeError("Path had not been initialized!")

    lgs = self.game_path[-1]
    if lgs.terminal:
      raise RuntimeError(f"Explore called on terminal state!")

    if lgs.state_path in AllPath2State:
      next_state = AllPath2State[lgs.state_path]
      if next_state is not lgs:
        raise RuntimeError(f"State in all path and last state in game path is not the same reference!")
    else:
      AllPath2State[lgs.state_path] = lgs

    next_state = None
    lgs.expand()
    cs = lgs.select()
    if cs is not None:
      next_state = cs

    if next_state:
      self.game_path.append(next_state)
      if next_state.state_path not in AllPath2State:
        raise RuntimeError(f"The global path list does not contain the path: {next_state.state_path}!")

    return not (self.game_path[-1] is lgs)

  def exploit(self):
    if len(self.game_path) == 0:
      raise RuntimeError("Path had not been initialized!")

    lgs = self.game_path[-1]
    if lgs.terminal or not lgs.children:
      raise RuntimeError(f"Exploit called on terminal or unexplored state !")

    if lgs.state_path in AllPath2State:
      next_state = AllPath2State[lgs.state_path]
      if next_state is not lgs:
        raise RuntimeError(f"State in all path and last state in game path is not the same reference!")
    else:
      AllPath2State[lgs.state_path] = lgs

    next_state = lgs.choose()
    self.game_path.append(next_state)
    if next_state.state_path not in AllPath2State:
      raise RuntimeError(f"The global path list does not contain the path: {next_state.state_path}!")

  def play_out(self, explore_cycles, exploit_cycles):
    if len(self.game_path) == 0:
      raise RuntimeError("Path had not been initialized!")

    if explore_cycles <= 0:
      raise ValueError("Invalid explore cycles!")

    if exploit_cycles <= 0:
      raise ValueError("Invalid exploit cycles!")

    cycle_count = 0
    lgs = self.game_path[-1]
    # Spend explore_cycles in the exploration stage
    while not lgs.fully_explored:
      split_branch = self.branch()

      while not split_branch.game_path[-1].terminal:
        if not split_branch.explore():
          cycle_count += 1  # Nothing left to explore for this branch
          state = split_branch.game_path[-1]
          if not state.terminal:
            if all(c.fully_explored for c in state.children):
              self._explore_to_end(split_branch)
              break
            else:
              raise RuntimeError(f"This branch {state.state_path} should have fully explored children but did not!")
        else:
          cycle_count += 1

        if cycle_count == explore_cycles:
          break

      if cycle_count == explore_cycles:
        break

      if split_branch.game_path[-1].terminal:
        split_branch.back_propagate(split_branch.game_path[-1].reward())
      else:
        raise RuntimeError(f"Branch: {split_branch.game_path[-1].state_path} did not terminate but have nothing to explore!")

    split_branch = self.branch()
    # Spending cycles in the exploitation stage
    for c in range(exploit_cycles):
      lgs = split_branch.game_path[-1]

      if lgs.terminal:
        split_branch.back_propagate(lgs.reward())
        split_branch = self.branch()
      else:
        lgs.expand()
        split_branch.exploit()
    else:
      if split_branch.game_path[-1].terminal:
        split_branch.back_propagate(split_branch.game_path[-1].reward())

  def branch(self):
    new_branch = GamePath()
    new_branch.game_path = list(self.game_path)
    return new_branch

  def simulate(self):
    if len(self.game_path) == 0:
      raise RuntimeError("Path had not been initialized!")

    while True:
      lgs = self.game_path[-1]
      if lgs.terminal:
        return lgs.reward()

      if not self.explore():
        # Only start exploiting once exploration is completed. This is fine for problems with small search space such
        # as Tic Tac Toe but for large, complex problem space, it may require a more balanced approach.
        self.exploit()

  def back_propagate(self, reward):
    if len(self.game_path) == 0:
      raise RuntimeError("Path had not been initialized!")

    lgs = self.game_path[-1]
    if not lgs.terminal:
      raise RuntimeError("Back propagation called on unterminated last state!")

    for state in reversed(self.game_path):
      if state is not INITIAL_GAME_STATE:
        state.traversals += 1
        state.wins += reward

        if not state.fully_explored:
          if not any(s is None for s in state.states):  # Not using children as state may not be expanded
            state.fully_explored = True
          elif state.terminal:
            state.fully_explored = True
          else:
            if state.children:
              if all(c.fully_explored for c in state.children):
                state.fully_explored = True

        reward = 1 - reward
      else:
        if not state.fully_explored:
          if all(c.fully_explored for c in state.children):
            state.fully_explored = True

  def _explore_to_end(self, split_branch):
    while not split_branch.game_path[-1].terminal:
      state = split_branch.game_path[-1]
      state = choice(list(state.children))
      split_branch.add(state)


