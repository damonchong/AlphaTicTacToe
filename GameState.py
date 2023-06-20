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
from abc import ABC, abstractmethod
from collections import namedtuple, OrderedDict
from random import choice

import GameContext


def _winning_combos():
  for start in range(0, 9, 3):  # three in a row
    yield (start, start + 1, start + 2)
  for start in range(3):  # three in a column
    yield (start, start + 3, start + 6)
  yield (0, 4, 8)  # down-right diagonal
  yield (2, 4, 6)  # down-left diagonal


def _find_winner(states):
  # Returns None if no winner, True or False if there's a winner
  for i1, i2, i3 in _winning_combos():
    v1, v2, v3 = states[i1], states[i2], states[i3]
    if False is v1 is v2 is v3:
      return False
    if True is v1 is v2 is v3:
      return True
  return None


def _win_lose_or_draw(gs):
  state_path = str(gs.tup_state[0]) + '-' + str(gs.tup_state[1])
  states = [None, ] * 9
  if gs.tup_state[0] != 0:
    states[gs.tup_state[1] - 1] = gs.get_flag()

  ps = gs
  while ps.parent is not None:
    ps = ps.parent
    state_path = str(ps.tup_state[0]) + '-' + str(ps.tup_state[1]) + ':' + state_path
    if ps.tup_state[1] != 0:
      states[ps.tup_state[1] - 1] = ps.get_flag()

  gs.state_path = state_path
  gs.states = tuple(states)
  terminal = not any(s is None for s in states)
  wld = _find_winner(states)
  if wld is None:
    if terminal:
      gs.terminal = True
  else:
    gs.winner = wld
    gs.terminal = True


class State(ABC):

  @abstractmethod
  def get_flag(self):
    raise NotImplementedError

  @abstractmethod
  def add_child(self, gs):
    raise NotImplementedError

  @abstractmethod
  def expand(self):
    raise NotImplementedError

  @abstractmethod
  def select(self):
    raise NotImplementedError

  @abstractmethod
  def reward(self):
    raise NotImplementedError

  @abstractmethod
  def choose(self, move=-1):
    raise NotImplementedError

  @abstractmethod
  def latest_score(self):
    raise NotImplementedError


_newState = namedtuple("GameState", "tup_state, parent")


class GameState(_newState, State):
  """
  The tup_state represents a tuple consisting of 2 numbers. The first number represents the
  current game step i.e. either 0, 1, 2....till 9. 0 represents the initial game state i.e.
  no move made and 1 stands for 1st move. The second number represents the square being occupied.
  So tuple (1, 5) means first move O occupied the center square of the Tic Tac Toe board.
  Implicitly, O always goes first followed by X so even moves means X while odd moves means O.
  """
  def __init__(self, tup_state, parent):
    super().__init__()
    if len(tup_state) != 2:
      raise ValueError("Expecting a tuple containing 2 numbers!")
    # Check the current step in the game.
    if tup_state[0] is not None:
      if not (-1 < tup_state[0] < 10):
        raise ValueError(f"Expecting 1st number to range from 0 to 9, instead found: {tup_state[0]}!")
    else:
      raise ValueError("Not expecting a None value!")
    # Check the square occupied.
    if tup_state[1] is not None:
      if not (-1 < tup_state[1] < 10):
        raise ValueError(f"Expecting 2nd number to range from 0 to 9, instead found: {tup_state[1]}!")
    else:
      raise ValueError("Not expecting a None value!")

    if parent is not None:
      self._check_parent(parent)  # There will always be only 1 parent.
    self.wins = 0
    self.traversals = 0
    self.children = set()
    # Next 3 class members to be populated by _win_lose_or_draw method
    self.winner = None
    self.terminal = False
    self.state_path = None  # This is a snapshot of the game state till now.

    _win_lose_or_draw(self)  # This sets the winner, terminal, path and states of the game state

    # This is true when there are no possible child states or all child states are fully explored
    self.fully_explored = False

  def __new__(cls, tup_state, parent):
    self = super(GameState, cls).__new__(cls, tup_state, parent)
    return self

  def get_flag(self):
    if self.tup_state[0] == 0:
      return None

    # For ease of displaying, odd steps equals True and even treated as False.
    return self.tup_state[0] % 2 != 0

  def _check_parent(self, gs):
    parent_state = gs.tup_state
    # Check if move index of parent is valid
    if parent_state[0] != 0:
      if parent_state[0] == self.tup_state[0] - 1:
        # Check if parent's position does not overlap with own position
        if parent_state[1] == self.tup_state[1]:
          raise ValueError(f"Parent to add has same position: {parent_state[1]} as own self!")
      else:
        raise ValueError(f"Parent to add has an invalid move index: {parent_state[0]} vs own index: {self.tup_state[0]}")

  def add_child(self, gs):
    child_state = gs.tup_state
    if self.tup_state[0] != 0:
      # Check if move index of child is valid
      if child_state[0] == self.tup_state[0] + 1:
        # Check if parent's position does not overlap with own position
        if child_state[1] != self.tup_state[1]:
          self.children.add(gs)
        else:
          raise ValueError(f"Child to add has same position: {child_state[1]} as own self!")
      else:
        raise ValueError(f"Child to add has an invalid move index: {child_state[0]} vs own index: {self.tup_state[0]}")
    else:
      self.children.add(gs)

  def expand(self):
    if self.children:
      return

    if self.terminal:
      raise RuntimeError(f"Expand called on terminal state!")

    self.children = self._find_paths()

  def _find_paths(self):
    if not any(s is None for s in self.states):
      return set()

    return {
      self._make_state(idx + 1) for idx, state in enumerate(self.states) if state is None
    }

  def _make_state(self, idx):
    tup_state = (self.tup_state[0] + 1, idx)
    gs = GameState(tup_state, self)
    if gs.state_path in GameContext.AllPath2State:
      return GameContext.AllPath2State[gs.state_path]
    else:
      GameContext.AllPath2State[gs.state_path] = gs

    return gs

  def choose(self, move=-1):
    if move == -1:
      if self.terminal:
        raise RuntimeError(f"choose called on terminal state: {self.state_path}!")

      if not self.children:
        raise RuntimeError(f"choose called on unexplored state: {self.state_path}!")

      def score(s):
        if s.traversals == 0:
          return float("-inf")  # avoid unexplored moves

        return s.wins / s.traversals

      win_list = [child for child in self.children if child.winner is not None]
      if win_list:
        return max(win_list, key=score)
      else:
        return max(self.children, key=score)
    else:
      return next((child for child in self.children if child.tup_state[1] == move), None)

  def select(self):
    if not self.children:
      if not self.terminal:
        raise RuntimeError(f"Invalid select operation, state's children not initialized!")
      else:
        return None

    if len(self.children) == 1:
      return next(iter(self.children))

    if all(c.fully_explored for c in self.children):
      return None
    else:
      return self._select_least_visited()

  def _select_least_visited(self):
    if self.tup_state[0] == 0:
      return choice(list(self.children))

    traversals2list = dict()
    for cs in self.children:
      if not cs.fully_explored:
        if cs.traversals in traversals2list:
          child_list = traversals2list[cs.traversals]
        else:
          child_list = list()
          traversals2list[cs.traversals] = child_list

        child_list.append(cs)

    if traversals2list:
      od = OrderedDict(sorted(traversals2list.items()))
      child_list = next(iter(od.items()))[1]
      return choice(child_list)

    return None

  def reward(self):
    if not self.terminal:
      raise RuntimeError(f"reward called on non-terminal board: {self.state_path}")

    if self.winner is None:
      return 0.5
    else:
      return 1.0  # Either way you win so rewarded.

  def latest_score(self):
    if self.traversals == 0:
      return 0

    return self.wins / self.traversals


def _initial_state():
  init_state = GameState((0, 0), None)
  _gs = GameState((1, 1), init_state)
  GameContext.AllPath2State[_gs.state_path] = _gs
  init_state.add_child(_gs)
  _gs = GameState((1, 2), init_state)
  GameContext.AllPath2State[_gs.state_path] = _gs
  init_state.add_child(_gs)
  _gs = GameState((1, 3), init_state)
  GameContext.AllPath2State[_gs.state_path] = _gs
  init_state.add_child(_gs)
  _gs = GameState((1, 4), init_state)
  GameContext.AllPath2State[_gs.state_path] = _gs
  init_state.add_child(_gs)
  _gs = GameState((1, 5), init_state)
  GameContext.AllPath2State[_gs.state_path] = _gs
  init_state.add_child(_gs)
  _gs = GameState((1, 6), init_state)
  GameContext.AllPath2State[_gs.state_path] = _gs
  init_state.add_child(_gs)
  _gs = GameState((1, 7), init_state)
  GameContext.AllPath2State[_gs.state_path] = _gs
  init_state.add_child(_gs)
  _gs = GameState((1, 8), init_state)
  GameContext.AllPath2State[_gs.state_path] = _gs
  init_state.add_child(_gs)
  _gs = GameState((1, 9), init_state)
  GameContext.AllPath2State[_gs.state_path] = _gs
  init_state.add_child(_gs)
  return init_state


INITIAL_GAME_STATE = _initial_state()
INITIAL_POSITIONS = list(INITIAL_GAME_STATE.children)