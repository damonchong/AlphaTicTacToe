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


def is_blank(s):
  if s:
    return s.isspace()

  return True


def clockwise_rotate(old_pos):
  if 1 > old_pos or old_pos > 9:
    raise ValueError(f"Expecting a position between the range of 1 to 9 but found: {old_pos} instead!")

  if old_pos == 1:
    return 3
  if old_pos == 2:
    return 6
  if old_pos == 3:
    return 9
  if old_pos == 4:
    return 2
  if old_pos == 5:
    return 5
  if old_pos == 6:
    return 8
  if old_pos == 7:
    return 1
  if old_pos == 8:
    return 4
  if old_pos == 9:
    return 7


def rotational_symmetries(tup):
  tup1 = rotate_clockwise(tup)
  tup2 = rotate_clockwise(tup1)
  tup3 = rotate_clockwise(tup2)
  return tup1, tup2, tup3


def rotate_clockwise(tup):
  if not tup:
    raise ValueError(f"Empty tuple found! Expecting a tuple with 9 elements!")
  #          0        1        2        3        4        5        6        7        8
  return tup[6], tup[3], tup[0], tup[7], tup[4], tup[1], tup[8], tup[5], tup[2]


