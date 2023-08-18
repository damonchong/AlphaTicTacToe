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
from keras.layers.core import Dense, Dropout
from keras.models import Sequential
from keras.optimizers import Adam

import ADQN


def split_batch(iterable, n=1, reverse=False):
  sz = len(iterable)
  if not reverse:
    rng = range(0, sz, n)
  else:
    rng = range(sz, 0, -n)

  for ndx in rng:
    yield iterable[ndx:min(ndx + n, sz)]

# Deep Q Dense Neural Network
class DQDNN(ADQN.ADQN):

  def __init__(self):
    ADQN.ACTIVATION_MODE = TANH_ACTIVATION
    super().__init__()
    self.input_shape = (1, 9)

  def network(self, weights=None, batch=0):
    model = Sequential()

    if 0 == batch:
      model.add(Dense(activation=ADQN.ACTIVATION_MODE, input_dim=9, units=90))
    else:
      model.add(Dense(activation=ADQN.ACTIVATION_MODE, input_dim=9, batch_size=batch, units=90))

    model.add(Dropout(rate=0.15))
    model.add(Dense(activation=ADQN.ACTIVATION_MODE, units=180))
    model.add(Dropout(rate=0.15))
    model.add(Dense(activation=ADQN.ACTIVATION_MODE, units=180))
    model.add(Dropout(rate=0.15))
    model.add(Dense(activation=ADQN.ACTIVATION_MODE, units=180))
    model.add(Dropout(rate=0.15))
    model.add(Dense(activation=ADQN.ACTIVATION_MODE, units=180))
    model.add(Dropout(rate=0.15))
    if 0 == batch:
      model.add(Dense(activation='softmax', units=9))
    else:
      model.add(Dense(activation='softmax', batch_size=batch, units=9))
    opt = Adam(self.learning_rate)
    model.compile(loss='mse', optimizer=opt)

    if weights:
      model.load_weights(weights)
    return model

