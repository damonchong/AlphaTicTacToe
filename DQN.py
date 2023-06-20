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
from _collections import deque
from keras.layers.core import Dense, Dropout
from keras.models import Sequential
from keras.optimizers import Adam
from random import shuffle
import os.path
import numpy as np

"""
TANH works best. RELU didn't produce optimal play and the complete check list to stop training didn't work and had to 
be partially commented out or the training won't stop. SIGMOID is the worst amongst the 3 activation functions tested, 
maybe vanishing gradient problem?
"""
TANH_ACTIVATION = 'tanh'
RELU_ACTIVATION = 'relu'
SIGMOID_ACTIVATION = 'sigmoid'  # Bad!

ACTIVATION_MODE = TANH_ACTIVATION
# Using MAX_DATA_LIMIT to reduce poor quality data from training set as well as to speed up training. This limit is
# approximately 1.5 times that of an optimal NN training data set i.e. no. of simulations = OPTIMAL_MCTS_COUNT.
MAX_DATA_LIMIT = 172500
weights_file = "weights"
minibatch_size = 8
weight_reset_size = 64
min_replay_size = 384
reset_count = 1


def split_batch(iterable, n=1, reverse=False):
  sz = len(iterable)
  if not reverse:
    rng = range(0, sz, n)
  else:
    rng = range(sz, 0, -n)

  for ndx in rng:
    yield iterable[ndx:min(ndx + n, sz)]


class DQN(object):

  def __init__(self):
    self.reward = 0
    self.gamma = 0.95  # discount to reward
    self.learning_rate = 0.0005  # Incremental adjustments to gradient
    self.file_name = weights_file + ".hdf5"
    self.model = None
    self.target_model = None
    self.is_trained = None

    if os.path.isfile(self.file_name):
      self.reset_weights(self.file_name)
    else:
      self.reset_weights()

    self.epsilon = 0
    self.memory = list()
    self.queue = deque()

  def reset_weights(self, file_name=None):
    self.target_model = self.network(batch=minibatch_size)
    if file_name:
      self.model = self.network(file_name)
      self.target_model.set_weights(self.model.get_weights())
      self.is_trained = True
    else:
      self.model = self.network()
      self.is_trained = False

  def sync_models(self):
    self.target_model.set_weights(self.model.get_weights())

  def save_weights_into_file(self):
    self.model.save_weights(self.file_name)

  def network(self, weights=None, batch=0):
    model = Sequential()

    if 0 == batch:
      model.add(Dense(activation=ACTIVATION_MODE, input_dim=9, units=90))
    else:
      model.add(Dense(activation=ACTIVATION_MODE, input_dim=9, batch_size=batch, units=90))

    model.add(Dropout(rate=0.15))
    model.add(Dense(activation=ACTIVATION_MODE, units=180))
    model.add(Dropout(rate=0.15))
    model.add(Dense(activation=ACTIVATION_MODE, units=180))
    model.add(Dropout(rate=0.15))
    model.add(Dense(activation=ACTIVATION_MODE, units=180))
    model.add(Dropout(rate=0.15))
    model.add(Dense(activation=ACTIVATION_MODE, units=180))
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

  def ready(self):
    return self.is_trained

  def train_short_memory(self, init_arr, next_move, reward, next_arr, done):
    fc_state = np.asarray(init_arr)
    fn_state = np.asarray(next_arr)

    target = reward
    if not done:
      target = reward + self.gamma * np.amax(self.model.predict(fn_state.reshape((1, 9)))[0])

    # reshaped and flattened current state
    rfc_state = fc_state.reshape((1, 9))
    future_target = self.model.predict(rfc_state)

    future_target[0][next_move] = target
    self.model.fit(rfc_state, future_target, epochs=1, verbose=0)

  def replay(self, complete=False, check_list=[]):
    global reset_count, weight_reset_size
    pass_check = False
    if len(self.memory) < min_replay_size:
      print(f"No data available for training!")
      return pass_check

    shuffle(self.memory)
    print(f"Size of training data: {len(self.memory)}")
    cnt = 0
    for each_batch in split_batch(self.memory, minibatch_size):
      if len(each_batch) < minibatch_size:
        break  # Cannot fit if not right size so break. Will missed the last bit of data.

      rfc_states = []
      fc_qs_list = []
      rfn_states = []

      for index, (init_arr, next_move, reward, next_arr, done) in enumerate(each_batch):
        fc_state = np.asarray(init_arr)
        fn_state = np.asarray(next_arr)
        rfc_states.append(fc_state.reshape((1, 9))[0])
        fc_qs_list.append(self.model.predict(fc_state.reshape((1, 9)))[0])
        rfn_state = fn_state.reshape((1, 9))
        rfn_states.append(rfn_state[0])

        target = reward
        if not done:
          target = reward + self.gamma * np.amax(self.model.predict(rfn_state)[0])

        fc_qs_list[index][next_move] = target

      self.target_model.fit(np.array(rfc_states), np.array(fc_qs_list), batch_size=minibatch_size, verbose=0)
      if len(check_list) > 0:
        cnt += 1
        if self.pass_check(check_list):
          pass_check = True
          print("Neural network pass check!")
          break

    if cnt:
      print(f"Total of {cnt} checks.")
    reset_count += 1
    self.is_trained = True
    # Switch the weights of the 1 batch increment model
    if complete or reset_count > weight_reset_size or check_list:
      print('Swap weights!')
      self.model.set_weights(self.target_model.get_weights())
      reset_count = 1

    return pass_check

  def remember(self, tup):
    self.queue.append(tup)
    self.memory.append(tup)

    if len(self.memory) > MAX_DATA_LIMIT:
      # Using this to remove data that are generated from exploration stage or not optimally exploited.
      # The MAX_DATA_LIMIT is approximately the size that MCTS generates at optimal simulations size
      # i.e. OPTIMAL_MCTS_COUNT.
      tup_out = self.queue.popleft()
      self.memory.remove(tup_out)

  def pass_check(self, check_list):
    # check_list consists of a list of tuples. Each tuple consist of 2 elements, 2 int array. 1st int array represents
    # the state and the 2nd represents unacceptable list of moves.
    for tup in check_list:
      rfc_state = np.asarray(tup[0]).reshape((1, 9))
      prediction = self.target_model.predict(rfc_state)
      idx = prediction[0, :].argsort()[::-1]
      if idx[0] in tup[1]:
        return False

    return True

  def clear_train_data(self):
    self.queue.clear()
    self.memory.clear()
