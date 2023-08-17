from abc import ABC, abstractmethod
from _collections import deque
from random import shuffle

import numpy as np
import os.path


"""
TANH works best. RELU didn't produce optimal play and the complete check list to stop training didn't work and had to 
be partially commented out or the training won't stop. SIGMOID is the worst amongst the 3 activation functions tested, 
maybe vanishing gradient problem?
"""
TANH_ACTIVATION = 'tanh'
RELU_ACTIVATION = 'relu'
SIGMOID_ACTIVATION = 'sigmoid'  # Bad!

ACTIVATION_MODE = None


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


# Abstract Deep Q Neural Network
class ADQN(ABC):

  def __init__(self):
    self.input_shape = None  # to be overridden by inherited classes.
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

    # self.epsilon = 0
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

  def remember(self, tup):
    self.queue.append(tup)
    self.memory.append(tup)

    if len(self.memory) > MAX_DATA_LIMIT:
      # Using this to remove data that are generated from exploration stage or not optimally exploited.
      # The MAX_DATA_LIMIT is approximately the size that MCTS generates at optimal simulations size
      # i.e. OPTIMAL_MCTS_COUNT.
      tup_out = self.queue.popleft()
      self.memory.remove(tup_out)

  def train_short_memory(self, init_arr, next_move, reward, next_arr, done):
    # reshaped and flattened current and new state
    rfc_state = np.asarray(init_arr).reshape(self.input_shape)
    rfn_state = np.asarray(next_arr).reshape(self.input_shape)

    target = reward
    if not done:
      target = reward + self.gamma * np.amax(self.model.predict(rfn_state)[0])

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
      rfn_states = []
      fc_qs_list = []

      for index, (init_arr, next_move, reward, next_arr, done) in enumerate(each_batch):
        rfc_state = np.asarray(init_arr).reshape(self.input_shape)
        rfn_state = np.asarray(next_arr).reshape(self.input_shape)
        rfc_states.append(rfc_state[0])
        rfn_states.append(rfn_state[0])
        fc_qs_list.append(self.model.predict(rfc_state)[0])

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

  def pass_check(self, check_list):
    # check_list consists of a list of tuples. Each tuple consist of 2 elements, 2 int array. 1st int array represents
    # the state and the 2nd represents unacceptable list of moves.
    for tup in check_list:
      rfc_state = np.asarray(tup[0]).reshape(self.input_shape)
      prediction = self.target_model.predict(rfc_state)
      idx = prediction[0, :].argsort()[::-1]
      if idx[0] in tup[1]:
        return False

    return True

  def sync_models(self):
    self.target_model.set_weights(self.model.get_weights())

  def save_weights_into_file(self):
    self.model.save_weights(self.file_name)

  def ready(self):
    return self.is_trained

  def clear_train_data(self):
    self.queue.clear()
    self.memory.clear()

  @abstractmethod
  def network(self, weights=None, batch=0):
    pass




