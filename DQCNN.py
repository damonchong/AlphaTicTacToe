from ADQN import TANH_ACTIVATION, RELU_ACTIVATION
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from random import shuffle

import ADQN
import keras
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

LABELS_FOR_WIN_LOSE_DRAW = set(["", "", ""])


# Deep Q Convolutional Neural Network
class DQCNN(ADQN.ADQN):

  def __init__(self):
    ADQN.ACTIVATION_MODE = TANH_ACTIVATION
    self.activation = tf.nn.relu
    super().__init__()
    self.input_shape = (1, 3, 3, 1)

  def network(self, weights=None, batch=1, num_classes=9):
    model = Sequential()
    # same effect if specify as kernel_size=3
    model.add(Conv2D(64, kernel_size=(3, 3), kernel_regularizer=tf.keras.regularizers.L1L2(),
                     activation=self.activation, padding='same', input_shape=(3, 3, 1)))
    model.add(MaxPooling2D((2, 2), padding='same'))
    # model.add(Dropout(0.25))
    model.add(Conv2D(128, (3, 3), kernel_regularizer=tf.keras.regularizers.L1L2(),
                     activation=self.activation, padding='same'))
    model.add(MaxPooling2D(pool_size=(2, 2), padding='same'))
    # model.add(Dropout(0.4))
    model.add(Flatten())
    model.add(Dense(256, activation=self.activation))
    # model.add(Dropout(0.3))
    model.add(Dense(num_classes, activation='softmax'))
    model.compile(loss='mean_squared_error',
                  optimizer=keras.optimizers.Adam(self.learning_rate))

    return model


if __name__ == "__main__":
  agent = DQCNN()
  init_arr = [0, 0, 0, 0, 1, 0, 0, 0, 0]
  next_arr = [-1, 0, 0, 0, 1, 0, 0, 0, 0]
  next_move = 0
  reward = 1
  tup = (init_arr, next_move, reward, next_arr, False)
  agent.train_short_memory(*tup)
  agent.remember(tup)

