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


weight_reset_size = 64
min_replay_size = 384
reset_count = 1

LABELS_FOR_WIN_LOSE_DRAW = set(["", "", ""])


#Deep Q Convolutional Neural Network
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
    # model.add(LeakyReLU(alpha=0.1))
    model.add(MaxPooling2D((2, 2), padding='same'))
    # model.add(Dropout(0.25))
    # model.add(Conv2D(64, (3, 3), activation='linear', padding='same'))
    # model.add(LeakyReLU(alpha=0.1))
    # model.add(MaxPooling2D(pool_size=(2, 2), padding='same'))
    # model.add(Dropout(0.25))
    model.add(Conv2D(128, (3, 3), kernel_regularizer=tf.keras.regularizers.L1L2(),
                     activation=self.activation, padding='same'))
    # model.add(LeakyReLU(alpha=0.1))
    model.add(MaxPooling2D(pool_size=(2, 2), padding='same'))
    # model.add(Dropout(0.4))
    model.add(Flatten())
    model.add(Dense(256, activation=self.activation))
    # model.add(LeakyReLU(alpha=0.1))
    # model.add(Dropout(0.3))
    model.add(Dense(num_classes, activation='softmax'))
    model.compile(loss='mean_squared_error',
                  optimizer=keras.optimizers.Adam(self.learning_rate))
    # fashion_train_dropout = model.fit(train_X, train_label, batch_size=batch_size,epochs=epochs,verbose=1,
    # validation_data=(valid_X, valid_label))
    # predicted_classes = model.predict(test_X)
    # test_eval = model.evaluate(test_X, test_Y_one_hot, verbose=1)
    return model

  def train_short_memory(self, init_arr, next_move, reward, next_arr, done):
    fc_state = np.asarray(init_arr).reshape(self.input_shape)
    fn_state = np.asarray(next_arr).reshape(self.input_shape)

    target = reward
    if not done:
      target = reward + self.gamma * np.amax(self.model.predict(fn_state)[0])

    # reshaped and flattened current state
    # rfc_state = fc_state.reshape((1, 9))
    future_target = self.model.predict(fc_state)

    future_target[0][next_move] = target
    self.model.fit(fc_state, future_target, epochs=1, verbose=0)

  def plot(self, fashion_train_dropout):
    accuracy = fashion_train_dropout.history['acc']
    val_accuracy = fashion_train_dropout.history['val_acc']
    loss = fashion_train_dropout.history['loss']
    val_loss = fashion_train_dropout.history['val_loss']
    epochs = range(len(accuracy))
    plt.plot(epochs, accuracy, 'bo', label='Training accuracy')
    plt.plot(epochs, val_accuracy, 'b', label='Validation accuracy')
    plt.title('Training and validation accuracy')
    plt.legend()
    plt.figure()
    plt.plot(epochs, loss, 'bo', label='Training loss')
    plt.plot(epochs, val_loss, 'b', label='Validation loss')
    plt.title('Training and validation loss')
    plt.legend()
    plt.show()

if __name__ == "__main__":
  agent = DQCNN()
  init_arr = [0, 0, 0, 0, 1, 0, 0, 0, 0]
  next_arr = [-1, 0, 0, 0, 1, 0, 0, 0, 0]
  next_move = 0
  reward = 1
  tup = (init_arr, next_move, reward, next_arr, False)
  agent.train_short_memory(*tup)
  agent.remember(tup)

