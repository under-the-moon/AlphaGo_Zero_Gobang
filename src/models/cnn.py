import keras
import os
from keras.layers import Input, Conv2D, Dense, Flatten
from keras.regularizers import l2
from keras.optimizers import Adam
import keras.backend as K
from keras.models import Model as KerasModel
from src import config
import numpy as np

"""
build an simple network
"""


class Model(object):

    def __init__(self, weights_path=None):
        self.width = config.default_width
        self.height = config.default_height
        self.channel = config.channel
        self.c = config.l2_c
        self.batch_size = config.batch_size
        self.weights_path = weights_path
        self.save_path = config.save_path
        self.classes_num = self.width * self.height
        self.model = self._init_model()
        self.load_weigths()

    def _init_model(self):
        x = Input((self.height, self.width, self.channel))
        output = Conv2D(filters=32, kernel_size=3, padding='same', activation='relu', kernel_regularizer=l2(self.c))(x)
        output = Conv2D(filters=64, kernel_size=3, padding='same', activation='relu', kernel_regularizer=l2(self.c))(output)
        output = Conv2D(filters=64, kernel_size=3, strides=(2, 2), padding='same', activation='relu', kernel_regularizer=l2(self.c))(output)
        output = Conv2D(filters=128, kernel_size=3, padding='same', activation='relu', kernel_regularizer=l2(self.c))(output)
        output = Conv2D(filters=128, kernel_size=3, strides=(2, 2), padding='same', activation='relu', kernel_regularizer=l2(self.c))(output)
        output = Flatten()(output)
        policy = Dense(self.classes_num, activation='softmax', kernel_regularizer=l2(self.c))(output)
        value = Dense(1, activation="tanh", kernel_regularizer=l2(self.c))(output)
        model = KerasModel(inputs=x, outputs=[policy, value])
        return model

    def train(self, X, y):
        opt = Adam(lr=config.lr)
        losses = ['categorical_crossentropy', 'mean_squared_error']
        self.model.compile(opt, loss=losses)
        samples_num = len(X)
        assert samples_num >= self.batch_size, 'sample_num should be larger than per batch data'
        steps_num = samples_num // self.batch_size
        losses = []
        for i in range(steps_num + 1):
            train_X, train_y = X[i * self.batch_size: (i + 1) * self.batch_size, ...], \
                               [y[0][i * self.batch_size: (i + 1) * self.batch_size, ...],
                                y[1][i * self.batch_size: (i + 1) * self.batch_size, ...]]
            loss = self.model.train_on_batch(train_X, train_y)
            losses.append(loss)
        return np.array(losses).mean()

    def policy_value(self, board):
        x = board.state
        if not isinstance(x, np.ndarray):
            x = np.array(x)
        # add batch dim
        if len(x.shape) == 3:
            x = np.expand_dims(x, axis=0)
        return self.model.predict_on_batch(x)

    def save_weights(self, epoch=100):
        epoch = epoch // 100
        self.model.save_weights(os.path.join(self.save_path, 'alphago_zero_' + str(epoch) + '.h5'))

    def load_weigths(self):
        if self.weights_path and os.path.exists(self.weights_path):
            self.model.load_weights(self.weights_path)

    def summary(self):
        self.model.summary()
