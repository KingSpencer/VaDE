#%%
import numpy as np
from keras.callbacks import Callback, ModelCheckpoint
from keras.optimizers import Adam
from keras.layers import Input, Dense, Lambda, Conv2D, Reshape, UpSampling2D, MaxPooling2D, Flatten
from keras.models import Model
from keras import backend as K
from keras.losses import mse
from keras import optimizers

from keras import objectives
import scipy.io as scio
import gzip
from six.moves import cPickle
import os
import sys
import argparse


#import theano.tensor as T
import math
from sklearn import mixture
from sklearn.cluster import KMeans
from keras.models import model_from_json
import tensorflow as tf

import pickle

def _gen_one_hot(Y, dim):
        num = len(Y)
        one_hot = np.zeros((num, dim))
        one_hot[np.arange(num), Y] = 1
        return one_hot

class ConvClassifier:
    def __init__(self, latent_dim = 10):
        self.latent_dim = latent_dim
        self.trained = False
        
    def load_data(self, root_path="/home/zifeng/Research/VaDE", flatten=False):
        dataset = 'mnist'
        path = os.path.join(os.path.join(root_path, 'dataset'), dataset)
        # path = 'dataset/'+dataset+'/'
        if dataset == 'mnist':
            path = os.path.join(path, 'mnist.pkl.gz')
            if path.endswith(".gz"):
                f = gzip.open(path, 'rb')
            else:
                f = open(path, 'rb')
        
            if sys.version_info < (3,):
                (x_train, y_train), (x_test, y_test) = cPickle.load(f)
            else:
                (x_train, y_train), (x_test, y_test) = cPickle.load(f, encoding="bytes")
        
            f.close()
            x_train = x_train.astype('float32') / 255.
            x_test = x_test.astype('float32') / 255.

        self.x_train = np.expand_dims(x_train, axis=-1)
        self.x_test = np.expand_dims(x_test, axis=-1)

        self.y_train = _gen_one_hot(y_train, 10)
        self.y_test = _gen_one_hot(y_test, 10)

    def construct_encoding_classifier(self):
        latent_dim = self.latent_dim
        input_img = Input(shape=(28, 28, 1))  # adapt this if using `channels_first` image data format
        x = Conv2D(32, (3, 3), activation='relu', padding='same')(input_img)
        x = MaxPooling2D((2, 2), padding='same')(x)
        x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
        x = MaxPooling2D((2, 2), padding='same')(x)
        x = Conv2D(128, (3, 3), activation='relu', padding='same')(x)
        x = MaxPooling2D((2, 2), padding='same')(x)
        x = Flatten()(x)
        # z_mean = Dense(latent_dim)(x)
        output = Dense(10, activation='softmax')(x)

        self.model = Model(input_img, output)
        self.model.summary()
        # at this point the representation is (4, 4, 8) i.e. 128-dimensional

    def train(self, epochs=10, batch_size=128, lr=0.001, save_path='./conv_classifier_pre_weights'):
        optimizer = optimizers.Adam(lr=lr, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
        filepath = os.path.join(save_path, 'cnn_classifier.{epoch:02d}-{val_loss:.2f}.hdf5')
        ckpt = ModelCheckpoint(filepath, monitor='val_loss', verbose=0, save_best_only=True, save_weights_only=False, mode='auto', period=1)
        self.model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
        self.model.fit(self.x_train, self.y_train, validation_data=(self.x_test, self.y_test), epochs=epochs, callbacks=[ckpt])

if __name__ == '__main__':
    model = ConvClassifier(latent_dim=10)
    model.load_data()
    model.construct_encoding_classifier()
    model.train(epochs=10, batch_size=128, lr=0.001, save_path='./conv_classifier_pre_weights')


