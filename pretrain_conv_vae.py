#%%
import numpy as np
from keras.callbacks import Callback, ModelCheckpoint
from keras.optimizers import Adam
from keras.layers import Input, Dense, Lambda, Conv2D, Reshape, UpSampling2D, MaxPooling2D, Flatten, Activation
from keras.models import Model, load_model
from keras import backend as K
from keras.losses import mse

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
import argparse
argv = sys.argv[1:]
parser = argparse.ArgumentParser()
parser.add_argument('-prop', action='store', type = float, dest='prop', default=0.4, help='proportion of whole data for training')

results = parser.parse_args()
prop = results.prop



class AE_model:
    def __init__(self, latent_dim = 10, batch_size=500):
        self.latent_dim = latent_dim
        self.trained = False
        self.batch_size = batch_size

    def load_data(self, root_path=".", flatten=False):
        dataset = 'mnist'
        path = os.path.join(os.path.join(root_path, 'dataset'), dataset)
        # path = 'dataset/'+dataset+'/'

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
        x_train = np.expand_dims(x_train, axis=-1)
        x_test = np.expand_dims(x_test, axis=-1)            
        y_test = np.eye(10)[y_test]
        y_train = np.eye(10)[y_train]

        #self.x_train = x_test
        #self.x_test = x_train
        #self.y_train = y_test
        #self.y_test = y_train
        
        ###########################################
        ## get more 4 and 9,  5, 3 and 8
        X = np.concatenate((x_train,x_test))
        Y = np.concatenate((y_train,y_test))
        ## sample 30% of the whole data as training
        ## the rest as validation
        total_obs = len(Y)
        nTrain = np.round(total_obs * prop)
        train_ind = np.random.choice(total_obs, int(nTrain), replace=False)
        test_ind = np.setdiff1d(np.arange(0, total_obs, 1), train_ind)
        
        self.x_train = X[train_ind, :, :, :]
        self.y_train = Y[train_ind, :]
        self.x_test = X[test_ind, :, :, :]
        self.y_test = Y[test_ind, :]
        
        #self.x_train = x_train
        #self.y_train = y_train
        #self.x_test = x_test
        #self.y_test = y_test

    '''def _sampling(self, args):
            latent_dim = self.latent_dim
            batch_size = self.batch_size
            z_mean, z_log_var = args
            epsilon = K.random_normal(shape=(batch_size, latent_dim), mean=0.)
            return z_mean + K.exp(z_log_var / 2) * epsilon'''

    def _sampling(self,args):
        """Reparameterization trick by sampling fr an isotropic unit Gaussian.

        # Arguments
            args (tensor): mean and log of variance of Q(z|X)

        # Returns
            z (tensor): sampled latent vector
        """

        z_mean, z_log_var = args
        batch = K.shape(z_mean)[0]
        dim = K.int_shape(z_mean)[1]
        # by default, random_normal has mean=0 and std=1.0
        epsilon = K.random_normal(shape=(batch, dim))
        return z_mean + K.exp(0.5 * z_log_var) * epsilon

    def construct_model(self):
        latent_dim = self.latent_dim
        batch_size = self.batch_size
        
        latent_dim = self.latent_dim
        input_img = Input(shape=(28, 28, 1))  # adapt this if using `channels_first` image data format
        x = Conv2D(32, (3, 3), activation='relu', padding='same')(input_img)
        x = MaxPooling2D((2, 2), padding='same')(x)
        x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
        x = MaxPooling2D((2, 2), padding='same')(x)
        x = Conv2D(128, (3, 3), activation='relu', padding='same')(x)
        x = MaxPooling2D((2, 2), padding='same')(x)
        # at this point the representation is (4, 4, 8) i.e. 128-dimensional
        # channel merge
        # x = Conv2D(1, (3, 3), activation='relu', padding='same')(x)
        # shape info needed to build decoder model
        shape = K.int_shape(x)
        x = Flatten()(x)
        z_mean = Dense(latent_dim, name='z_mean')(x)
        y_pred = Activation('softmax', name='pred_out')(z_mean)
        #z_log_var = Dense(latent_dim, name='z_log_var')(x)
        #z = Lambda(self._sampling, output_shape=(latent_dim,))([z_mean, z_log_var])
        # build decoder model
        # for generative model
        x = Dense(shape[1] * shape[2] * shape[3], activation='relu')(z_mean)
        x = Reshape((shape[1], shape[2], shape[3]))(x)

        x = Conv2D(128, (3, 3), activation='relu', padding='same')(x)
        x = UpSampling2D((2, 2))(x)
        x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
        x = UpSampling2D((2, 2))(x)
        x = Conv2D(32, (3, 3), activation='relu')(x)
        x = UpSampling2D((2, 2))(x)
        decoded = Conv2D(1, (3, 3), activation='sigmoid', padding='same', name='decoded')(x)

        # constructing several models
        #encoder = Model(input_img, z_mean, name='encoder')
        #decoder = Model(latent_inputs, decoded, name='decoder')
        #decoded_for_vade = decoder(encoder(input_img))
        vade = Model(input_img, [decoded, y_pred], name='vade')

        #reconstruction_loss = mse(K.flatten(input_img), K.flatten(decoded_for_vade))

        #vade.add_loss(reconstruction_loss)
        self.vade = vade
        vade.summary()
        # self.vade.compile(optimizer='adadelta', loss='binary_crossentropy')
        #self.encoder = encoder
        #self.decoder = decoder



    def train(self, epochs=5, save_path="./conv_vae_pre_weights"):
        self.vade.compile(optimizer='adadelta', loss=['mse', 'categorical_crossentropy'], loss_weights=[1.0, 1.0], metrics={'decoded':'mae', 'pred_out':'acc'})
        batch_size = self.batch_size
        if not os.path.exists(save_path):
            os.mkdir(save_path)
        #ckpt = ModelCheckpoint(save_path, monitor='loss', verbose=0, save_best_only=True, save_weights_only=True, mode='auto', period=1)
        self.vade.fit(self.x_train, [self.x_train, self.y_train],
                epochs=epochs,
                batch_size=batch_size,
                validation_data=(self.x_test, [self.x_test, self.y_test]),
                shuffle=True)
         #       callbacks=[ckpt])
        # self.vade.save('./conv_vae_pre_weights/vae_cnn_mnist.model')
        self.vade.save_weights('./conv_vae_pre_weights/vae_cnn_mnist_semi_supervised.weights')
        model_json = self.vade.to_json()
        # output_path = '/Users/crystal/Documents/VaDE/pretrain_weights'
        output_path = './conv_vae_pre_weights/'
        with open(os.path.join(output_path, "vae_cnn_mnist_semi_supervised.json"), "w") as json_file:
            json_file.write(model_json)
        self.trained = True

    def test_sample(self, save_path = "./sample.pkl"):
        if not self.trained:
            self.vade.load_weights('./conv_vae_pre_weights/vae_cnn_mnist_semi_supervised.weights')
        sample = self.vade.predict(self.X_data[0:2])
        with open(save_path, 'wb') as f:
            pickle.dump(sample, f)



if __name__ == "__main__":
    conv_AE = AE_model(10)
    conv_AE.load_data()
    conv_AE.construct_model()
    conv_AE.train(epochs = 5)
    #conv_AE.test_sample()
