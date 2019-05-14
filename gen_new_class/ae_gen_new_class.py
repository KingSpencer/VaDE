from keras.optimizers import Adam
from keras.optimizers import Nadam
from keras.layers import Input, Dense, Activation, Lambda
from keras import backend as K
from keras.models import Model, model_from_json
import pickle
import os
import scipy.io as scio
from sklearn.preprocessing import LabelEncoder
from keras.utils import np_utils
import argparse
import gzip
from six.moves import cPickle
import numpy as np
import sys
from keras import optimizers
from scipy.misc import imsave
'''
argv = sys.argv[1:]
parser = argparse.ArgumentParser()
parser.add_argument('-dataset', action='store', type = str, dest='dataset',  default = 'mnist', help='the options can be mnist,reuters10k and har')
parser.add_argument('-prop', action='store', type = float, dest='prop', default=0.2, help='proportion of whole data for training')
parser.add_argument('-newCluster', action='store_true', dest='newCluster', help='indicator for running the new cluster experiment')

results = parser.parse_args()
dataset = results.dataset
prop = results.prop
newCluster = results.newCluster
'''

parser = argparse.ArgumentParser()
parser.add_argument('-m', '--model', type = str,  default = 'vae', help='use vae or ae')
args = parser.parse_args()

def sampling(args):
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

def get_ae(original_dim=784, latent_dim=10, intermediate_dim=[500,500,2000]):
    x = Input(shape=(original_dim, ))
    h = Dense(intermediate_dim[0], activation='relu')(x)
    h = Dense(intermediate_dim[1], activation='relu')(h)
    h = Dense(intermediate_dim[2], activation='relu')(h)
    z = Dense(latent_dim)(h)
    h_decoded = Dense(intermediate_dim[-1], activation='relu')(z)
    h_decoded = Dense(intermediate_dim[-2], activation='relu')(h_decoded)
    h_decoded = Dense(intermediate_dim[-3], activation='relu')(h_decoded)
    x_decoded = Dense(original_dim, activation='sigmoid')(h_decoded)


    ae = Model(x, x_decoded)
    adam = optimizers.Adam(lr=0.0005, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0, amsgrad=False)
    ae.compile(optimizer=adam, loss='mse')
    ae.summary()
    return ae

def get_vae(original_dim=784, latent_dim=10, intermediate_dim=[500,500,2000]):
    x = Input(shape=(original_dim, ))
    h = Dense(intermediate_dim[0], activation='relu')(x)
    h = Dense(intermediate_dim[1], activation='relu')(h)
    h = Dense(intermediate_dim[2], activation='relu')(h)

    z_mean = Dense(latent_dim)(h)
    z_log_var = Dense(latent_dim)(h)
    z = Lambda(sampling, output_shape=(latent_dim,))([z_mean, z_log_var])

    h_decoded = Dense(intermediate_dim[-1], activation='relu')(z)
    h_decoded = Dense(intermediate_dim[-2], activation='relu')(h_decoded)
    h_decoded = Dense(intermediate_dim[-3], activation='relu')(h_decoded)
    x_decoded = Dense(original_dim, activation='sigmoid')(h_decoded)


    ae = Model(x, x_decoded)
    adam = optimizers.Adam(lr=0.0005, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0, amsgrad=False)
    ae.compile(optimizer=adam, loss='mse')
    ae.summary()
    return ae

if __name__ == '__main__':
    batch_size = 128
    path = '../dataset/mnist'
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
    x_train = x_train.reshape((len(x_train), np.prod(x_train.shape[1:])))
    x_test = x_test.reshape((len(x_test), np.prod(x_test.shape[1:])))

    X = np.concatenate((x_train,x_test))
    Y = np.concatenate((y_train,y_test))
    
    numList = [1, 3, 4, 5, 6, 7, 8, 9]   
        
    ## extract X and Y without 0 and 2
    indices = []
    for number in numList:
        indices += list(np.where(Y == number)[0])
    #indices = np.vstack(indices)
    x_major = X[indices]
    y_major = Y[indices]

    minor_indices = list(set(range(len(Y))).difference(set(indices)))
    x_minor = X[minor_indices]
    y_minor = Y[minor_indices]

    #print(set(y_minor))
    #exit(0)
        
           
    ## change Y to one hot encoding
    #encoder = LabelEncoder()
    #encoder.fit(Y)
    #encoded_Y = encoder.transform(Y)
    # convert integers to dummy variables (i.e. one hot encoded)
    #dummy_y_major = np_utils.to_categorical(Y)
    if args.model == 'ae':
        ae = get_ae()
    elif args.model == 'vae':
        ae = get_vae()


    ae.fit(x_major, x_major, epochs=5, batch_size=batch_size, validation_data=(x_minor, x_minor), shuffle=True)
    
    # ae.fit(X, [X, dummy_y], epochs=2, batch_size=batch_size)
    
    # recon = ae.predict(X, batch_size=batch_size)
    ## output z  

        # generate a sample
    indices_sample = list(np.where(Y == 0)[0])[:10] + list(np.where(Y == 2)[0])[:10]
    x_sample = X[indices_sample]
    img_sample = ae.predict(x_sample)
    img_sample *= 255
    x_sample *= 255
    #img_sampe = img_sample.astype(np.uint8)
    for i in range(20):
        img_original = (x_sample[i]).reshape(28,28).astype(np.uint8)
        imsave('sample_original_%d.png' % i, img_original)
        imsave('sample_recon_%d.png' % i, img_sample[i].reshape(28,28))


