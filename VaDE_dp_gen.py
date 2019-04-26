import numpy as np
from numpy.random import multivariate_normal
from keras.callbacks import Callback
from keras.optimizers import Adam
from keras.optimizers import Nadam
from keras.layers import Input, Dense, Lambda, Conv2D, Reshape, UpSampling2D, MaxPooling2D, Flatten
from keras.models import Model, load_model
from keras import backend as K

from keras import objectives
import scipy.io as scio
import gzip
from six.moves import cPickle
import os
import sys
import argparse

from scipy.misc import imsave

import math
from tqdm import tqdm
from sklearn import mixture
from sklearn.cluster import KMeans
from keras.models import model_from_json
import tensorflow as tf
from sklearn.externals import joblib

# fix path
sys.path.append("../bnpy")
sys.path.append("../bnpy/bnpy")

class DPVAE_Generator:
    def __init__():
        pass

def sampling(args):
    z_mean, z_log_var = args
    epsilon = K.random_normal(shape=(batch_size, latent_dim), mean=0.)
    return z_mean + K.exp(z_log_var / 2) * epsilon

def get_models(batch_size, original_dim, latent_dim, intermediate_dim):
    x = Input(batch_shape=(batch_size, original_dim))
    h = Dense(intermediate_dim[0], activation='relu')(x)
    h = Dense(intermediate_dim[1], activation='relu')(h)
    h = Dense(intermediate_dim[2], activation='relu')(h)
    z_mean = Dense(latent_dim)(h)
    z_log_var = Dense(latent_dim)(h)
    z = Lambda(sampling, output_shape=(latent_dim,))([z_mean, z_log_var])

    latent_inputs = Input(shape=(latent_dim,), name='z_sampling')

    h_decoded = Dense(intermediate_dim[-1], activation='relu')(latent_inputs)
    h_decoded = Dense(intermediate_dim[-2], activation='relu')(h_decoded)
    h_decoded = Dense(intermediate_dim[-3], activation='relu')(h_decoded)
    x_decoded_mean = Dense(original_dim, activation='sigmoid')(h_decoded)

    encoder = Model(x, z, name='encoder')
    decoder = Model(latent_inputs, x_decoded_mean, name='decoder')

    vade = Model(x, decoder(encoder(x)))
    #vade = Model(x, x_decoded_mean)
    return vade , encoder, decoder

def get_temp_vade(batch_size, original_dim, latent_dim, intermediate_dim):
    x = Input(batch_shape=(batch_size, original_dim))
    h = Dense(intermediate_dim[0], activation='relu')(x)
    h = Dense(intermediate_dim[1], activation='relu')(h)
    h = Dense(intermediate_dim[2], activation='relu')(h)
    z_mean = Dense(latent_dim)(h)
    z_log_var = Dense(latent_dim)(h)
    z = Lambda(sampling, output_shape=(latent_dim,))([z_mean, z_log_var])

    h_decoded = Dense(intermediate_dim[-1], activation='relu')(z)
    h_decoded = Dense(intermediate_dim[-2], activation='relu')(h_decoded)
    h_decoded = Dense(intermediate_dim[-3], activation='relu')(h_decoded)
    x_decoded_mean = Dense(original_dim, activation='sigmoid')(h_decoded)

    vade = Model(x, x_decoded_mean)
    return vade

'''def load_pretrain_vade(vade, root_path="/home/zifeng/Research/VaDE/results"):
    batch_size, latent_dim = 128, 10
    path = os.path.join(root_path, 'vade_DP.hdf5')
    prev_vade = load_model(path)
    for i in range(1, 5):
        vade.layers[i].set_weights(prev_vade.layers[i].get_weights())
        vade.layers[-i].set_weights(prev_vade.layers[-i].get_weights())
    return vade'''

def load_pretrain_weights(encoder, decoder, dataset="mnist", root_path="/home/zifeng/Research/VaDE"):
    
    path = os.path.join(root_path, 'pretrain_weights')
    filename = 'ae_' + dataset + '.json'
    fullFileName = os.path.join(path, filename)
    ae = model_from_json(open(fullFileName).read())
    # ae = model_from_json(open('pretrain_weights/ae_'+dataset+'.json').read())
    weightFileName = 'ae_' + dataset + '_weights.h5'
    weightFullFileName = os.path.join(path, weightFileName)
    ae.load_weights(weightFullFileName)
    
    #ae.load_weights('pretrain_weights/ae_'+dataset+'_weights.h5')
    encoder.layers[1].set_weights(ae.layers[0].get_weights())
    encoder.layers[2].set_weights(ae.layers[1].get_weights())
    encoder.layers[3].set_weights(ae.layers[2].get_weights())
    encoder.layers[4].set_weights(ae.layers[3].get_weights())
    decoder.layers[-1].set_weights(ae.layers[-1].get_weights())
    decoder.layers[-2].set_weights(ae.layers[-2].get_weights())
    decoder.layers[-3].set_weights(ae.layers[-3].get_weights())
    decoder.layers[-4].set_weights(ae.layers[-4].get_weights())
    return encoder, decoder

def load_pretrain_vade_weights(encoder, decoder, vade_temp):
    encoder.layers[1].set_weights(vade_temp.layers[1].get_weights())
    encoder.layers[2].set_weights(vade_temp.layers[2].get_weights())
    encoder.layers[3].set_weights(vade_temp.layers[3].get_weights())
    encoder.layers[4].set_weights(vade_temp.layers[4].get_weights())
    decoder.layers[-1].set_weights(vade_temp.layers[-1].get_weights())
    decoder.layers[-2].set_weights(vade_temp.layers[-2].get_weights())
    decoder.layers[-3].set_weights(vade_temp.layers[-3].get_weights())
    decoder.layers[-4].set_weights(vade_temp.layers[-4].get_weights())
    return encoder, decoder

if __name__ == "__main__":
    batch_size = 128
    original_dim = 784
    latent_dim = 10
    intermediate_dim = [500,500,2000]
    vade, encoder, decoder = get_models(batch_size, original_dim, latent_dim, intermediate_dim)
    vade_temp = get_temp_vade(batch_size, original_dim, latent_dim, intermediate_dim)
    #vade.summary()
    #encoder.summary()
    #decoder.summary()
    # encoder, decoder = load_pretrain_weights(encoder, decoder)
    vade_temp.load_weights("/home/zifeng/Research/VaDE/results/vade_DP_weights.h5")
    print("************* weights loaded successfully! **************")
    encoder, decoder = load_pretrain_vade_weights(encoder, decoder, vade_temp)
    # test loading and saving model structure
    #jj = vade.to_json()
    #aa = model_from_json(jj)
    # not working ...
    # print(len(vade.get_weights()))
    # TODO: Load DP parameters and generate new data
    # with open('./results/m.pkl', 'rb') as f:
    #    m = joblib.load(f)
    # with open('./results/W.pkl', 'rb') as f:
    #    W = joblib.load(f)
    with open('./results/DPParam.pkl', 'rb') as f:
        DPParam = joblib.load(f)
    m = DPParam['m']
    W = DPParam['B']
    nu = DPParam['nu']
    beta = DPParam['kappa']

    # sampling from gaussians
    cluster_sample_list = []
    print("************* generating new data! **************")
    for nc in tqdm(range(len(m))):
        mean = m[nc]
        #lam = np.linalg.inv(W[nc]) * nu[nc]
        #var = np.linalg.inv(lam)
        var = W[nc] * 1 / float(nu[nc])
        z_sample = multivariate_normal(mean, var, 12)
        # we then feed z_sample to the decoder
        generated = decoder.predict(z_sample)
        generated = generated.reshape(-1, 28, 28)
        generated = generated * 255
        generated = generated.astype(np.uint8)
        generated_list = [generated[x] for x in range(generated.shape[0])]
        flattened_generated = np.hstack(generated_list)
        cluster_sample_list.append(flattened_generated)
        # print(flattened_generated.shape)
        # print(z_sample.shape)
    merged_sample = np.vstack(cluster_sample_list)
    imsave('./results/sample.png', merged_sample)

    cluster_mean_list = []
    print("************* generating new data with mean! **************")
    for nc in tqdm(range(len(m))):
        mean = m[nc]

        z_sample = np.expand_dims(mean, 0)
        # we then feed z_sample to the decoder
        generated = decoder.predict(z_sample)
        generated = generated.reshape(28, 28)
        generated = generated * 255
        generated = generated.astype(np.uint8)
        cluster_mean_list.append(generated)
        # print(flattened_generated.shape)
        # print(z_sample.shape)
    merged_mean_sample = np.hstack(cluster_mean_list)
    imsave('./results/mean_sample.png', merged_mean_sample)
