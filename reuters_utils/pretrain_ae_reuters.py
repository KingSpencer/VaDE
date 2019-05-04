from keras.optimizers import Adam
from keras.optimizers import Nadam
from keras.layers import Input, Dense, Activation
from keras.models import Model
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

argv = sys.argv[1:]
parser = argparse.ArgumentParser()
parser.add_argument('-dataset', action='store', type = str, dest='dataset',  default = 'reuters10k', help='the options can be mnist,reuters10k and har')
results = parser.parse_args()
dataset = results.dataset


def get_ae(original_dim=2000, latent_dim=10, intermediate_dim=[500,500,2000]):
    x = Input(shape=(original_dim, ))
    h = Dense(intermediate_dim[0], activation='relu')(x)
    h = Dense(intermediate_dim[1], activation='relu')(h)
    h = Dense(intermediate_dim[2], activation='relu')(h)
    z = Dense(latent_dim)(h)
    h_decoded = Dense(intermediate_dim[-1], activation='relu')(z)
    h_decoded = Dense(intermediate_dim[-2], activation='relu')(h_decoded)
    h_decoded = Dense(intermediate_dim[-3], activation='relu')(h_decoded)
    x_decoded = Dense(original_dim, activation='linear')(h_decoded)


    ae = Model(x, x_decoded)
    ae.compile(optimizer='adadelta', loss='mse')
    return ae

def get_ae_supervised(original_dim=2000, latent_dim=10, intermediate_dim=[500,500,2000]):
    x = Input(shape=(original_dim, ))
    h = Dense(intermediate_dim[0], activation='relu')(x)
    h = Dense(intermediate_dim[1], activation='relu')(h)
    h = Dense(intermediate_dim[2], activation='relu')(h)
    z = Dense(latent_dim)(h)
    if dataset == 'reuters10k':
        y_pred = Dense(4, activation = 'softmax', name='prediction_out')(z)
    if dataset == 'mnist':
        y_pred = Dense(10, activation = 'softmax', name='prediction_out')(z)
    h_decoded = Dense(intermediate_dim[-1], activation='relu')(z)
    h_decoded = Dense(intermediate_dim[-2], activation='relu')(h_decoded)
    h_decoded = Dense(intermediate_dim[-3], activation='relu')(h_decoded)
    x_decoded = Dense(original_dim, activation='linear', name='decoded_out')(h_decoded)
    ae = Model(x, [x_decoded, y_pred])
    encoder = Model(x, z)
    ae.compile(optimizer='adadelta', loss=['mse','categorical_crossentropy'], loss_weights=[1,1], metrics={'decoded_out':'mae', 'prediction_out':'acc'})
    ae.summary()
    return ae, encoder

if __name__ == '__main__':
    batch_size = 128
    ## load X and Y from Reuters10k 
    if dataset == 'reuters10k':
        path = '../dataset/reuters10k'
        # path = '/Users/crystal/Documents/VaDE/dataset/reuters10k'
        data=scio.loadmat(os.path.join(path,'reuters10k.mat'))
        X = data['X']
        Y = data['Y'].squeeze()
    if dataset == 'mnist':
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
        
        
    
    ## change Y to one hot encoding
    #encoder = LabelEncoder()
    #encoder.fit(Y)
    #encoded_Y = encoder.transform(Y)
    # convert integers to dummy variables (i.e. one hot encoded)
    dummy_y = np_utils.to_categorical(Y)
    if dataset == 'reuters10k':
        ae,encoder = get_ae_supervised()
    if dataset == 'mnist':
        ae,encoder = get_ae_supervised(original_dim=784, latent_dim=10)
        
    ## choose part of data as training and part of it as validation
    total_obs = len(Y)
    prop = 0.4
    nTrain = np.round(total_obs * prop)
    train_ind = np.random.choice(total_obs, int(nTrain), replace=False)
    test_ind = np.setdiff1d(np.arange(0, total_obs, 1), train_ind)
    
    x_train = X[train_ind, :]
    y_train = dummy_y[train_ind, :]
    x_test = X[test_ind, :]
    y_test = dummy_y[test_ind, :]
    ae.fit(x_train, [x_train, y_train], epochs=2, batch_size=batch_size, validation_data=(x_test, [x_test, y_test]), shuffle=True)
    
    # ae.fit(X, [X, dummy_y], epochs=2, batch_size=batch_size)
    
    latent_z = encoder.predict(X, batch_size=batch_size)
    ## output z  
    latent_mnist = {'z': latent_z, 'y': Y}
    
    if dataset == 'mnist':
        with open('./latent_mnist_supervised.pkl', 'wb') as f:
            pickle.dump(latent_mnist, f)
    if dataset == 'reuters10k':
        with open('./latent_reuters10k_supervised.pkl', 'wb') as f:
            pickle.dump(latent_mnist, f)
            
    model_json = ae.to_json()
    # output_path = '/Users/crystal/Documents/VaDE/pretrain_weights'
    output_path = '../pretrain_weights'
    
    if dataset == 'reuters10k':
        with open(os.path.join(output_path, "ae_reuters10k_supervised.json"), "w") as json_file:
            json_file.write(model_json)
        ae.save_weights(os.path.join(output_path, "ae_reuters10k_supervised_weights.h5"))
    if dataset == 'mnist':
        with open(os.path.join(output_path, "ae_mnist_supervised.json"), "w") as json_file:
            json_file.write(model_json)
        ae.save_weights(os.path.join(output_path, "ae_mnist_supervised_weights.h5"))
        


