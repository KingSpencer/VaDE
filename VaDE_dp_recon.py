import os, sys
from six.moves import cPickle
from keras.models import model_from_json
import scipy.io as scio
import gzip
import numpy as np
from GenImageUtil import get_models


def load_pretrain_online_weights(vade, online_path):
    OnlineModelFolder = online_path
    OnlineModelName = os.path.join(OnlineModelFolder, 'vade_DP_model.json')
    ae = model_from_json(open(OnlineModelName).read())

    OnlineWeightsName = os.path.join(OnlineModelFolder, 'vade_DP_weights.h5')
    vade.load_weights(OnlineWeightsName)
    return vade

def load_data(root_path='./VaDE', flatten=True):
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
    #x_train = x_train.astype('float32') / 255.
    x_test = x_test.astype('float32') / 255.
    if flatten:
        #x_train = x_train.reshape((len(x_train), np.prod(x_train.shape[1:])))
        x_test = x_test.reshape((len(x_test), np.prod(x_test.shape[1:])))

    #X = np.concatenate((x_train, x_test))
    X = x_test
    X_dict = {}
    if not flatten:
        X = np.expand_dims(X, axis=-1)
    # Y = np.concatenate((y_train, y_test))
    y = y_test
    for i in range(10):
        indices = np.where(y == i)[0]
        X_dict[i] = X[indices, :]
        #X_reordered[i*1000:(i+1)*1000, :] = X[indices, :]
    return X_dict

def reconstruction_error(X, X_recon):
    return np.mean(np.sum(((X - X_recon) ** 2), axis=1))

if __name__ == "__main__":
    digits = range(10)
    online_path = ''
    vade_ini, encoder, decoder = get_models(model_flag='dense', batch_size=128, original_dim=784, latent_dim=10, intermediate_dim=[500, 500, 2000])
    vade = load_pretrain_online_weights(vade_ini, online_path)
    X_dict = load_data(root_path='.')
    recon_dict = {}  
    for digit in digits:
        X_per_digit = X_dict[digit]
        X_per_digit_recon = vade.predict(X_per_digit)
        recon_dict[digit] = (X_per_digit.shape[0], reconstruction_error(X_per_digit, X_per_digit_recon))
        print("Reconstruction Error for digit {}".format(digit))
        print(recon_dict[digit])

    #print(y)