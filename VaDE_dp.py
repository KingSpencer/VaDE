import numpy as np
from keras.callbacks import Callback
from keras.optimizers import Adam
from keras.layers import Input, Dense, Lambda
from keras.models import Model
from keras import backend as K
from keras import objectives
import scipy.io as scio
import gzip
from six.moves import cPickle
import sys

#import theano.tensor as T
import math
from sklearn import mixture
from sklearn.cluster import KMeans
from keras.models import model_from_json
import tensorflow as tf

def sampling(args):
    z_mean, z_log_var = args
    epsilon = K.random_normal(shape=(batch_size, latent_dim), mean=0.)
    return z_mean + K.exp(z_log_var / 2) * epsilon

def load_data(dataset):
    path = 'dataset/'+dataset+'/'
    if dataset == 'mnist':
        path = path + 'mnist.pkl.gz'
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
        
    if dataset == 'reuters10k':
        data=scio.loadmat(path+'reuters10k.mat')
        X = data['X']
        Y = data['Y'].squeeze()
        
    if dataset == 'har':
        data=scio.loadmat(path+'HAR.mat')
        X=data['X']
        X=X.astype('float32')
        Y=data['Y']-1
        X=X[:10200]
        Y=Y[:10200]

    return X,Y

def config_init(dataset):
    if dataset == 'mnist':
        return 784,3000,10,0.002,0.002,10,0.9,0.9,1,'sigmoid'
    if dataset == 'reuters10k':
        return 2000,15,4,0.002,0.002,5,0.5,0.5,1,'linear'
    if dataset == 'har':
        return 561,120,6,0.002,0.00002,10,0.9,0.9,5,'linear'

def penalized_loss(noise):
    def loss(y_true, y_pred):
        return K.mean(K.square(y_pred - y_true) - K.square(y_true - noise), axis=-1)
    return loss

def load_pretrain_weights(vade):
    ae = model_from_json(open('pretrain_weights/ae_'+dataset+'.json').read())
    ae.load_weights('pretrain_weights/ae_'+dataset+'_weights.h5')
    vade.layers[1].set_weights(ae.layers[0].get_weights())
    vade.layers[2].set_weights(ae.layers[1].get_weights())
    vade.layers[3].set_weights(ae.layers[2].get_weights())
    vade.layers[4].set_weights(ae.layers[3].get_weights())
    vade.layers[-1].set_weights(ae.layers[-1].get_weights())
    vade.layers[-2].set_weights(ae.layers[-2].get_weights())
    vade.layers[-3].set_weights(ae.layers[-3].get_weights())
    vade.layers[-4].set_weights(ae.layers[-4].get_weights())
    return vade

'''def elbo_nn(DPParam):
    #gamma = DPParam['LPMtx']
    #N = DPParam['Nvec']
    #m = DPParam['m']
    #W = DPParam['W']
    #v = DPParam['nu']
    #k = v.shape[0]
    def loss(x, x_decoded_mean):
        N = tf.convert_to_tensor(DPParam, dtype=tf.float32)
        loss_=alpha*original_dim * objectives.mean_squared_error(x, x_decoded_mean) + \
        -0.5 * K.sum(z_log_var, axis = -1) + N
        return loss_
        
        # line 93 term
    return loss'''



dataset = 'mnist'
#db = sys.argv[1]
#if db in ['mnist','reuters10k','har']:
#    dataset = db
print ('training on: ' + dataset)
ispretrain = True
batch_size = 50
latent_dim = 10
intermediate_dim = [500,500,2000]
#theano.config.floatX='float32'
accuracy=[]
X, Y = load_data(dataset)
original_dim,epoch,n_centroid,lr_nn,lr_gmm,decay_n,decay_nn,decay_gmm,alpha,datatype = config_init(dataset)
global DPParam

# gamma: 'LPMtx' (batch_size, # of cluster)
# N : 'Nvec' (# of cluster, )
# m : 'm' (# of cluster, latent_dim)
# W : 'B' (# of cluster, latent_dim, latent_dim)
# v: 'nu' (# of cluster) 
def loss(x, x_decoded_mean):
    #N = tf.convert_to_tensor(DPParam, dtype=tf.float32)
    

    gamma = tf.convert_to_tensor(DPParam['LPMtx'], dtype=tf.float32)
    N = tf.convert_to_tensor(DPParam['Nvec'], dtype=tf.float32)
    m = tf.convert_to_tensor(DPParam['m'], dtype=tf.float32)
    W = tf.convert_to_tensor(DPParam['B'], dtype=tf.float32)
    v = tf.convert_to_tensor(DPParam['nu'], dtype=tf.float32)

    num_cluster = N.shape[0]
    z_mean_1_last = tf.expand_dims(z_mean, -1) # bs, latent_dim, 1
    z_mean_1_mid = tf.expand_dims(z_mean, 1) # bs, 1, latent_dim

    for k in range(num_cluster):
        gamma_k_rep = tf.squeeze(K.repeat(tf.expand_dims(gamma[:, k], -1), latent_dim))
        z_k_bar = 1/N[k] * K.sum(tf.multiply(gamma_k_rep, z_mean), axis=0) #(latent_dim, )
        z_k_bar_batch = tf.squeeze(K.repeat(tf.expand_dims(z_k_bar, 0), batch_size))
        #tf.transpose(z_k_bar_batch, perm=[1, 0])
        z_k_bar_batch_1_last = tf.expand_dims(z_k_bar_batch, -1) # bs, latent_dim, 1
        z_k_bar_batch_1_mid = tf.expand_dims(z_k_bar_batch, 1) # bs, 1, latent_dim
        
        # TODO:!
        S_k = 1/N[k] * K.sum(K.batch_dot(tf.multiply(tf.expand_dims(gamma_k_rep,-1), (z_mean_1_last-z_k_bar_batch_1_last)), z_mean_1_mid - z_k_bar_batch_1_mid), axis=0) # (latent_dim, latent_dim)
        temp = tf.linalg.trace(tf.matmul(S_k, W[k]))
        temp2 = tf.matmul(tf.expand_dims((z_k_bar-m[k]), 0), W[k])
        temp3 = tf.squeeze(tf.matmul(temp2, tf.expand_dims((z_k_bar-m[k]), -1)))
        if k == 0:
            e = 0.5*N[k]*(v[k]*(temp + temp3))
        else:
            e += 0.5*N[k]*(v[k]*(temp + temp3))

    loss_= alpha*original_dim * objectives.mean_squared_error(x, x_decoded_mean)-0.5 * K.sum(z_log_var, axis = -1)
    loss = K.sum(loss_, axis = 0) + e
    #for i in range(5):
    #    loss_ += N
        
    return loss

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
x_decoded_mean = Dense(original_dim, activation=datatype)(h_decoded)

sample_output = Model(x, z_mean)

vade = Model(x, x_decoded_mean)
if ispretrain == True:
    vade = load_pretrain_weights(vade)

num_of_exp = X.shape[0]
num_of_epoch = 1
num_of_iteration = int(num_of_exp / batch_size)
adam_nn= Adam(lr=lr_nn,epsilon=1e-4)
for epoch in range(num_of_epoch):
    id_list = np.arange(num_of_exp)
    np.random.shuffle(id_list)
    #print(id_list)
    #exit(0)
    for iteration in range(num_of_iteration):
        indices = id_list[iteration*batch_size:(iteration+1)*batch_size]
        x_batch = X[indices, :]
        #print(x_batch)
        # forward pass
        z_batch = sample_output.predict(x_batch)
        #print(z_batch)

        # to DP
        # DPParam = DP_fit(z_batch)
        # DPParam = np.ones((batch_size))
        # gamma: 'LPMtx' (batch_size, # of cluster)
        # N : 'Nvec' (# of cluster, )
        # m : 'm' (# of cluster, latent_dim)
        # W : 'B' (# of cluster, latent_dim, latent_dim)
        # v: 'nu' (# of cluster) 
        k = 5
        DPParam = \
        {
            'LPMtx': np.ones((batch_size, k)),
            'Nvec' : np.ones(k),
            'm'    : np.ones((k, latent_dim)),
            'B'    : np.ones((k, latent_dim, latent_dim)),
            'nu'   : np.ones(k)
        }
        vade.compile(optimizer=adam_nn, loss=loss)
        neg_elbo = vade.train_on_batch(x_batch, x_batch)
        print("Iteration: {}, ELBO: {}".format(iteration, -neg_elbo))
        if iteration == 5:
            exit(0)














