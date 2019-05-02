#%%
import numpy as np
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


#import theano.tensor as T
import math
from sklearn import mixture
from sklearn.cluster import KMeans
from keras.models import model_from_json
import tensorflow as tf
from sklearn.externals import joblib ## replacement of pickle to carry large numpy arrays
import pickle


os.environ['KMP_DUPLICATE_LIB_OK']='True'

argv = sys.argv[1:]
parser = argparse.ArgumentParser()
parser.add_argument('-bnpyPath', action='store', type = str, dest='bnpyPath', default='/Users/crystal/Documents/bnpy/', \
                    help='path to bnpy code repo')
parser.add_argument('-outputPath', action='store', type = str, dest='outputPath', default='/Users/crystal/Documents/VaDE_results/', \
                    help='path to output')
parser.add_argument('-rootPath', action='store', type = str, dest='rootPath', default='/Users/crystal/Documents/VaDE', \
                    help='root path to VaDE')
parser.add_argument('-conv', action='store_true', \
                    help='using convolutional autoencoder or not')
parser.add_argument('-logFile', action='store_true', dest='logFile', help='if logfile exists, save the log file to txt')
parser.add_argument('-useLocal', action='store_true', dest='useLocal', help='if use Local, rep environment variable will not be used')
## add argument for the maximum number of clusters in DP
parser.add_argument('-Kmax', action='store', type = int, dest='Kmax',  default=50, help='the maximum number of clusters in DPMM')
## parse data set option as an argument
parser.add_argument('-dataset', action='store', type = str, dest='dataset',  default = 'reuters10k', help='the options can be mnist,reuters10k and har')
parser.add_argument('-epoch', action='store', type = int, dest='epoch', default = 20, help='The number of epochs')
parser.add_argument('-batch_iter', action='store', type = int, dest='batch_iter', default = 10, help='The number of updates in SGVB')
parser.add_argument('-scale', action='store', type = float, dest='scale', default = 1.0, help='the scale parameter in the loss function')
parser.add_argument('-batchsize', action='store', type = int, dest='batchsize', default = 5000, help='the default batch size when training neural network')

parser.add_argument('-sf', action='store', type = float, dest='sf', default=0.1, help='the prior diagonal covariance matrix for Normal mixture in DP')
parser.add_argument('-gamma0', action='store', type = float, dest='gamma0', default=5.0, help='hyperparameters for DP in Beta dist')
parser.add_argument('-gamma1', action='store', type = float, dest='gamma1', default=1.0, help='hyperparameters for DP in Beta dist')

results = parser.parse_args()
if results.useLocal:
    parser.add_argument('-rep', action='store', type=int, dest = 'rep', default=1, help='add replication number as argument')
    results = parser.parse_args()
    rep = results.rep
else:
    rep = os.environ["rep"]
    rep = int(float(rep))
    

bnpyPath = results.bnpyPath
sys.path.append(bnpyPath)
outputPath = results.outputPath

if not os.path.exists(outputPath):
    os.mkdir(outputPath)

root_path = results.rootPath
sys.path.append(root_path)
Kmax = results.Kmax
dataset = results.dataset
epoch = results.epoch
batch_iter = results.batch_iter
scale = results.scale
batchsize = results.batchsize

## DP hyper-parameters
sf = results.sf
gamma0 = results.gamma0
gamma1 = results.gamma1

from OrganizeResultUtil import createOutputFolderName, createFullOutputFileName


rep = None


    
    
## Rep is useful when running the same experiment multiple times to obtain a standard error

flatten = True
if results.conv:
    flatten = False
    

## specify full output path
fullOutputPath = createOutputFolderName(outputPath, Kmax, dataset, epoch, batch_iter, scale, batchsize, rep)
## name log file and write console output to log.txt
logFileName = os. path.join(fullOutputPath, 'log.txt')
if results.logFile:
    sys.stdout = open(logFileName, 'w')

#############################################    
import DP as DP
from bnpy.util.AnalyzeDP import * 
from bnpy.data.XData import XData
import pickle


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

def load_data(dataset, root_path, flatten=True, numbers=range(10)):
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
        if flatten:
            x_train = x_train.reshape((len(x_train), np.prod(x_train.shape[1:])))
            x_test = x_test.reshape((len(x_test), np.prod(x_test.shape[1:])))

        X = np.concatenate((x_train,x_test))
        if not flatten:
            X = np.expand_dims(X, axis=-1)
        Y = np.concatenate((y_train,y_test))

        if len(numbers) == 10:
            pass
        else:
            indices = []
            for number in numbers:
                indices += list(np.where(Y == number)[0])
            #indices = np.vstack(indices)
            X = X[indices]
            Y = Y[indices]
        
    if dataset == 'reuters10k':
        data=scio.loadmat(os.path.join(path,'reuters10k.mat'))
        X = data['X']
        Y = data['Y'].squeeze()
        
    if dataset == 'har':
        data=scio.loadmat(path+'HAR.mat')
        X=data['X']
        X=X.astype('float32')
        Y=data['Y']-1
        X=X[:10200]
        Y=Y[:10200]

    if dataset == 'stl10':
        with open('./dataset/stl10/X.pkl', 'rb') as f:
            X = pickle.load(f)
        with open('./dataset/stl10/Y.pkl', 'rb') as f:
            Y = pickle.load(f)
            # here Y is one-hot, turn it back
            Y = np.argmax(Y, axis=1)

    return X,Y

def config_init(dataset):
    # original_dim,epoches,n_centroid,lr_nn,lr_gmm,decay_n,decay_nn,decay_gmm,alpha,datatype 
    if dataset == 'mnist':
        return 784,3000,10,0.002,0.002,10,0.9,0.9,1,'sigmoid'
    if dataset == 'reuters10k':
        return 2000,15,4,0.002,0.002,5,0.5,0.5,1,'linear'
    if dataset == 'har':
        return 561,120,6,0.002,0.00002,10,0.9,0.9,5,'linear'
    if dataset == 'stl10':
        return 2048,10,10,0.002,0.002,10,0.9,0.9,1,'linear'


def penalized_loss(noise):
    def loss(y_true, y_pred):
        return K.mean(K.square(y_pred - y_true) - K.square(y_true - noise), axis=-1)
    return loss

def load_pretrain_weights(vade, root_path, dataset):
    if dataset == 'stl10':
        dataset += '_supervised'
    if dataset == 'reuters10k':
        dataset += '_supervised'
    path = os.path.join(root_path, 'pretrain_weights')
    filename = 'ae_' + dataset + '.json'
    fullFileName = os.path.join(path, filename)
    ae = model_from_json(open(fullFileName).read())
    # ae = model_from_json(open('pretrain_weights/ae_'+dataset+'.json').read())
    weightFileName = 'ae_' + dataset + '_weights.h5'
    weightFullFileName = os.path.join(path, weightFileName)
    ae.load_weights(weightFullFileName)
    

    if 'stl10' not in dataset and 'reuters10k' not in dataset:
    #ae.load_weights('pretrain_weights/ae_'+dataset+'_weights.h5')
        vade.layers[1].set_weights(ae.layers[0].get_weights())
        vade.layers[2].set_weights(ae.layers[1].get_weights())
        vade.layers[3].set_weights(ae.layers[2].get_weights())
        vade.layers[4].set_weights(ae.layers[3].get_weights())

        vade.layers[-1].set_weights(ae.layers[-1].get_weights())
        vade.layers[-2].set_weights(ae.layers[-2].get_weights())
        vade.layers[-3].set_weights(ae.layers[-3].get_weights())
        vade.layers[-4].set_weights(ae.layers[-4].get_weights())
    else:
        vade.layers[1].set_weights(ae.layers[1].get_weights())
        vade.layers[2].set_weights(ae.layers[2].get_weights())
        vade.layers[3].set_weights(ae.layers[3].get_weights())
        vade.layers[4].set_weights(ae.layers[4].get_weights())
        vade.layers[-1].set_weights(ae.layers[-2].get_weights())
        vade.layers[-2].set_weights(ae.layers[-3].get_weights())
        vade.layers[-3].set_weights(ae.layers[-4].get_weights())
        vade.layers[-4].set_weights(ae.layers[-5].get_weights())
    
    return vade

def load_pretrain_cnn_encoder(encoder, root_path, model='cnn_classifier.05-0.02.hdf5'):
    print("Loading Pretrained Weights for CNN-VAE-Encoder!")
    path = os.path.join(root_path, 'conv_classifier_pre_weights', model)
    # layer cnn: 1, 3, 5, dense:8
    pre_encoder = load_model(path)
    for lid in [1, 3, 5, 8]:
        encoder.layers[lid].set_weights(pre_encoder.layers[lid].get_weights()) 
    return encoder
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





# gamma: 'LPMtx' (batch_size, # of cluster)
# N : 'Nvec' (# of cluster, )
# m : 'm' (# of cluster, latent_dim)
# W : 'B' (# of cluster, latent_dim, latent_dim)
# v: 'nu' (# of cluster) 
#def loss_full_DP(x, x_decoded_mean):
    ## given z_mean, calculate the new ELBO in DP
#    model = DPParam['model']
    ## transform z_mean as tensor object into a python numpy array
#    z_mean_np = tf.keras.backend.eval(z_mean)
    ## transform the numpy array as XData type requrired by bnpy
#    z_mean_xdata = XData(z_mean_np,dtype='auto')   
     
    ## get sufficient statistics
#    LP = model.calc_local_params(z_mean_xdata)
#    SS = model.get_global_suff_stats(z_mean, LP, doPrecompEntropy=1)
#   elbo = tf.convert_to_tensor(model.calc_evidence(z_mean_xdata, SS, LP), dtype=tf.float32)
    
#    loss_ = alpha*original_dim * objectives.mean_squared_error(x, x_decoded_mean) - elbo
    
#    ELBO = tf.convert_to_tensor(DPParam['elbo'], dtype = tf.float32)
#    loss_= alpha*original_dim * objectives.mean_squared_error(x, x_decoded_mean) - ELBO
#   loss = K.sum(loss_, axis = 0)
#    return loss
    
    
    
    
    

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
        temp = tf.linalg.trace(tf.linalg.solve(W[k], S_k))
        temp2 = tf.matmul(tf.expand_dims((z_k_bar-m[k]), 0), tf.linalg.inv(W[k]))
        temp3 = tf.squeeze(tf.matmul(temp2, tf.expand_dims((z_k_bar-m[k]), -1)))
        if k == 0:
            e = 0.5*N[k]*(v[k]*(temp + temp3))
        else:
            e += 0.5*N[k]*(v[k]*(temp + temp3))

    loss_= alpha*original_dim * objectives.mean_squared_error(K.flatten(x), K.flatten(x_decoded_mean)) - scale * 0.5 * K.sum((z_log_var+1), axis = -1)
    loss_ =  K.sum(loss_, axis=0) + e
    # loss = K.sum(loss_, axis = 0)
    #for i in range(5):
    #    loss_ += N
        
    #return loss_
    return loss_

def cnn_loss(x, x_decoded_mean):
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
        temp = tf.linalg.trace(tf.linalg.solve(W[k], S_k))
        temp2 = tf.matmul(tf.expand_dims((z_k_bar-m[k]), 0), tf.linalg.inv(W[k]))
        temp3 = tf.squeeze(tf.matmul(temp2, tf.expand_dims((z_k_bar-m[k]), -1)))
        if k == 0:
            e = 0.5*N[k]*(v[k]*(temp + temp3))
        else:
            e += 0.5*N[k]*(v[k]*(temp + temp3))

    loss_= alpha*original_dim * objectives.mean_squared_error(K.flatten(x), K.flatten(x_decoded_mean)) - scale * K.sum((z_log_var+1), axis = -1)
    loss_ =  K.sum(loss_, axis=0) + e
    # loss = K.sum(loss_, axis = 0)
    #for i in range(5):
    #    loss_ += N
        
    #return loss_
    return loss_

# dataset = 'reuters10k'
#db = sys.argv[1]
#if db in ['mnist','reuters10k','har']:
#    dataset = db
print ('training on: ' + dataset)
ispretrain = True
batch_size = batchsize
latent_dim = 10
intermediate_dim = [500,500,2000]
#theano.config.floatX='float32'
accuracy=[]
X, Y = load_data(dataset, root_path, flatten)
original_dim,epoches,n_centroid,lr_nn,lr_gmm,decay_n,decay_nn,decay_gmm,alpha,datatype = config_init(dataset)
global DPParam

if flatten:
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
        vade = load_pretrain_weights(vade, root_path, dataset)

else: # use CNN
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
    z_log_var = Dense(latent_dim, name='z_log_var')(x)
    z = Lambda(sampling, output_shape=(latent_dim,))([z_mean, z_log_var])
    # build decoder model
    # for generative model
    latent_inputs = Input(shape=(latent_dim,), name='z_sampling')
    x = Dense(shape[1] * shape[2] * shape[3], activation='relu')(latent_inputs)
    x = Reshape((shape[1], shape[2], shape[3]))(x)

    x = Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    x = UpSampling2D((2, 2))(x)
    x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
    x = UpSampling2D((2, 2))(x)
    x = Conv2D(32, (3, 3), activation='relu')(x)
    x = UpSampling2D((2, 2))(x)
    decoded = Conv2D(1, (3, 3), activation='sigmoid', padding='same')(x)

    # constructing several models
    sample_output = Model(input_img, z, name='encoder')
    decoder = Model(latent_inputs, decoded, name='decoder')

    decoded_for_vade = decoder(sample_output(input_img))
    vade = Model(input_img, decoded_for_vade, name='vade')

    vade.summary()
    sample_output.summary()
    decoder.summary()

    if ispretrain == True:
        sample_output = load_pretrain_cnn_encoder(sample_output, root_path)




num_of_exp = X.shape[0]

num_of_epoch = epoch
num_of_iteration = int(num_of_exp / batch_size)
adam_nn= Adam(lr=lr_nn,epsilon=1e-5, decay = 0.1)

#%%
global newinitname 

if not flatten:
    print("Pretraining VaDE first!")
    vade.compile(optimizer='adadelta', loss='binary_crossentropy')
    vade.fit(X, X, epochs=2, batch_size=batch_size, validation_data=(X, X), shuffle=True)

gamma1 = 1.0
gamma0 = 5.0
for epoch in range(num_of_epoch):    
    id_list = np.arange(num_of_exp)
    np.random.shuffle(id_list)
    #print(id_list)
    #exit(0)
    print("The current epoch is epoch: {}".format(epoch))
    for iteration in range(num_of_iteration):
        indices = id_list[iteration*batch_size:(iteration+1)*batch_size]
        x_batch = X[indices, :]
        
        #print(x_batch)
        # forward pass
        z_batch = sample_output.predict_on_batch(x_batch)
        #print(z_batch)
        
        # to DP
        # DPParam = DP_fit(z_batch)
        # DPParam = np.ones((batch_size))
        # gamma: 'LPMtx' (batch_size, # of cluster)
        # N : 'Nvec' (# of cluster, )
        # m : 'm' (# of cluster, latent_dim)
        # W : 'B' (# of cluster, latent_dim, latent_dim)
        # v: 'nu' (# of cluster) 
        
        # DPParam = DPObj.fit(z_batch)
        
        if epoch ==0 and iteration == 0:
            newinitname = 'randexamples'
            if dataset == 'reuters10k':
                DPObj = DP.DP(output_path = fullOutputPath, initname = newinitname, gamma1=gamma1, gamma0=gamma0, Kmax = Kmax)
            else:
                DPObj = DP.DP(output_path = fullOutputPath, initname = newinitname, gamma1=gamma1, gamma0=gamma0)
            DPParam, newinitname = DPObj.fit(z_batch)
        else:
            # if iteration == (num_of_iteration-1) and epoch !=0:
            if epoch != 0:
                if dataset == 'reuters10k':
                    DPObj = DP.DP(output_path = fullOutputPath, initname = newinitname, gamma1=gamma1, gamma0=gamma0, Kmax = Kmax)
                else:    
                    DPObj = DP.DP(output_path = fullOutputPath, initname = newinitname, gamma1=gamma1, gamma0=gamma0)
                DPParam, newinitname = DPObj.fitWithWarmStart(z_batch, newinitname)
        
        # if iteration == (num_of_iteration-1):
        if not iteration is None:
            trueY = Y[indices]    
            fittedY = DPParam['Y']
            ## get the true number of clusters
            trueCluster, counts = np.unique(trueY, return_counts = True)
            trueK = len(trueCluster)
            print(("The true number of cluster is" + " "+ str(trueK)))
            print("The proportion of image with true cluster in the batch: \n")
            print(counts/len(trueY))
            clusterResult =  clusterEvaluation(trueY, fittedY)
            print("The cluster evaluation result is \n")
            for key,val in clusterResult.items():
                print(key,"=>", val)
            ## get the true cluster and fitted cluster relationship
            dictFitted2True = obtainTrueClusterLabel4AllFittedCluster(trueY, fittedY)
            fittedClusters = dictFitted2True.keys()
            for key in fittedClusters:
                prec = dictFitted2True[key]['prec']
                recall = dictFitted2True[key]['recall']
                trueC =  dictFitted2True[key]['trueCluster']
                print("Precision: {}, Recall: {}, fitted: {}, true: {}".format(prec, recall, key, trueC))
            
        #k = 5
        #DPParam = \
        #{
        #    'LPMtx': np.ones((batch_size, k)),
        #    'Nvec' : np.ones(k),
        #    'm'    : np.ones((k, latent_dim)),
        #    'B'    : np.ones((k, latent_dim, latent_dim)),
        #    'nu'   : np.ones(k)
        #}

        if epoch ==0 and iteration ==0:
            if flatten:
                vade.compile(optimizer=adam_nn, loss=loss)
            else:
                vade.compile(optimizer=adam_nn, loss=cnn_loss)
        for j in range(batch_iter):
            neg_elbo = vade.train_on_batch(x_batch, x_batch)
            print("Iteration: {}-{}, ELBO: {}".format(iteration, j, -neg_elbo))

            
        #if iteration == 5:
        #    exit(0)
        
#%%
################################################
## get z_fit from the encoder and fit with DP model to get all the labels for all training data
z_fit = sample_output.predict(X, batch_size=batch_size)        
fittedY = obtainFittedYFromDP(DPParam, z_fit)
####################################
## Obtain the relationship between fittec class lable and true label, stored in a dictionary
true2Fitted =  obtainDictFromTrueToFittedUsingLabel(Y, fittedY)
## dump true2Fitted using full folder path, whose folder name saves the value of the cmd argument
true2FittedPath = os.path.join(fullOutputPath, 'true2Fitted.json')
# write to a file
pickle.dump(true2Fitted, open(true2FittedPath, 'wb'))
# reads it back
# true2Fitted = pickle.load(open(true2FittedPath, "rb"))
####################################

#%%
################################################
## obtain cluster accuracy
accResult = clusterAccuracy(Y, fittedY)
## this is the overall accuracy
acc = accResult['overallRecall']
## accResult['moreEvaluation'] is the dictionary saves all NMI, ARS, HS, CS, VM
print("The overall recall across all samples: {}".format(acc))
###############################################
## save DP model 
dp_model_path = os.path.join(fullOutputPath, 'dp_model.pkl')
dp_model_param = os.path.join(fullOutputPath, 'DPParam.pkl')
accResult_path = os.path.join(fullOutputPath, 'acc_result.pkl')
fittedY_path = os.path.join(fullOutputPath, 'fittedY.pkl')
joblib.dump(DPParam['model'], dp_model_path) 
joblib.dump(DPParam, dp_model_param) 
joblib.dump(accResult, accResult_path)
joblib.dump(fittedY, fittedY_path)
# m : 'm' (# of cluster, latent_dim)
# W : 'B' (# of cluster, latent_dim, latent_dim)
m = os.path.join(outputPath, 'm.pkl')
W = os.path.join(outputPath, 'W.pkl')
joblib.dump(DPParam['m'], m)
joblib.dump(DPParam['B'], W)
## save neural network model     
# vade.save(os.path.join(outputPath, "vade_DP.hdf5"))
# we should save the model structure and weights seperately.
# serialize model to JSON
# this one is not working for now, don't know how to load self-defined layer
model_json = vade.to_json()
with open(os.path.join(fullOutputPath, "vade_DP_model.json"), "w") as json_file:
    json_file.write(model_json)
# save the weights separately
vade.save_weights(os.path.join(fullOutputPath, "vade_DP_weights.h5"))














