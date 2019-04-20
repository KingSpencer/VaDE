#%%
import numpy as np
from keras.callbacks import Callback
from keras.optimizers import Adam
from keras.layers import Input, Dense, Lambda, Conv2D, Reshape, UpSampling2D, MaxPooling2D, Flatten
from keras.models import Model
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


os.environ['KMP_DUPLICATE_LIB_OK']='True'

argv = sys.argv[1:]
parser = argparse.ArgumentParser()
parser.add_argument('-bnpyPath', action='store', type = str, dest='bnpyPath', default='/Users/crystal/Documents/bnpy/', \
                    help='path to bnpy code repo')
parser.add_argument('-outputPath', action='store', type = str, dest='outputPath', default='/Users/crystal/Documents/VaDE_results', \
                    help='path to output')
parser.add_argument('-rootPath', action='store', type = str, dest='rootPath', default='/Users/crystal/Documents/VaDE', \
                    help='root path to VaDE')
parser.add_argument('-conv', action='store_true', \
                    help='using convolutional autoencoder or not')


results = parser.parse_args()
bnpyPath = results.bnpyPath
sys.path.append(bnpyPath)
outputPath = results.outputPath
root_path = results.rootPath
sys.path.append(root_path)

flatten = True
if results.conv:
    flatten = False


import DP as DP
from bnpy.util.AnalyzeDP import * 
from bnpy.data.XData import XData


def sampling(args):
    z_mean, z_log_var = args
    epsilon = K.random_normal(shape=(batch_size, latent_dim), mean=0.)
    return z_mean + K.exp(z_log_var / 2) * epsilon

def load_data(dataset, root_path, flatten=True):
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

def load_pretrain_weights(vade, root_path):
    
    path = os.path.join(root_path, 'pretrain_weights')
    filename = 'ae_' + dataset + '.json'
    fullFileName = os.path.join(path, filename)
    ae = model_from_json(open(fullFileName).read())
    # ae = model_from_json(open('pretrain_weights/ae_'+dataset+'.json').read())
    weightFileName = 'ae_' + dataset + '_weights.h5'
    weightFullFileName = os.path.join(path, weightFileName)
    ae.load_weights(weightFullFileName)
    
    #ae.load_weights('pretrain_weights/ae_'+dataset+'_weights.h5')
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
        temp = tf.linalg.trace(tf.matmul(S_k, W[k]))
        temp2 = tf.matmul(tf.expand_dims((z_k_bar-m[k]), 0), W[k])
        temp3 = tf.squeeze(tf.matmul(temp2, tf.expand_dims((z_k_bar-m[k]), -1)))
        if k == 0:
            e = 0.5*N[k]*(v[k]*(temp + temp3))
        else:
            e += 0.5*N[k]*(v[k]*(temp + temp3))

    loss_= alpha*original_dim * objectives.mean_squared_error(K.flatten(x), K.flatten(x_decoded_mean)) # -0.5 * K.sum(z_log_var, axis = -1)
    # loss = K.sum(loss_, axis = 0) + e
    # loss = K.sum(loss_, axis = 0)
    #for i in range(5):
    #    loss_ += N
        
    return loss_

dataset = 'mnist'
#db = sys.argv[1]
#if db in ['mnist','reuters10k','har']:
#    dataset = db
print ('training on: ' + dataset)
ispretrain = True
batch_size = 5000
latent_dim = 10
intermediate_dim = [500,500,2000]
#theano.config.floatX='float32'
accuracy=[]
X, Y = load_data(dataset, root_path, flatten)
original_dim,epoch,n_centroid,lr_nn,lr_gmm,decay_n,decay_nn,decay_gmm,alpha,datatype = config_init(dataset)
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
        vade = load_pretrain_weights(vade, root_path)

else: # use CNN
    input_img = Input(shape=(28, 28, 1))  # adapt this if using `channels_first` image data format
    x = Conv2D(16, (3, 3), activation='relu', padding='same')(input_img)
    x = MaxPooling2D((2, 2), padding='same')(x)
    x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
    x = MaxPooling2D((2, 2), padding='same')(x)
    x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
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

    x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
    x = UpSampling2D((2, 2))(x)
    x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
    x = UpSampling2D((2, 2))(x)
    x = Conv2D(16, (3, 3), activation='relu')(x)
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



num_of_exp = X.shape[0]

num_of_epoch = 10
num_of_iteration = int(num_of_exp / batch_size)
adam_nn= Adam(lr=lr_nn,epsilon=1e-4, decay = 0.01)


#%%
global newinitname 

if not flatten:
    print("Pretraining VaDE first!")
    vade.compile(optimizer='adadelta', loss='binary_crossentropy')
    vade.fit(X, X, epochs=100, batch_size=batch_size, validation_data=(X, X), shuffle=True)


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
            DPObj = DP.DP(initname = newinitname)
            DPParam, newinitname = DPObj.fit(z_batch)
        else:
            if iteration == (num_of_iteration-1) and epoch !=0:
                DPObj = DP.DP(initname = newinitname)
                DPParam, newinitname = DPObj.fitWithWarmStart(z_batch, newinitname)
        
        if iteration == (num_of_iteration-1):
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
            vade.compile(optimizer=adam_nn, loss=loss)
        for j in range(10):
            neg_elbo = vade.train_on_batch(x_batch, x_batch)
            print("Iteration: {}-{}, ELBO: {}".format(iteration, j, -neg_elbo))

            
        #if iteration == 5:
        #    exit(0)
        
#%%
################################################
## get z_fit from the encoder and fit with DP model to get all the labels for all training data
z_fit = sample_output.predict(X, batch_size=batch_size)        
fittedY = obtainFittedYFromDP(DPParam, z_fit)
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
dp_model_path = os.path.join(outputPath, 'dp_model.pkl')
dp_model_param = os.path.join(outputPath, 'DPParam.pkl')
accResult_path = os.path.join(outputPath, 'acc_result.pkl')
fittedY_path = os.path.join(outputPath, 'fittedY.pkl')
joblib.dump(DPParam['model'], dp_model_path) 
joblib.dump(DPParam, dp_model_param) 
joblib.dump(accResult, accResult_path)
joblib.dump(fittedY, fittedY_path)

## save neural network model     
vade.save(os.path.join(outputPath, "vade_DP.hdf5"))














