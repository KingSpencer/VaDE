#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May  3 14:16:51 2019

@author: crystal
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May  3 12:06:24 2019

@author: crystal
"""
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
parser.add_argument('-rep', action='store', type=int, dest = 'rep', default=1, help='add replication number as argument')
parser.add_argument('-taskID', action='store', type=int, dest = 'taskID', default=1, help='use taskID to random seed for bnpy') 
 

results = parser.parse_args()
if results.useLocal:
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
randomState = results.taskID 

## DP hyper-parameters
sf = results.sf
gamma0 = results.gamma0
gamma1 = results.gamma1
from OrganizeResultUtil import createOutputFolderName, createFullOutputFileName
import DP as DP
from bnpy.util.AnalyzeDP import * 
from bnpy.data.XData import XData
######################################################################
import pickle
# path for latent z
if dataset == 'mnist':
    pathOfZ = os.path.join(root_path, 'latent_mnist.pkl')
if dataset == 'reuters10k':
    pathOfZ = os.path.join(root_path, 'reuters_utils/latent_reuters10k_supervised.pkl')
aa = pickle.load(open(pathOfZ, 'rb'))
######################################################################
## make full output path 
fullOutputPath = createOutputFolderName(outputPath, Kmax, dataset, epoch, batch_iter, scale, batchsize, rep, sf)
## name log file and write console output to log.txt
logFileName = os. path.join(fullOutputPath, 'log.txt')

if results.logFile:
    sys.stdout = open(logFileName, 'w')

###########################################################
## fit a GMM mixture, replace the DP part
from sklearn.mixture import GaussianMixture 
gmm = GaussianMixture(n_components=10, random_state = RandomState)
gmm.fit(aa['z'])
fittedY = gmm.predict(aa['z'])
Y = aa['y']

true2Fitted =  obtainDictFromTrueToFittedUsingLabel(Y, fittedY)
## dump true2Fitted using full folder path, whose folder name saves the value of the cmd argument
true2FittedPath = os.path.join(fullOutputPath, 'true2Fitted.json')
# write to a file
pickle.dump(true2Fitted, open(true2FittedPath, 'wb'))
# reads it back
# true2Fitted = pickle.load(open(true2FittedPath, "rb"))
################################################
clusterResult =  clusterEvaluation(Y, fittedY)
print("The cluster evaluation result is \n")
for key,val in clusterResult.items():
    print(key,"=>", val)
    
## obtain cluster accuracy
accResult = clusterAccuracy(Y, fittedY)
## this is the overall accuracy
acc = accResult['overallRecall']
## accResult['moreEvaluation'] is the dictionary saves all NMI, ARS, HS, CS, VM
dictFitted2True = obtainTrueClusterLabel4AllFittedCluster(Y, fittedY)
fittedClusters = dictFitted2True.keys()
for key in fittedClusters:
    prec = dictFitted2True[key]['prec']
    recall = dictFitted2True[key]['recall']
    trueC =  dictFitted2True[key]['trueCluster']
    print("Precision: {}, Recall: {}, fitted: {}, true: {}".format(prec, recall, key, trueC))
print("The overall recall across all samples: {}".format(acc))
###############################################
## save DP model 
accResult_path = os.path.join(fullOutputPath, 'acc_result.pkl')
fittedY_path = os.path.join(fullOutputPath, 'fittedY.pkl')
joblib.dump(accResult, accResult_path)
joblib.dump(fittedY, fittedY_path)




