#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 10 10:05:11 2019

@author: crystal
"""

## load the three ZLatent representation for tsne plot
from sklearn.externals import joblib ## replacement of pickle to carry large numpy arrays
import pickle
from sklearn.manifold import TSNE
%matplotlib inline
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
import time
import numpy as np

## the first row is always zero and should be discarded
firstBatchZ = joblib.load( open('/home/tingting/Documents/newCluster/firstBatch/output/Kmax30firstBatchepoch15batch_iter3scale0.05bs2104rep1sf0.1/tsneZ.pkl', 'rb'))
secondBatchZ = joblib.load(open('/home/tingting/Documents/newCluster/secondBatch/output/Kmax30secondBatchepoch15batch_iter3scale0.05bs2104rep1sf0.1/tsneZ.pkl', 'rb'))
fittedY = joblib.load(open('/home/tingting/Documents/newCluster/firstBatch/output/Kmax30firstBatchepoch15batch_iter3scale0.05bs2104rep1sf0.1/fittedY.pkl', 'rb'))

## get the original data size of Z
## in the firstBatch each Z should be of dimension 14674 * 10
## in the secondBatch each Z should be of dimension 18167* 10
firstZ = firstBatchZ[1: firstBatchZ.shape[0] , :]
secondZ = secondBatchZ[1: secondBatchZ.shape[0] , :]

lastFirstZ = firstZ[ (firstZ.shape[0]-14674+1) :firstZ.shape[0], :]
lastSecondZ = secondZ[ (secondZ.shape[0]-18167+1) :secondZ.shape[0], :]


time_start = time.time()
tsne = TSNE(n_components=2, verbose=1, perplexity=40, n_iter=300)
tsne_results = tsne.fit_transform(lastFirstZ)
print('t-SNE done! Time elapsed: {} seconds'.format(time.time()-time_start))

result = dict()
result['tsne-2d-one'] = tsne_results[:,0]
result['tsne-2d-two'] = tsne_results[:,1]
totalY = len(fittedY)
result['fittedy'] = fittedY[ (totalY-14674+1):totalY]

## map fittedY to true Y as legend
acc_result = joblib.load(open("/home/tingting/Documents/newCluster/firstBatch/output/Kmax30firstBatchepoch15batch_iter3scale0.05bs2104rep1sf0.1/acc_result.pkl", 'rb'))
trueToFitted = acc_result['match']
fittedToTrue = dict()
## from Fitted Values to true values
keys = trueToFitted.keys()
for key in keys:
    values = trueToFitted[key]
    for value in values:
        fittedToTrue[value] = key
## fitted cluster 6 is the mixture cluster in our fit
import matplotlib.pyplot as plt

result['y'] = [None]*len(result['fittedy'])
for i in range(len(result['y'])):
    if result['fittedy'][i] == 6:
        result['y'][i] = 'mixture'
    else:
        result['y'][i] = fittedToTrue[result['fittedy'][i]]


plt.figure(figsize=(16,10))
ax1 = sns.scatterplot(
    x="tsne-2d-one", y="tsne-2d-two",
    hue="y",
    palette=sns.color_palette("Paired", 10),
    data=result,
    legend="full",
    alpha=1.0
)
ax1.legend(fancybox=True, framealpha=0.1)
plt.setp(ax1.get_legend().get_texts(), fontsize='36') # for legend text
plt.setp(ax1.get_legend().get_title(), fontsize='36')
plt.show()
########################################################
## draw tsne for second batch
time_start = time.time()
tsne = TSNE(n_components=2, verbose=1, perplexity=40, n_iter=300)
tsne_results = tsne.fit_transform(lastSecondZ)
print('t-SNE done! Time elapsed: {} seconds'.format(time.time()-time_start))

fittedY = joblib.load(open('/home/tingting/Documents/newCluster/secondBatch/output/Kmax30secondBatchepoch15batch_iter3scale0.05bs2104rep1sf0.1/fittedY.pkl', 'rb'))

acc_result = joblib.load(open("/home/tingting/Documents/newCluster/secondBatch/output/Kmax30secondBatchepoch15batch_iter3scale0.05bs2104rep1sf0.1/acc_result.pkl", 'rb'))
trueToFitted = acc_result['match']
fittedToTrue = dict()
## from Fitted Values to true values
keys = trueToFitted.keys()
for key in keys:
    values = trueToFitted[key]
    for value in values:
        fittedToTrue[value] = key
## fitted cluster 6 is the mixture cluster in our fit
result = dict()
result['tsne-2d-one'] = tsne_results[:,0]
result['tsne-2d-two'] = tsne_results[:,1]
totalY = len(fittedY)
result['fittedy'] = fittedY[ (totalY-18167+1):totalY]        
        

result['y'] = [None]*len(result['fittedy'])
for i in range(len(result['y'])):
    if result['fittedy'][i] == 9:
        result['y'][i] = 'mixture'
    else:
        result['y'][i] = fittedToTrue[result['fittedy'][i]]





plt.figure(figsize=(16,10))
ax2 = sns.scatterplot(
    x="tsne-2d-one", y="tsne-2d-two",
    hue="y",
    palette=sns.color_palette("Paired", 11),
    data=result,
    legend="full",
    alpha=1.0
)
ax2.legend(fancybox=True, framealpha=0.1)
plt.setp(ax2.get_legend().get_texts(), fontsize='36') # for legend text
plt.setp(ax2.get_legend().get_title(), fontsize='36')
plt.show()

