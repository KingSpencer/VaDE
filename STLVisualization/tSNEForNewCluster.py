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

## the first row is always zero and should be discarded
firstBatchZ = joblib.load( open('/Users/crystal/Documents/newCluster/firstBatch/output/Kmax30firstBatchepoch15batch_iter3scale0.05bs2104rep1sf0.1/tsneZ.pkl', 'rb'))
secondBatchZ = joblib.load(open('/Users/crystal/Documents/newCluster/secondBatch/output/Kmax30secondBatchepoch15batch_iter3scale0.05bs2104rep1sf0.1/tsneZ.pkl', 'rb'))
fittedY = joblib.load(open('/Users/crystal/Documents/newCluster/firstBatch/output/Kmax30firstBatchepoch15batch_iter3scale0.05bs2104rep1sf0.1/fittedY.pkl', 'rb'))

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
result['y'] = fittedY[ (totalY-14674+1):totalY]

plt.figure(figsize=(16,10))
sns.scatterplot(
    x="tsne-2d-one", y="tsne-2d-two",
    hue="y",
    palette=sns.color_palette("hls", 10),
    data=result,
    legend="full",
    alpha=0.3
)

########################################################
## draw tsne for second batch
ßtime_start = time.time()
tsne = TSNE(n_components=2, verbose=1, perplexity=40, n_iter=300)
tsne_results = tsne.fit_transform(lastSecondZ)
print('t-SNE done! Time elapsed: {} seconds'.format(time.time()-time_start))

fittedY = joblib.load(open('/Users/crystal/Documents/newCluster/secondBatch/output/Kmax30secondBatchepoch15batch_iter3scale0.05bs2104rep1sf0.1/fittedY.pkl', 'rb'))

result = dict()
result['tsne-2d-one'] = tsne_results[:,0]
result['tsne-2d-two'] = tsne_results[:,1]
totalY = len(fittedY)
result['y'] = fittedY[ (totalY-18167+1):totalY]

plt.figure(figsize=(16,10))
sns.scatterplot(
    x="tsne-2d-one", y="tsne-2d-two",
    hue="y",
    palette=sns.color_palette("hls", 11),
    data=result,
    legend="full",
    alpha=0.3
)


