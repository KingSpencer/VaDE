from keras.models import load_model, model_from_json
import os
import gzip
import sys
from six.moves import cPickle
import numpy as np
from scipy.misc import imsave

if __name__ == "__main__":
    path = '/home/zifeng/Research/DPVAE/pretrain_weights'
    filename = 'ae_mnist_supervised.json'
    fullFileName = os.path.join(path, filename)
    ae = model_from_json(open(fullFileName).read())
    # ae = model_from_json(open('pretrain_weights/ae_'+dataset+'.json').read())
    weightFileName = 'ae_mnist_supervised_weights.h5'
    weightFullFileName = os.path.join(path, weightFileName)
    ae.load_weights(weightFullFileName)
    #ae.summary()
    #ae.compile(optimizer='adam', loss=None)


    dataset = 'mnist'
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

    [img_sample, y] = ae.predict(X[0:2])
    img_sample *= 255
    img_sample = img_sample.astype(np.uint8)
    #print(img_sample[0])
    imsave('sample.png', img_sample[0].reshape(28,28))