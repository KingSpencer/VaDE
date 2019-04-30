from keras.applications.resnet50 import ResNet50
'''
def ResNet50(include_top=True,
             weights='imagenet',
             input_tensor=None,
             input_shape=None,
             pooling=None,
             classes=1000,
             **kwargs):
        """Instantiates the ResNet50 architecture.
    Optionally loads weights pre-trained on ImageNet.
    Note that the data format convention used by the model is
    the one specified in your Keras config at `~/.keras/keras.json`.
    # Arguments
        include_top: whether to include the fully-connected
            layer at the top of the network.
        weights: one of `None` (random initialization),
              'imagenet' (pre-training on ImageNet),
              or the path to the weights file to be loaded.
        input_tensor: optional Keras tensor (i.e. output of `layers.Input()`)
            to use as image input for the model.
        input_shape: optional shape tuple, only to be specified
            if `include_top` is False (otherwise the input shape
            has to be `(224, 224, 3)` (with `channels_last` data format)
            or `(3, 224, 224)` (with `channels_first` data format).
            It should have exactly 3 inputs channels,
            and width and height should be no smaller than 32.
            E.g. `(200, 200, 3)` would be one valid value.
        pooling: Optional pooling mode for feature extraction
            when `include_top` is `False`.
            - `None` means that the output of the model will be
                the 4D tensor output of the
                last convolutional block.
            - `avg` means that global average pooling
                will be applied to the output of the
                last convolutional block, and thus
                the output of the model will be a 2D tensor.
            - `max` means that global max pooling will
                be applied.
        classes: optional number of classes to classify images
            into, only to be specified if `include_top` is True, and
            if no `weights` argument is specified.
    # Returns
        A Keras model instance.
'''
from download_stl import load_dataset
import numpy as np
import os
import pickle

if __name__ == "__main__":
    STL_shape = (96, 96, 3)
    resnet = ResNet50(include_top=False, weights='imagenet', input_tensor=None, input_shape=STL_shape, pooling='avg')
    (x_train, y_train), (x_test, y_test) = load_dataset()
    X = np.vstack([x_train, x_test])
    Y = np.vstack([y_train, y_test])
    ###
    X_feature = resnet.predict(X, verbose=1)
    print(X_feature.shape)
    # save it
    root_path = '../dataset/stl10'
    if not os.path.exists(root_path):
        os.mkdir(root_path)
    with open(os.path.join(root_path, 'X.pkl'), 'wb') as f:
        pickle.dump(X_feature, f)
    with open(os.path.join(root_path, 'Y.pkl'), 'wb') as f:
        pickle.dump(Y, f)


