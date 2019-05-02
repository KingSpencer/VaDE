from keras.optimizers import Adam
from keras.optimizers import Nadam
from keras.layers import Input, Dense, Activation
from keras.models import Model
import pickle
import os
import scipy.io as scio
from sklearn.preprocessing import LabelEncoder
from keras.utils import np_utils

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
    y_pred = Dense(4, activation = 'softmax', name='prediction_out')(z)
    h_decoded = Dense(intermediate_dim[-1], activation='relu')(z)
    h_decoded = Dense(intermediate_dim[-2], activation='relu')(h_decoded)
    h_decoded = Dense(intermediate_dim[-3], activation='relu')(h_decoded)
    x_decoded = Dense(original_dim, activation='linear', name='decoded_out')(h_decoded)
    ae = Model(x, [x_decoded, y_pred])
    ae.compile(optimizer='adadelta', loss=['mse','categorical_crossentropy'], loss_weights=[1,1], metrics={'decoded_out':'mae', 'prediction_out':'acc'})
    ae.summary()
    return ae

if __name__ == '__main__':
    batch_size = 128
    ## load X and Y from Reuters10k 
    path = '../dataset/reuters10k'
    # path = '/Users/crystal/Documents/VaDE/dataset/reuters10k'
    data=scio.loadmat(os.path.join(path,'reuters10k.mat'))
    X = data['X']
    Y = data['Y'].squeeze()
    ## change Y to one hot encoding
    encoder = LabelEncoder()
    encoder.fit(Y)
    encoded_Y = encoder.transform(Y)
    # convert integers to dummy variables (i.e. one hot encoded)
    dummy_y = np_utils.to_categorical(encoded_Y)
    ae = get_ae_supervised()
    ae.fit(X, [X, dummy_y], epochs=50, batch_size=batch_size)
    model_json = ae.to_json()
    output_path = '../pretrain_weights'
    with open(os.path.join(output_path, "ae_reuters10k_supervised.json"), "w") as json_file:
        json_file.write(model_json)
    ae.save_weights(os.path.join(output_path, "ae_reuters10k_supervised_weights.h5"))


