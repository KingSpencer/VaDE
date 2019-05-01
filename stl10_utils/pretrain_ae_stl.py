from keras.optimizers import Adam
from keras.optimizers import Nadam
from keras.layers import Input, Dense, Activation
from keras.models import Model
import pickle
import os


def get_ae(original_dim=2048, latent_dim=10, intermediate_dim=[500,500,2000]):
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

def get_ae_supervised(original_dim=2048, latent_dim=10, intermediate_dim=[500,500,2000]):
    x = Input(shape=(original_dim, ))
    h = Dense(intermediate_dim[0], activation='relu')(x)
    h = Dense(intermediate_dim[1], activation='relu')(h)
    h = Dense(intermediate_dim[2], activation='relu')(h)
    z = Dense(latent_dim)(h)
    y_pred = Activation('softmax', name='prediction_out')(z)
    h_decoded = Dense(intermediate_dim[-1], activation='relu')(z)
    h_decoded = Dense(intermediate_dim[-2], activation='relu')(h_decoded)
    h_decoded = Dense(intermediate_dim[-3], activation='relu')(h_decoded)
    x_decoded = Dense(original_dim, activation='linear', name='decoded_out')(h_decoded)


    ae = Model(x, [x_decoded, y_pred])
    ae.compile(optimizer='adadelta', loss=['mse','categorical_crossentropy'], loss_weights=[1,1], metrics={'decoded_out':'mae', 'prediction_out':'acc'})
    return ae

if __name__ == '__main__':
    batch_size = 128
    with open('../dataset/stl10/X.pkl', 'rb') as f:
        X = pickle.load(f)
    with open('../dataset/stl10/Y.pkl', 'rb') as f:
        Y = pickle.load(f)
    ae = get_ae_supervised()
    ae.fit(X, [X, Y], epochs=80, batch_size=batch_size)
    model_json = ae.to_json()
    output_path = '../pretrain_weights'
    with open(os.path.join(output_path, "ae_stl10_supervised.json"), "w") as json_file:
        json_file.write(model_json)
    ae.save_weights(os.path.join(output_path, "ae_stl10_supervised_weights.h5"))


