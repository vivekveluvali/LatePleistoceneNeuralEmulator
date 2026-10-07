import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, Model

## Set up functions
def set_seeds(seed=42):
    # set random see for tensorflow, keras, numpy etc.
    tf.keras.utils.set_random_seed(seed)
    # only relevant if calculation is done on GPU
    tf.config.experimental.enable_op_determinism()

def create_windows(data, window_size, rollout_length):
    X_list, y_list, Insol_rollout_list = [], [], []
    for i in range(len(data) - window_size - rollout_length):
        X_list.append(data[i:i + window_size,:])
        y_list.append(data[i + window_size: i + window_size + rollout_length, :-1])
        Insol_rollout_list.append(data[i + window_size: i + window_size + rollout_length, -1])

    return np.array(X_list), np.array(y_list), np.array(Insol_rollout_list)

## Function encoding actual model architecture
def CNN_model(window_size, n_features, params):
    """
    Convolutional Neural Network architecture to be used for the SR-LPNE model
    --------------------------------------------------------------------------

    window_size  - input window size
    n_features   - number of input variables (here set to 4 for X,Y,Z,Insol)
    params       - hyperparameter dict with keys 'conv filters', 'conv kernel',
                   'dense unit', 'dropout rate' (see run_config.DATASETS)

    """
    # layer names are pinned explicitly so save_weights()/load_weights() - which
    # match variables by layer name - keep working no matter how many times this
    # function has already been called in the current kernel session
    inp = layers.Input(shape=(window_size, n_features))

    x = layers.GaussianNoise(0.05, name="gaussian_noise")(inp)

    x = layers.Conv1D(filters = params['conv filters'],
                      kernel_size = params['conv kernel'],
                      activation = 'swish',
                      padding = "causal",
                      name = "conv1d")(x)

    x = layers.LayerNormalization(name="layer_normalization")(x)

    # extract last tensor from convolutional layer output
    x = layers.Lambda(lambda x: x[:, -1, :], name="lambda")(x)

    x = layers.Dense(units = params['dense unit'],
                     activation="swish",
                     name = "dense")(x)

    # dropout layer to reduce overfitting; dropout rate is a hyperparameter
    x = layers.Dropout(rate=params['dropout rate'], name="dropout")(x)

    # output layer, returns the vector [\Delta_X,\Delta_Y,\Delta_Z]
    out = layers.Dense(units = 3,
                       activation = 'linear',
                       kernel_initializer = 'zeros',
                       bias_initializer = 'zeros',
                       use_bias=True,
                       name = "dense_1")(x)

    # physical constraint: scaling by maximum incerement in data
    # WARNING: NEEDS tO BE ADJUSTED FOR DIFFERENT DATA SETS
    #out = out * tf.constant([0.47893476,0.78733971,0.21167531], dtype=tf.float32)

    return Model(inputs = inp,outputs = out)

def TCN_model():
    print('not implemented yet')
    return None

def reservoircomputer_model():
    print('not implemented yet')
    return None