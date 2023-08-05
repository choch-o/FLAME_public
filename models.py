from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, LSTM, GRU, SimpleRNN, Dropout, Conv2D, Lambda, Input, Bidirectional, Flatten, MaxPool2D
from layers import Attention, SelfAttention
from tensorflow.keras import backend as K
import tensorflow as tf


def DeepConvLSTM(input_shape, num_labels, LSTM_units, num_conv_filters, batch_size, F, D):
    """
    The proposed model with CNN layer, LSTM RNN layer and self attention layers.
    Inputs:
    - x_train: required for creating input shape for RNN layer in Keras
    - num_labels: number of output classes (int)
    - LSTM_units: number of RNN units (int)
    - num_conv_filters: number of CNN filters (int)
    - batch_size: number of samples to be processed in each batch
    - F: the attention length (int)
    - D: the length of the output (int)
    Returns
    - model: A Keras model
    """
    # input_shape (sample_size, num_features)
    # num_features for HHAR: 6 (accX, accY, accZ, gyroX, gyroY, gyroZ)
    tf.random.set_seed(42)
    cnn_inputs = Input(shape=(input_shape[0], input_shape[1], 1), batch_size=batch_size, name='rnn_inputs')
    cnn_layer = Conv2D(num_conv_filters, kernel_size = (1, input_shape[1]), strides=(1, 1), padding='valid', data_format="channels_last")
    cnn_out = cnn_layer(cnn_inputs)

    sq_layer = Lambda(lambda x: K.squeeze(x, axis = 2))
    sq_layer_out = sq_layer(cnn_out)

    rnn_layer = LSTM(LSTM_units, return_sequences=True, name='lstm', return_state=True) #return_state=True
    rnn_layer_output, _, _ = rnn_layer(sq_layer_out)

    encoder_output, attention_weights = SelfAttention(size=F, num_hops=D, use_penalization=False, batch_size=batch_size)(rnn_layer_output)
    dense_layer = Dense(num_labels, activation = 'softmax')
    dense_layer_output = dense_layer(encoder_output)

    model = Model(inputs=cnn_inputs, outputs=dense_layer_output)
    # print(model.summary())

    return model

def SimCLR(input_shape, num_labels, model_name="simclr"):
    """
    Create a model for activity recognition
    Reference (TPN model):
        Saeed, A., Ozcelebi, T., & Lukkien, J. (2019). Multi-task self-supervised learning for human activity detection. Proceedings of the ACM on Interactive, Mobile, Wearable and Ubiquitous Technologies, 3(2), 1-30.

    Architecture:
        Input
        -> Conv 1D: 32 filters, 24 kernel_size, relu, L2 regularizer
        -> Dropout: 10%
        -> Conv 1D: 64 filters, 16 kernel_size, relu, L2 regularizer
        -> Dropout: 10%
        -> Conv 1D: 96 filters, 8 kernel_size, relu, L2 regularizer
        -> Dropout: 10%
        -> Global Maximum Pooling 1D
        -> Dense
        -> Dense
        -> Softmax
    
    Parameters:
        input_shape
            the input shape for the model, should be (window_size, num_channels)
    
    Returns:
        model (tf.keras.Model), 
    """

    inputs = tf.keras.Input(shape=input_shape, name='input')
    x = inputs
    x = tf.keras.layers.Conv1D(
            32, 24,
            activation='relu',
            kernel_regularizer=tf.keras.regularizers.l2(l=1e-4)
        )(x)
    x = tf.keras.layers.Dropout(0.1)(x)

    x = tf.keras.layers.Conv1D(
            64, 16,
            activation='relu',
            kernel_regularizer=tf.keras.regularizers.l2(l=1e-4),
        )(x)
    x = tf.keras.layers.Dropout(0.1)(x)
    
    x = tf.keras.layers.Conv1D(
        96, 8,
        activation='relu',
        kernel_regularizer=tf.keras.regularizers.l2(l=1e-4),
        )(x)
    x = tf.keras.layers.Dropout(0.1)(x)
    
    x = tf.keras.layers.GlobalMaxPool1D(data_format='channels_last', name='global_max_pooling1d')(x)

    x = tf.keras.layers.Dense(1024, activation='relu')(x)
    x = tf.keras.layers.Dense(num_labels)(x)
    outputs = tf.keras.layers.Softmax()(x)

    model = tf.keras.Model(inputs=inputs, outputs=outputs, name=model_name)
    return model



def FEMNIST_CNN():
    """
    The proposed model with CNN layer, LSTM RNN layer and self attention layers.
    Inputs:
    - x_train: required for creating input shape for RNN layer in Keras
    - num_labels: number of output classes (int)
    - LSTM_units: number of RNN units (int)
    - num_conv_filters: number of CNN filters (int)
    - batch_size: number of samples to be processed in each batch
    - F: the attention length (int)
    - D: the length of the output (int)
    Returns
    - model: A Keras model
    """
    # input_shape (sample_size, num_features)
    IMAGE_SIZE = 28
    lr, num_classes = 0.0003, 62
    tf.random.set_seed(42)

    cnn_inputs = Input(shape=(IMAGE_SIZE, IMAGE_SIZE, 1))
    conv1 = Conv2D(32, kernel_size=[5, 5], padding='same', activation='relu')(cnn_inputs)
    pool1 = MaxPool2D(pool_size=[2, 2], strides=2)(conv1)
    conv2 = Conv2D(64, kernel_size=[5, 5], padding='same', activation='relu')(pool1)
    pool2 = MaxPool2D(pool_size=[2, 2], strides=2)(conv2)
    pool2_flat = tf.reshape(pool2, [-1, 7 * 7 * 64])
    dense = Dense(2048, activation='relu')(pool2_flat)
    logits = Dense(num_classes)(dense)
    outputs = tf.keras.layers.Softmax()(logits)

    model = tf.keras.Model(inputs=cnn_inputs, outputs=outputs, name='femnist_cnn')
    return model