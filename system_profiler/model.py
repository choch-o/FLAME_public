from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, LSTM, GRU, SimpleRNN, Dropout, Conv2D, Lambda, Input, Bidirectional, Flatten
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