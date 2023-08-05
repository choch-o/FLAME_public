import pickle as pkl
import os, time, pdb, argparse

def train_FL_local(epochs=10):

    import tensorflow as tf
    from model import DeepConvLSTM

    class TimeLogger(tf.keras.callbacks.Callback):
        def on_epoch_end(self, epoch, logs=None):
            global epoch_times
            epoch_times.append(time.time())

    option = {
        'seq_len': 150,
        'input_dim': 6,
        'devices': ['forearm', 'thigh', 'head', 'chest', 'upperarm', 'waist', 'shin'],
        'users': ['S1', 'S2', 'S3', 'S4', 'S5', 'S6', 'S7', 'S8', 'S9', 'S10', 'S11', 'S12', 'S13','S14', 'S15'],
        'classes': ['climbingdown', 'climbingup', 'jumping', 'lying', 'running', 'sitting', 'standing', 'walking'],
        'num_class': 8,
        'dataset_path': './',
        'dataset_name': 'realworld-3.0-0.0.dat',
        'batch_size': 16,

        'lstm_units': 32,
        'cnn_filters': 3,
        'num_lstm_layers': 1,
        'patience': 20,
        'F': 32,
        'D': 10
    }
    input_shape = (option['seq_len'], option['input_dim'])

    batch_size = 32
    # epochs = 20
    
    f = open('data.pickle', 'rb')
    data = pkl.load(f)
    X_train = data["X_train"]
    y_train = data["y_train"]

    X_train = X_train[:192]
    y_train = y_train[:192]    

    epoch_times = [time.time()]

    def log_time():
        epoch_times.append(time.time())
    
    callbacks = [tf.keras.callbacks.LambdaCallback(on_epoch_end=lambda epoch, logs: log_time())]

    model = DeepConvLSTM(input_shape=input_shape, num_labels=option['num_class'],
                                            LSTM_units=option['lstm_units'],
                                            num_conv_filters=option['cnn_filters'],
                                            batch_size=batch_size, F=option['F'], D=option['D'])

    model.compile('adam', loss='categorical_crossentropy', metrics=['accuracy'])
    model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=1, callbacks=callbacks)
    model.save("deepconvlstm.h5")
    return epoch_times


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Training params')
    parser.add_argument('--outfile', type=str, required=True)
    parser.add_argument('--epochs', type=int)
    args = parser.parse_args()

    epoch_times = train_FL_local(args.epochs)
    output_dict = {}
    output_dict['epoch_times'] = epoch_times

    f = open(args.outfile + '.pickle', "wb")
    pkl.dump(output_dict, f)
    f.close()


