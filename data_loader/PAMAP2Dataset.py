from os import path

import load_data
import tensorflow as tf
import numpy as np

import options

opt = options.PAMAP2Opt
SEED = 913

class PAMAP2Dataset():
    # def __init__(self, dataset_path, dataset_name):
    #     tf_dataset_full = load_data.Data(dataset_path, dataset_name)
    #     self.data = tf_dataset_full

    def __init__(self, dataset_path, position, user, resample=False, test_devices='all'):
        # if path.exists(dataset_path + f"{position}-{user}-")
        dataset_path = path.join(dataset_path, "pamap2_data")
        data = load_data.Data(dataset_path, position, user, resample, test_devices)
        self.device = position
        self.ds_train = data.get_train_data()
        self.ds_test = data.get_test_data()

    def to_np_data(self, dataset, batch_size):
        X = []
        y = []

        for xx, yy in dataset:
            X.append(xx)
            y.append(yy)
        X = np.array(X, dtype=np.float)
        y = np.array(y)
        num_labels = opt['num_class']
        y_one_hot = np.zeros((y.size, num_labels))
        if y.size > 0:
            y_one_hot[np.arange(y.size), y] = 1

            # np_test = (np_test_x, np_test_y)
            X = np.expand_dims(X, axis=3)

            trailing_samples = X.shape[0] % batch_size

            if trailing_samples != 0:
                X = X[0:-trailing_samples]
                y_one_hot = y_one_hot[0:-trailing_samples]

        return X, y_one_hot

    def get_dataset(self, batch_size, take=1.0, partition_idx=0, time_aligned='aligned'):
        train, test = self.ds_train, self.ds_test

        training_full_size = len(train)
        train_samples_count = int(training_full_size * take)

        # Keep class balance
        def get_dataset_by_label(ds, label):
            return ds.filter(lambda x, y: y == label)  # 0-11
        datasets_by_class = [get_dataset_by_label(train, label + 1) for label in
                             range(len(opt['classes']))]

        tf_train_split = None

        device_id = opt['devices'].index(self.device)
        seed = SEED if time_aligned == 'aligned' else SEED + device_id

        for ds in datasets_by_class:
            tf.random.set_seed(seed)
            ds_full_size = len(list(ds.as_numpy_iterator()))
            ds_samples_count = int(ds_full_size * take)
            ds_train = ds.shuffle(buffer_size=5000000, seed=seed, reshuffle_each_iteration=False) \
                .skip(ds_samples_count * partition_idx) \
                .take(ds_samples_count).cache()

            tf_train_split = ds_train if tf_train_split is None else tf_train_split.concatenate(ds_train)

        X_train, y_train = self.to_np_data(tf_train_split, batch_size)
        X_val, y_val = (np.array([]), np.array([]))
        X_test, y_test = self.to_np_data(test, batch_size)

        return (X_train, y_train), (X_val, y_val), (X_test, y_test)

    def get_dataset_by_class(self, batch_size, take=1.0, partition_idx=0, class_label=0, time_aligned='aligned'):
        train, test = self.ds_train, self.ds_test

        # Keep class balance
        def get_dataset_by_label(ds, label):
            return ds.filter(lambda x, y: y == label)  # 0-7

        ds_class_train = get_dataset_by_label(train, class_label)
        ds_test = get_dataset_by_label(test, class_label)

        train_samples_count = len(list(ds_class_train.as_numpy_iterator()))
        ds_train_samples_count = int(train_samples_count * take)
        # ~184 for opportunity

        device_id = opt['devices'].index(self.device)
        seed = SEED if time_aligned == 'aligned' else SEED + device_id
        tf.random.set_seed(seed)
        ds_train = ds_class_train.shuffle(buffer_size=5000000, seed=seed, reshuffle_each_iteration=False) \
            .skip(ds_train_samples_count * partition_idx) \
            .take(ds_train_samples_count).cache()

        return ds_train, ds_test
