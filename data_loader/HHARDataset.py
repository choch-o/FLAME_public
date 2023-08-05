import sys, os
import time
import itertools
import pandas as pd
import tensorflow as tf
import numpy as np

sys.path.append('..')
import options

opt = options.HHAROpt
OVERLAPPING_WIN_LEN = opt['seq_len'] // 2
WIN_LEN = opt['seq_len']


class HHARDataset():
    # load static files
    def __init__(self, dataset_path, model, user, resample=False, test_devices='trained'):
        """
        Args:
            file_path (string): Path to the csv file with annotations.
            transform (callable, optional): Optional transform to be applied
                on a sample.
            gt: condition on action
            user: condition on user
            model: condition on model
            device: condition on device instance
            complementary: is it complementary dataset for given conditions? (used for "multi" case)
        """
        st = time.time()
        self.user = user
        self.device = model
        # self.device = device

        self.df = pd.read_csv(os.path.join(dataset_path, f"{user}_{model}.csv"))

        self.X = []
        self.y = []
        self.y_one_hot = []

        if user:
            self.df = self.df[self.df['User'] == user]
        if model:
            self.df = self.df[self.df['Model'] == model]
        # if device:
        #     self.df = self.df[self.df['Device'] == device]
        # if gt:
        #     self.df = self.df[self.df['gt'] == gt]

        # self.transform = transform
        ppt = time.time()

        self.preprocessing()
        print('Loading data done with rows:{:d}\tPreprocessing:{:f}\tTotal Time:{:f}'.format(len(self.df.index),
                                                                                             time.time() - ppt,
                                                                                             time.time() - st))

    def preprocessing(self):
        features = []
        class_labels = []

        users = {self.user}  # bracket is required to append a tuple
        models = {self.device}

        for idx in range(max(len(self.df) // OVERLAPPING_WIN_LEN - 1, 0)):
            # user = self.df.iloc[idx * OVERLAPPING_WIN_LEN, 9]  # starting point, column index
            # model = self.df.iloc[idx * OVERLAPPING_WIN_LEN, 10]
            # device = self.df.iloc[idx * OVERLAPPING_WIN_LEN, 11]
            class_label = self.df.iloc[idx * OVERLAPPING_WIN_LEN, 13]
            # domain_label = -1

            # Save a feature: AccelX, AccelY, AccelZ, GyroX, GyroY, GyroZ of window size (overlap * 2; =256)
            feature = self.df.iloc[idx * OVERLAPPING_WIN_LEN:(idx + 2) * OVERLAPPING_WIN_LEN, 3:9].values  # (256, 6)
            # feature = feature.T # (6, 256)

            features.append(feature)
            class_labels.append(self.class_to_number(class_label))

        features = np.array(features, dtype=np.float)
        class_labels = np.array(class_labels)

        self.X = features
        self.y = class_labels
        one_hot_labels = np.zeros((class_labels.size, opt['num_class']))
        one_hot_labels[np.arange(class_labels.size), class_labels] = 1
        self.y_one_hot = one_hot_labels

        # dataset = tf.data.Dataset.from_tensor_slices((features, one_hot_labels))
        dataset = tf.data.Dataset.from_tensor_slices((self.X, self.y))
        self.dataset = dataset

    def __len__(self):
        # return max(len(self.df) // OVERLAPPING_WIN_LEN - 1, 0)
        return len(self.dataset)

    def get_num_domains(self):
        return self.num_domains

    def get_raw_dataset(self):
        return self.X, self.y_one_hot

    def get_dataset(self, batch_size, take, partition_idx, time_aligned):
        # HHAR dataset doesn't have train-test partition in the dataset
        ds = tf.data.Dataset.from_tensor_slices((self.X, self.y))

        # Keep class balance
        def get_dataset_by_label(dataset, label):
            return dataset.filter(lambda x, y: y == label)  # 0-5

        datasets_by_class = [get_dataset_by_label(ds, label) for label in
                             range(len(opt['classes']))]

        tf_train_split = None
        tf_test_split = None

        device_id = opt['models'].index(self.device)
        seed = 42 if time_aligned == 'aligned' else 42 + device_id

        for ds in datasets_by_class:
            ds_full_size = len(list(ds.as_numpy_iterator()))
            ds_samples_count = int(ds_full_size * take)

            ds = ds.shuffle(buffer_size=5000000, seed=seed, reshuffle_each_iteration=False)
            ds_train = ds.take(int(ds_full_size * 0.8))
            ds_test = ds.skip(int(ds_full_size * 0.8))

            ds_train = ds_train.skip(ds_samples_count * partition_idx)\
                .take(ds_samples_count).cache()

            tf_train_split = ds_train if tf_train_split is None else tf_train_split.concatenate(ds_train)
            tf_test_split = ds_test if tf_test_split is None else tf_test_split.concatenate(ds_test)

        tf_train_split = tf_train_split.shuffle(buffer_size=5000000, seed=seed, reshuffle_each_iteration=False)
        tf_test_split = tf_test_split.shuffle(buffer_size=5000000, seed=seed, reshuffle_each_iteration=False)
        X_train, y_train = self.to_np_data(tf_train_split, batch_size)
        X_test, y_test = self.to_np_data(tf_test_split, batch_size)

        # return (X_train, y_train), (X_val, y_val), (X_test, y_test)
        return (X_train, y_train), (np.array([]), np.array([])), (X_test, y_test)

    def get_dataset_by_class(self, batch_size, take=1.0, partition_idx=0, class_label=0, time_aligned='aligned'):
        ds = tf.data.Dataset.from_tensor_slices((self.X, self.y))

        # Keep class balance
        def get_dataset_by_label(dataset, label):
            return dataset.filter(lambda x, y: y == label)  # 0-5

        device_id = opt['models'].index(self.device)
        seed = 42 if time_aligned == 'aligned' else 42 + device_id
        tf.random.set_seed(seed)

        ds = get_dataset_by_label(ds, class_label)
        ds_full_size = len(list(ds.as_numpy_iterator()))
        ds_samples_count = int(ds_full_size * take)

        ds = ds.shuffle(buffer_size=5000000, seed=seed, reshuffle_each_iteration=False)
        ds_train = ds.take(int(ds_full_size * 0.8))
        ds_test = ds.skip(int(ds_full_size * 0.8))

        ds_train = ds_train.skip(ds_samples_count * partition_idx) \
            .take(ds_samples_count).cache()

        # ds_class_size = len(list(ds_class.as_numpy_iterator()))
        # ds_class_samples_count = int(ds_class_size * take)
        #
        #
        # ds = ds_class.shuffle(buffer_size=5000000, seed=seed, reshuffle_each_iteration=False)
        # ds_size = len(list(ds.as_numpy_iterator()))
        # ds_train = ds.take(int(ds_size * 0.8))
        # ds_test = ds.skip(int(ds_size * 0.8))
        #
        # ds_train = ds_train.skip(ds_class_samples_count * partition_idx) \
        #     .take(ds_class_samples_count).cache()
        # #
        return ds_train, ds_test


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

    def class_to_number(self, label):
        dic = {'bike': 0,
               'sit': 1,
               'stand': 2,
               'walk': 3,
               'stairsup': 4,
               'stairsdown': 5,
               'null': 6}
        return dic[label]

    def __getitem__(self, idx):
        if isinstance(idx, tf.Tensor):
            idx = idx.item()

        return self.dataset[idx]


if __name__ == '__main__':
    pass