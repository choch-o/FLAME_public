
import numpy as np
import pickle
import os
import pdb
import random
from collections import Counter
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
import tensorflow as tf
import argparse

import options


class DictMap(dict):
    """
    Supports python dictionary dot notation with existing bracket notation
    Example:
    m = DictMap({'first_name': 'Eduardo'}, last_name='Pool', age=24, sports=['Soccer'])
    m.first_name == m['first_name']
    """

    def __init__(self, *args, **kwargs):
        super(DictMap, self).__init__(*args, **kwargs)
        for arg in args:
            if isinstance(arg, dict):
                for k, v in arg.items():
                    self[k] = v

        if kwargs:
            for k, v in kwargs.items():
                self[k] = v

    def __getattr__(self, attr):
        return self.get(attr)

    def __setattr__(self, key, value):
        self.__setitem__(key, value)

    def __setitem__(self, key, value):
        super(DictMap, self).__setitem__(key, value)
        self.__dict__.update({key: value})

    def __delattr__(self, item):
        self.__delitem__(item)

    def __delitem__(self, key):
        super(DictMap, self).__delitem__(key)
        del self.__dict__[key]


class Data(object):
    """
    Creates tf.data.Dataset object for each device
    Note: words device and position are used interchangeably
    """

    def __init__(self, path, position, user, resample, test_devices='all'):
    # def __init__(self, path='/mnt/data/gsl', dataset_name='realworld-3.0-0.0.dat', load_path=None):
        super(Data, self).__init__()
        self.path = path
        self.train_set_name = f"{position}-{user}-{resample}.pickle"  # dataset_name
        self.norm = 30.0  # max value
        filename = os.path.join(self.path, self.train_set_name)

        if test_devices != 'trained' and test_devices != 'all':
            option = options.RealWorldOpt

        # Load train dataset
        if os.path.exists(filename):
            obj = self.load_dataset(filename)
            train_tmp = obj['train']
            self.ds_train = tf.data.Dataset.from_tensor_slices(train_tmp)

            if test_devices == 'trained':  # Test on trained device data
                test_tmp = obj['test']
                self.ds_test = tf.data.Dataset.from_tensor_slices(test_tmp)
            elif test_devices == 'all':  # Test on all devices data
                filename = os.path.join(self.path, f"{user}-test.pickle")
                if os.path.exists(filename):
                    obj = self.load_dataset(filename)
                    test_tmp = obj['test']
                    self.ds_test = tf.data.Dataset.from_tensor_slices(test_tmp)
            elif test_devices in option['users']:  # For LOUO, test on specific user, e.g., test_devices == 'S1'
                filename = os.path.join(self.path, f"{test_devices}-test.pickle")
                print("users test_devices path", filename)
                if os.path.exists(filename):
                    obj = self.load_dataset(filename)
                    test_tmp = obj['test']
                    self.ds_test = tf.data.Dataset.from_tensor_slices(test_tmp)
                    print("Test data length: ", len(list(self.ds_test.as_numpy_iterator())))
            elif test_devices in option['devices']:  # Test on specific device, e.g., test_devices == 'waist'
                filename = os.path.join(self.path, f"{test_devices}-{user}-{resample}.pickle")
                if os.path.exists(filename):
                    obj = self.load_dataset(filename)
                    test_tmp = obj['test']
                    self.ds_test = tf.data.Dataset.from_tensor_slices(test_tmp)
        else:
            print("Dataset doesn't exist")

    def get_train_data(self):
        return self.ds_train

    def get_test_data(self):
        return self.ds_test

    def load_dataset(self, filename):
        f = open(filename, 'rb')
        obj = pickle.load(f)
        f.close()
        return obj


if __name__ == "__main__":
    # os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    # os.environ["CUDA_VISIBLE_DEVICES"] = '0'

    parser = argparse.ArgumentParser(
        description='Inputs to data loading script')
    parser.add_argument('--dataset_path', default='/mnt/data/gsl',
                        help='path of the dataset .dat file')
    parser.add_argument('--dataset_name', default='realworld-3.0-0.0.dat', choices=[
                        'realworld-3.0-0.0.dat', 'opportunity-2.0-0.0.dat'], help='name of dataset file')
    parser.add_argument('--load_path', default=None,
                        help='path of the dataset TFrecords files')

    args = parser.parse_args()
    print(args.dataset_path, args.dataset_name)
    data = Data(args.dataset_path, args.dataset_name)
    import pdb
    pdb.set_trace()

    # walking = get_test_dataset(positions='head', activity='walking')
    # print(walking)
