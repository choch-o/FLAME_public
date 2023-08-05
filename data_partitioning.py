import os
from itertools import permutations, product
import random
import pprint

import tensorflow as tf
import numpy as np
import pickle

from data_loader.RealWorldDataset import RealWorldDataset
from data_loader.HHARDataset import HHARDataset
from data_loader.OpportunityDataset import OpportunityDataset
from data_loader.PAMAP2Dataset import PAMAP2Dataset
import options


def create_new_devices(dataset, directory_path, batch_size, test_devices, time_aligned):
    """Load 1/10th of the training and test data to simulate a partition"""
    """
        Input: dataset name, path to save partitions, batch size, test set ('train' or 'all')
        Return: total number of clients (int)
        Output: client_id.pickle files that contain train datset (train + val)
    """
    print("create new device, dataset: ", dataset)
    if not os.path.exists(directory_path):
        os.mkdir(directory_path)

    if dataset == "hhar":
        option = options.HHAROpt
        """
        'users': ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i'],
        'models': ['nexus4', 's3', 's3mini', 'lgwatch'],
        'devices': ['lgwatch_1', 'lgwatch_2', 'gear_1', 'gear_2', 'nexus4_1', 'nexus4_2',
                    's3_1', 's3_2', 's3mini_1', 's3mini_2'],

        'classes': ['bike', 'sit', 'stand', 'walk', 'stairsup', 'stairsdown'],
        """
        # models = option['models']
        devices = option['models']
        users = option['users']
        classes = option['classes']

        m = len(classes)
        class_per_user = 2
        take = 0.1

    elif dataset == "realworld":
        option = options.RealWorldOpt
        """
        'devices': ['forearm', 'thigh', 'head', 'chest', 'upperarm', 'waist', 'shin'],
        'users': ['S1', 'S2', 'S3', 'S4', 'S5', 'S6', 'S7', 'S8', 'S9', 'S10', 'S11', 'S12', 'S13','S14', 'S15'],
        'classes': ['climbingdown', 'climbingup', 'jumping', 'lying', 'running', 'sitting', 'standing', 'walking'],
        """

        users = option['users'].copy()
        devices = option['devices'].copy()
        classes = option['classes']

        m = len(classes)
        class_per_user = 2  # number of classes to take from each user
        take = 0.1

    elif dataset == "opportunity":
        """
        'devices': ['back', 'lla', 'rshoe', 'rua', 'lshoe'],
        'users': ['S1', 'S2', 'S3', 'S4'],
        'classes': [0, 1, 2, 3],
        """
        print("opportunity dataset")
        option = options.OpportunityOpt
        users = option['users']
        devices = option['devices']
        classes = option['classes']
        m = len(classes)
        class_per_user = 1  # number of classes to take from each user
        take = 0.1

    elif dataset == "pamap2":
        """
        'devices': ['hand', 'chest', 'ankle'],
        'users': ['S1', 'S2', 'S3', 'S4', 'S5', 'S6', 'S7', 'S8'],
        'classes': ['lying', 'sitting', 'standing', 'walking', 'running',
                    'cycling', 'nordic_walking', 'ascending_stairs', 'descending_stairs', 'vacuum_cleaning',
                    'ironing', 'rope_jumping'],
        """
        print("pamap2 dataset partitioning")
        option = options.PAMAP2Opt
        users = option['users'].copy()
        devices = option['devices'].copy()
        classes = option['classes'].copy()

        m = len(classes)
        class_per_user = 3
        take = 0.1

    r = int(m / class_per_user)

    # For LOUO experiment, exclude one user
    if test_devices in users:
        print("test devices in users")
        users.remove(test_devices)

    # (user1 [class 0-1], user2 [class 2-3], user3 [class 4-5], user4 [class 6-7])
    P_list = list(permutations(users, r))  # Example: ('S1', 'S5', 'S6', 'S10'')
    #
    # print("15P4 permutations")
    # print(P_list)

    # Randomly pick a permutation
    # Use seed, shuffle, and idx to support reproducing
    random.seed(42)
    random.shuffle(P_list)

    print("Shuffled P_list")
    print(len(P_list))

    # Initialize available partition flag dictionary

    # avail_p = {user: {class_idx: list(range(int(1/take))) for class_idx in range(len(classes))} for user in users}

    avail_p = {user: {class_idx: list(range(1, int(1/take))) for class_idx in range(len(classes))} for user in users}
    print("Initialized avail_p")
    pprint.pprint(avail_p)

    cid = 0

    if not os.path.exists('result_logs'):
        os.mkdir('result_logs')
    file_user_set = open('result_logs/user_set.txt', 'w')

    ds_map = {}
    for user in users:
        for device in devices:
            # For LODO experiment, exclude one device

            if device == test_devices:
                continue
            if dataset == "hhar":
                ds_map[(user, device)] = HHARDataset(
                    option['dataset_path'],
                    device, user, resample=True, test_devices=test_devices
                )
            elif dataset == "opportunity":
                ds_map[(user, device)] = OpportunityDataset(
                    option['dataset_path'],
                    device, user, resample=True, test_devices=test_devices
                )
            elif dataset == "pamap2":
                ds_map[(user, device)] = PAMAP2Dataset(
                    option['dataset_path'],
                    device, user, resample=True, test_devices=test_devices
                )
            else:
                ds_map[(user, device)] = RealWorldDataset(
                    option['dataset_path'],
                    device, user, resample=True, test_devices=test_devices
                )

    # Keep 0.1 of original devices

    for user in users:
        X_train, y_train, X_val, y_val, X_test, y_test = None, None, None, None, None, None
        for device in devices:
            # For LODO experiment, exclude one device
            if device in test_devices:
                continue
            file_path = os.path.join(directory_path, f"{cid}_{device}.pickle")
            if dataset == "hhar":
                # model = models[idx]
                # device = devices[idx]
                # user = users[idx]
                (X_train, y_train), (X_val, y_val), (X_test, y_test) = ds_map[(user, device)]\
                    .get_dataset(batch_size=batch_size, take=take, partition_idx=0, time_aligned=time_aligned)
                # (X_train, y_train), (X_val, y_val), (X_test, y_test) = split_data(X, y, batch_size)
            elif dataset == "realworld":
                (X_train, y_train), (X_val, y_val), (X_test, y_test) = ds_map[(user, device)]\
                    .get_dataset(batch_size=batch_size, take=take, partition_idx=0, time_aligned=time_aligned)
            elif dataset == "opportunity":
                (X_train, y_train), (X_val, y_val), (X_test, y_test) = ds_map[(user, device)]\
                    .get_dataset(batch_size=batch_size, take=take, partition_idx=0, time_aligned=time_aligned)
            elif dataset == "pamap2":
                (X_train, y_train), (X_val, y_val), (X_test, y_test) = ds_map[(user, device)]\
                    .get_dataset(batch_size=batch_size, take=take, partition_idx=0, time_aligned=time_aligned)

            if len(X_train) > 0:
                obj = {"X_train": X_train,
                       "y_train": y_train,
                       "X_val": X_val,
                       "y_val": y_val,
                       "X_test": X_test,
                       "y_test": y_test}

                with open(file_path, 'wb') as f:
                    pickle.dump(obj, f)
        cid += 1

    # Create new devices combining different classes from users in the user set
    while len(P_list) > 0:
        user_set = P_list.pop(0)
        file_user_set.write(f"{cid}, {user_set}\n")

        print(f"CID: {cid}, User Set: {user_set}\n")

        # X_train_all, y_train_all, X_val_all, y_val_all, X_test_all, y_test_all = None, None, None, None, None, None

        for device in devices:
            # For LODO experiment, exclude one device
            if device in test_devices:
                continue
            # X_train_all, y_train_all, X_test_all, y_test_all = None, None, None, None, None, None
            ds_train_all, ds_test_all = None, None
            file_path = os.path.join(directory_path, f"{cid}_{device}.pickle")
            if os.path.exists(file_path):
                continue
            for i, user in enumerate(user_set):
                ds = ds_map[(user, device)]
                for class_idx in [i * class_per_user + c for c in range(class_per_user)]:
                    partition_idx = avail_p[user][class_idx][0]

                    ds_train, ds_test = ds.get_dataset_by_class(batch_size=batch_size, take=take,
                                                                partition_idx=partition_idx,
                                                                class_label=class_idx,
                                                                time_aligned=time_aligned)

                    if len(list(ds_train.as_numpy_iterator())) > 0:
                        ds_train_all = ds_train if ds_train_all is None else ds_train_all.concatenate(ds_train)
                        ds_test_all = ds_test if ds_test_all is None else ds_test_all.concatenate(ds_test)
                        print("DS TRAIN LEN: ", len(list(ds_train_all.as_numpy_iterator())))
                        print("DS TEST LEN: ", len(list(ds_test_all.as_numpy_iterator())))


            if ds_train_all is not None:
                device_idx = devices.index(device)
                seed = 42 if time_aligned == 'aligned' else 42 + device_idx
                ds_train_all = ds_train_all.shuffle(buffer_size=500000, seed=seed)
                if test_devices != 'trained' and test_devices != 'all':
                    ds_test_all = ds_map[(user_set[0], device)].ds_test
                ds_test_all = ds_test_all.shuffle(buffer_size=500000, seed=seed)

                X_train, y_train = ds.to_np_data(ds_train_all, batch_size)
                X_val, y_val = np.array([]), np.array([])
                X_test, y_test = ds.to_np_data(ds_test_all, batch_size)

                if len(X_train) > 0:
                    obj = {"X_train": X_train,
                           "y_train": y_train,
                           "X_val": X_val,
                           "y_val": y_val,
                           "X_test": X_test,
                           "y_test": y_test}
                    with open(file_path, 'wb') as f:
                        pickle.dump(obj, f)
            else:
                print("NONE X TRAIN")

        for i, user in enumerate(user_set):
            for class_idx in [i * class_per_user + c for c in range(class_per_user)]:
                partition_idx = avail_p[user][class_idx].pop(0)
                if len(avail_p[user][class_idx]) < 1:
                    # remove all permutations with user with corresponding class_label
                    print(f"Exhausted user class: {user}-{class_idx}")
                    pprint.pprint(avail_p)

                    if class_idx % class_per_user == 0:
                        exhausted_user_idx = class_idx // class_per_user
                        print(f"P list size before deletion: {len(P_list)}")
                        P_list = [p for p in P_list if p[exhausted_user_idx] != user]
                        print(f"P list size after deletion: {len(P_list)}")
        cid += 1
    file_user_set.close()
    return cid
#
#
#
# directory_path = os.path.join('realworld_data', "data_partitions")
# create_new_devices('realworld', directory_path, 16, 'trained')

def create_new_devices_centralized(dataset, directory_path, batch_size, test_devices, time_aligned):
    """Load 1/10th of the training and test data to simulate a partition"""
    """
        Input: dataset name, path to save partitions, batch size, test set ('train' or 'all')
        Return: None
        Output: client_id.pickle files that contain train datset (train + val)
    """

    if not os.path.exists(directory_path):
        os.mkdir(directory_path)

    if dataset == "realworld":
        option = options.RealWorldOpt
        """
        'devices': ['forearm', 'thigh', 'head', 'chest', 'upperarm', 'waist', 'shin'],
        'users': ['S1', 'S2', 'S3', 'S4', 'S5', 'S6', 'S7', 'S8', 'S9', 'S10', 'S11', 'S12', 'S13','S14', 'S15'],
        'classes': ['climbingdown', 'climbingup', 'jumping', 'lying', 'running', 'sitting', 'standing', 'walking'],
        """

        users = option['users'].copy()
        devices = option['devices'].copy()
        classes = option['classes']

        m = len(classes)
        class_per_user = 2  # number of classes to take from each user
        take = 0.1

    r = int(m / class_per_user)

    # For LOUO experiment, exclude one user
    if test_devices in users:
        print("test devices in users")
        users.remove(test_devices)

    # (user1 [class 0-1], user2 [class 2-3], user3 [class 4-5], user4 [class 6-7])
    P_list = list(permutations(users, r))  # Example: ('S1', 'S5', 'S6', 'S10'')
    #
    # print("15P4 permutations")
    # print(P_list)

    # Randomly pick a permutation
    # Use seed, shuffle, and idx to support reproducing
    random.seed(42)
    random.shuffle(P_list)

    print("Shuffled P_list")
    print(len(P_list))

    # Initialize available partition flag dictionary

    # avail_p = {user: {class_idx: list(range(int(1/take))) for class_idx in range(len(classes))} for user in users}

    avail_p = {user: {class_idx: list(range(1, int(1/take))) for class_idx in range(len(classes))} for user in users}
    print("Initialized avail_p")
    pprint.pprint(avail_p)

    cid = 0

    if not os.path.exists('result_logs'):
        os.mkdir('result_logs')
    file_user_set = open('result_logs/user_set.txt', 'w')

    ds_map = {}
    for user in users:
        for device in devices:
            # For LODO experiment, exclude one device
            if device == test_devices:
                continue

            ds_map[(user, device)] = RealWorldDataset(
                option['dataset_path'],
                device, user, resample=True, test_devices=test_devices
            )

    # Keep 0.1 of original devices

    for user in users:
        X_train, y_train, X_val, y_val, X_test, y_test = None, None, None, None, None, None
        for device in devices:
            # For LODO experiment, exclude one device
            if device in test_devices:
                continue
            file_path = os.path.join(directory_path, f"{cid}_{device}.pickle")
            if dataset == "realworld":
                (X_train, y_train), (X_val, y_val), (X_test, y_test) = ds_map[(user, device)]\
                    .get_dataset_centralized(batch_size=batch_size, take=take, partition_idx=0,
                                             time_aligned=time_aligned)

            if len(X_train) > 0:
                obj = {"X_train": X_train,
                       "y_train": y_train,
                       "X_val": X_val,
                       "y_val": y_val,
                       "X_test": X_test,
                       "y_test": y_test}

                with open(file_path, 'wb') as f:
                    pickle.dump(obj, f)
        cid += 1

    # Create new devices combining different classes from users in the user set
    while len(P_list) > 0:
        user_set = P_list.pop(0)
        file_user_set.write(f"{cid}, {user_set}\n")

        print(f"CID: {cid}, User Set: {user_set}\n")

        # X_train_all, y_train_all, X_val_all, y_val_all, X_test_all, y_test_all = None, None, None, None, None, None

        for device in devices:
            # For LODO experiment, exclude one device
            if device in test_devices:
                continue
            # X_train_all, y_train_all, X_test_all, y_test_all = None, None, None, None, None, None
            ds_train_all, ds_test_all = None, None
            file_path = os.path.join(directory_path, f"{cid}_{device}.pickle")
            if os.path.exists(file_path):
                continue
            for i, user in enumerate(user_set):
                ds = ds_map[(user, device)]
                for class_idx in [i * class_per_user + c for c in range(class_per_user)]:
                    partition_idx = avail_p[user][class_idx][0]

                    ds_train, ds_test = ds.get_dataset_by_class(batch_size=batch_size, take=take,
                                                                partition_idx=partition_idx,
                                                                class_label=class_idx,
                                                                time_aligned=time_aligned)

                    if len(list(ds_train.as_numpy_iterator())) > 0:
                        ds_train_all = ds_train if ds_train_all is None else ds_train_all.concatenate(ds_train)
                        ds_test_all = ds_test if ds_test_all is None else ds_test_all.concatenate(ds_test)
                        print("DS TRAIN LEN: ", len(list(ds_train_all.as_numpy_iterator())))
                        print("DS TEST LEN: ", len(list(ds_test_all.as_numpy_iterator())))

            if ds_train_all is not None:
                device_idx = devices.index(device)
                seed = 42 if time_aligned == 'aligned' else 42 + device_idx
                ds_train_all = ds_train_all.shuffle(buffer_size=500000, seed=seed)
                train_size = len(ds_train_all)
                train_split = ds_train_all.skip(int(train_size * 0.2))
                val_split = ds_train_all.take(int(train_size * 0.2))
                if test_devices != 'trained' and test_devices != 'all':
                    ds_test_all = ds_map[(user_set[0], device)].ds_test
                ds_test_all = ds_test_all.shuffle(buffer_size=500000, seed=seed)

                X_train, y_train = ds.to_np_data(train_split, batch_size)
                X_val, y_val = ds.to_np_data(val_split, batch_size)
                X_test, y_test = ds.to_np_data(ds_test_all, batch_size)

                if len(X_train) > 0:
                    obj = {"X_train": X_train,
                           "y_train": y_train,
                           "X_val": X_val,
                           "y_val": y_val,
                           "X_test": X_test,
                           "y_test": y_test}
                    with open(file_path, 'wb') as f:
                        pickle.dump(obj, f)
            else:
                print("NONE X TRAIN")

        for i, user in enumerate(user_set):
            for class_idx in [i * class_per_user + c for c in range(class_per_user)]:
                partition_idx = avail_p[user][class_idx].pop(0)
                if len(avail_p[user][class_idx]) < 1:
                    # remove all permutations with user with corresponding class_label
                    print(f"Exhausted user class: {user}-{class_idx}")
                    pprint.pprint(avail_p)

                    if class_idx % class_per_user == 0:
                        exhausted_user_idx = class_idx // class_per_user
                        print(f"P list size before deletion: {len(P_list)}")
                        P_list = [p for p in P_list if p[exhausted_user_idx] != user]
                        print(f"P list size after deletion: {len(P_list)}")
        cid += 1
    file_user_set.close()
    return cid

def process_nonpartitioned(dataset, directory_path, batch_size):
    if dataset == "realworld":
        option = options.RealWorldOpt
        """
        'devices': ['forearm', 'thigh', 'head', 'chest', 'upperarm', 'waist', 'shin'],
        'users': ['S1', 'S2', 'S3', 'S4', 'S5', 'S6', 'S7', 'S8', 'S9', 'S10', 'S11', 'S12', 'S13','S14', 'S15'],
        'classes': ['climbingdown', 'climbingup', 'jumping', 'lying', 'running', 'sitting', 'standing', 'walking'],
        """
    elif dataset == "opportunity":
        option = options.OpportunityOpt
    elif dataset == "pamap2":
        option = options.PAMAP2Opt

    users = option['users'].copy()
    devices = option['devices'].copy()

    ds_map = {}
    for user in users:
        for device in devices:
            if dataset == "opportunity":
                ds_map[(user, device)] = OpportunityDataset(
                    option['dataset_path'],
                    device, user, resample=True, test_devices='trained'
                )
            elif dataset == "pamap2":
                ds_map[(user, device)] = PAMAP2Dataset(
                    option['dataset_path'],
                    device, user, resample=True, test_devices='trained'
                )
            else:
                ds_map[(user, device)] = RealWorldDataset(
                    option['dataset_path'],
                    device, user, resample=True, test_devices='trained'
                )

    cid = 0
    take = 1.0
    for user in users:
        X_train, y_train, X_val, y_val, X_test, y_test = None, None, None, None, None, None
        for device in devices:
            file_path = os.path.join(directory_path, f"{cid}_{device}.pickle")
            if dataset == "realworld":
                (X_train, y_train), (X_val, y_val), (X_test, y_test) = ds_map[(user, device)]\
                    .get_dataset(batch_size=batch_size, take=take, partition_idx=0, time_aligned='aligned')
            elif dataset == "opportunity":
                (X_train, y_train), (X_val, y_val), (X_test, y_test) = ds_map[(user, device)]\
                    .get_dataset(batch_size=batch_size, take=take, partition_idx=0, time_aligned='aligned')
            elif dataset == "pamap2":
                (X_train, y_train), (X_val, y_val), (X_test, y_test) = ds_map[(user, device)]\
                    .get_dataset(batch_size=batch_size, take=take, partition_idx=0, time_aligned='aligned')

            if len(X_train) > 0:
                obj = {"X_train": X_train,
                       "y_train": y_train,
                       "X_val": X_val,
                       "y_val": y_val,
                       "X_test": X_test,
                       "y_test": y_test}

                with open(file_path, 'wb') as f:
                    pickle.dump(obj, f)
        cid += 1