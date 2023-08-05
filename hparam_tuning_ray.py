'''

1) To view the results in tensorboard, run the following inside docker with the appropriate values. Port number is the port you opened while running the docker container. 

tensorboard --logdir={working_directory}/logs/hparam_tuning_{exp_name}/ --port 6055 --bind_all

E.g., tensorboard --logdir=/mnt/data/gsl/runs/logs/hparam_tuning_waist_baseline/ --port 6055 --bind_all


2) Once tensorboard is running, you can ssh into the machine with port forwarding. Change the port as per your docker scripts

ssh -L 6055:localhost:6055 <server_machine_name>

Then go to your browser and open localhost:6055

'''

# thigh/supervised/models/

# thigh/supervised/logs/

# thigh/supervised/results
import argparse, os, datetime, signal
import subprocess
from itertools import permutations, product
import random

from tensorboard.plugins.hparams import api as hp
import tensorflow as tf
import numpy as np
import pickle
from collections import Counter
from imblearn.over_sampling import SMOTE
from load_data import DictMap
from data_loader.HHARDataset import HHARDataset
from data_loader.RealWorldDataset import RealWorldDataset
from data_partitioning import create_new_devices, process_nonpartitioned
import options


parser = argparse.ArgumentParser(description="Flower")
parser.add_argument("--exp_name", type=str, required=True)
parser.add_argument("--working_directory", type=str, default='/workspace/mdfl/')
parser.add_argument("--dataset", type=str, default='realworld')
parser.add_argument("--gpu_device", type=str, default='0,1,2,3,4,5,6,7')
args = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_device

if args.dataset == "realworld":
    option = options.RealWorldOpt
elif args.dataset == "hhar":
    option = options.HHAROpt
elif args.dataset == "opportunity":
    option = options.OpportunityOpt
elif args.dataset == "pamap2":
    option = options.PAMAP2Opt
elif args.dataset == "femnist":
    option = options.RealWorldOpt

print("args.dataset : ", args.dataset)
devices = option['devices']
users = option['users']

def flatten_sessions(data):
    X = None
    y = None

    # Merge session data
    for session in data:
        XX = data[session][0]
        yy = data[session][1]

        if len(XX) == 0:
            continue

        X = XX if X is None else np.vstack((X, XX))
        y = yy if y is None else np.hstack((y, yy))

    # norm = 30.0  # max value
    norm = np.max(X)
    X = X / norm
    X = X.astype(np.float32)
    return X, y

SEED = 913
def get_position_user_dataset(train_data, test_data, resample=False):
    train = flatten_sessions(data=train_data)
    test = flatten_sessions(data=test_data)

    if resample:
        X, y = train
        distribution = Counter(y)
        max_class = max(distribution.values())
        for d in distribution:
            distribution[d] = max_class

        sample_dict = dict(distribution)
        sm = SMOTE(random_state=SEED, sampling_strategy=sample_dict)
        X = np.reshape(X, (X.shape[0], -1))
        X_res, y_res = sm.fit_resample(X, y)
        X_res = np.reshape(X_res, (X_res.shape[0], option['seq_len'], 6))  # For RealWorld, option['seq_len'] = 150
        train = X_res, y_res

    return train, test

# Preprocessing
print("Dataset: ", args.dataset)
if args.dataset == "realworld":
    dataset_path = "./realworld_data"
    if not os.path.exists(dataset_path):
        os.mkdir(dataset_path)
        f = open(os.path.join('/mnt/data/gsl/', option['dataset_name']), 'rb')
        info, train_data, test_data = pickle.load(f)
        f.close()
        resample = True

        for user in users:
            X_test_all = None
            y_test_all = None
            for device in devices:

                file_path = os.path.join(dataset_path, f"{device}-{user}-{resample}.pickle")
                if not os.path.exists(file_path):
                    train_tmp, test_tmp = get_position_user_dataset(train_data[device][user], test_data[device][user],
                                                                    resample=True)
                    obj = {"train": train_tmp, "test": test_tmp}
                    with open(file_path, 'wb') as f:
                        pickle.dump(obj, f)

                    X_test, y_test = test_tmp
                    X_test_all = X_test if X_test_all is None else np.vstack((X_test_all, X_test))
                    y_test_all = y_test if y_test_all is None else np.hstack((y_test_all, y_test))

            file_path = os.path.join(dataset_path, f"{user}-test.pickle")
            with open(file_path, 'wb') as f:
                test_tmp = (X_test_all, y_test_all)
                pickle.dump({"test": test_tmp}, f)
elif args.dataset == "hhar":
    dataset_path = "./hhar_data"
    if not os.path.exists(dataset_path):
        os.mkdir(dataset_path)

elif args.dataset == "opportunity":
    dataset_path = "./opportunity_data"
    option = options.OpportunityOpt
    if not os.path.exists(dataset_path):
        os.mkdir(dataset_path)
        f = open(os.path.join('/mnt/data/gsl/', option['dataset_name']), 'rb')
        info, train_data, test_data = pickle.load(f)
        f.close()
        resample = True

        for user in users:
            X_test_all = None
            y_test_all = None
            for device in devices:

                file_path = os.path.join(dataset_path, f"{device}-{user}-{resample}.pickle")
                if not os.path.exists(file_path):
                    train_tmp, test_tmp = get_position_user_dataset(train_data[device][user], test_data[device][user],
                                                                    resample=True)
                    obj = {"train": train_tmp, "test": test_tmp}
                    with open(file_path, 'wb') as f:
                        pickle.dump(obj, f)

                    X_test, y_test = test_tmp
                    X_test_all = X_test if X_test_all is None else np.vstack((X_test_all, X_test))
                    y_test_all = y_test if y_test_all is None else np.hstack((y_test_all, y_test))

            file_path = os.path.join(dataset_path, f"{user}-test.pickle")
            with open(file_path, 'wb') as f:
                test_tmp = (X_test_all, y_test_all)
                pickle.dump({"test": test_tmp}, f)
    print("Opportunity processing done")

elif args.dataset == "pamap2":
    dataset_path = "./pamap2_data"
    option = options.PAMAP2Opt
    if not os.path.exists(dataset_path):
        os.mkdir(dataset_path)
        f = open(os.path.join('/mnt/data/gsl/', option['dataset_name']), 'rb')
        info, train_data, test_data = pickle.load(f)
        f.close()
        resample = True

        for user in users:
            X_test_all = None
            y_test_all = None
            for device in devices:
                file_path = os.path.join(dataset_path, f"{device}-{user}-{resample}.pickle")
                if not os.path.exists(file_path):
                    train_tmp, test_tmp = get_position_user_dataset(train_data[device][user], test_data[device][user],
                                                                    resample=True)
                    obj = {"train": train_tmp, "test": test_tmp}
                    with open(file_path, 'wb') as f:
                        pickle.dump(obj, f)

                    X_test, y_test = test_tmp
                    X_test_all = X_test if X_test_all is None else np.vstack((X_test_all, X_test))
                    y_test_all = y_test if y_test_all is None else np.hstack((y_test_all, y_test))

                file_path = os.path.join(dataset_path, f"{user}-test.pickle")
                with open(file_path, 'wb') as f:
                    test_tmp = (X_test_all, y_test_all)
                    pickle.dump({"test": test_tmp}, f)
    print("PAMAP2 processing done")
elif args.dataset == "femnist":
    dataset_path = "./realworld_data"

'''
Specify the training hyperparams here
'''

HP_LR_DECAY = hp.HParam('learning_rate_decay', hp.Discrete(['cosine']))

HP_OPTIM = hp.HParam('optimizer', hp.Discrete(['adam']))

HP_LR = hp.HParam('learning_rate', hp.Discrete([1e-3])) #, 1e-2]))
HP_LOCAL_EPOCHS = hp.HParam('local_epochs', hp.Discrete([10 if args.dataset == "opportunity" else 20]))  # 10])) 10 for opportunity, 20 for realworld + pamap2
HP_ROUND = hp.HParam('round', hp.Discrete([100]))
HP_BATCH_SIZE = hp.HParam('batch_size', hp.Discrete([32]))
HP_MODEL = hp.HParam('model', hp.Discrete(['DeepConvLSTM']))
# HP_MODEL = hp.HParam('model', hp.Discrete(['cnn']))
HP_TEST_DEVICES = hp.HParam('test_devices', hp.Discrete(['trained']))
# HP_TEST_DEVICES = hp.HParam('test_devices', hp.Discrete(['S2', 'S1', 'S11']))  # 'shin', 'waist', 'head','S11'])) #,
# HP_TEST_DEVICES = hp.HParam('test_devices', hp.Discrete(['shin', 'thigh', 'upperarm', 'waist']))
# HP_TAKE = hp.HParam('take', hp.Discrete([0.1, 0.25, 0.5, 0.7, 1.0]))
HP_TAKE = hp.HParam('take', hp.Discrete([1.0]))
HP_PERSONALIZATION = hp.HParam('personalization', hp.Discrete(['personal', 'no_personal', 'local_finetuning']))
# HP_DEVICE_SELECTION = hp.HParam('device_selection', hp.Discrete(['select_devices'])) #, 'select_all']))
HP_DEVICE_SELECTION = hp.HParam('device_selection', hp.Discrete(['best_utility_ours', 'random', 'stat']))
# HP_DEVICE_SELECTION = hp.HParam('device_selection', hp.Discrete(['best_utility_ours', 'oort', 'random', 'stat', 'device', 'time']))
HP_TIME_ALIGNED = hp.HParam('time_aligned', hp.Discrete(['aligned', 'not_aligned']))
# HP_TIME_ALIGNED = hp.HParam('time_aligned', hp.Discrete(['aligned']))
HP_DATA_PARTITIONING = hp.HParam('data_partitioning', hp.Discrete(['partitioning_by_class', 'no_partitioning']))
HP_CANDIDATE_LAM = hp.HParam('candidate_lam', hp.Discrete([1.0]))
HP_DATASET = hp.HParam('dataset', hp.Discrete([args.dataset]))  # Log dataset as hparm
HP_SAMPLE_DEVICES_PER = hp.HParam('num_sample_devices', hp.Discrete([0.5]))
# HP_SAMPLE_DEVICES_PER = hp.HParam('num_sample_devices', hp.Discrete([0.25, 0.1]))

METRIC_LOSS = 'loss'
METRIC_ACCURACY = 'accuracy'

working_directory = os.path.join(args.working_directory, args.exp_name)
if not os.path.exists(working_directory):
    os.makedirs(working_directory)
    os.makedirs(os.path.join(working_directory, 'models/'))
    os.makedirs(os.path.join(working_directory, 'logs/'))
    os.makedirs(os.path.join(working_directory, 'results/'))
    os.makedirs(os.path.join(working_directory, 'user_model_parameters/'))

start_time = str(int(datetime.datetime.now().timestamp()))

with tf.summary.create_file_writer(working_directory + '/logs/hparam_tuning_' + start_time).as_default():
    hp.hparams_config(
        hparams=[HP_LR, HP_OPTIM, HP_BATCH_SIZE, HP_LOCAL_EPOCHS, HP_ROUND, HP_MODEL, HP_TEST_DEVICES, HP_TAKE,
                 HP_PERSONALIZATION, HP_DEVICE_SELECTION, HP_TIME_ALIGNED, HP_DATA_PARTITIONING, HP_CANDIDATE_LAM,
                 HP_DATASET, HP_SAMPLE_DEVICES_PER],
        metrics=[hp.Metric(METRIC_LOSS, display_name='loss'), hp.Metric(METRIC_ACCURACY, display_name='accuracy')],
    )


session_num = 0


conditions = [('aligned', 'personal', 'best_utility_ours', 'partitioning_by_class')]
# conditions = [('not_aligned', 'no_personal', 'random', 'partitioning_by_class')] #, 'no_partitioning')]
    # [('not_aligned', 'local_finetuning', 'random')]
              # ('not_aligned', 'no_personal', 'random')]
              # ('aligned', 'personal', 'random')]

# conditions = [('aligned', 'personal', 'device')]
              # ('not_aligned', 'personal', 'random')]
              # ('not_aligned', 'no_personal', 'random')

# if args.dataset == "opportunity":
#     conditions = [('not_aligned', 'no_personal', 'random', 0.5),
#                   ('not_aligned', 'personal', 'random', 0.5)]
# elif args.dataset == "pamap2":
#     conditions = [('aligned', 'personal', 'best_utility_ours', 0.5),
#                   ('not_aligned', 'no_personal', 'random', 0.5),
#                   ('not_aligned', 'personal', 'random', 0.5)]

# FedAvg
# conditions = [('not_aligned', 'no_personal', 'random', 0.5),
#               ('not_aligned', 'personal', 'random', 0.5)]

#Ditto
# conditions = [('not_aligned', 'personal', 'random', 0.5)]

# conditions = [('aligned', 'personal', 'best_utility_ours', 0.25),
#               ('aligned', 'personal', 'random', 0.25),
#               ('aligned', 'personal', 'stat', 0.25),
#               ('aligned', 'personal', 'device', 0.25),
#               ('aligned', 'personal', 'time', 0.25),
#               ('aligned', 'personal', 'stat', 0.5),
#               ('aligned', 'personal', 'time', 0.5)]

# ('not_aligned', 'no_personal', 'random')
# ('not_aligned', 'personal', 'best_utility_ours'),
              # ('not_aligned', 'personal', 'random')]

with open(os.path.join(working_directory, 'client_log.csv'), 'w') as file_client_log:
    file_client_log.write('run,' + ",".join(users) + ',\n')
    for lr in HP_LR.domain.values:  # 1
        for optim in HP_OPTIM.domain.values:  # 1
            for batch_size in HP_BATCH_SIZE.domain.values:  # 1
                for local_epochs in HP_LOCAL_EPOCHS.domain.values:  # 1
                    for rnd in HP_ROUND.domain.values:  # 1
                        for test_devices in HP_TEST_DEVICES.domain.values:  # 1
                            for model in HP_MODEL.domain.values:  # 2
                                # for take in HP_TAKE.domain.values:  # 1
                                for sample_device_per in HP_SAMPLE_DEVICES_PER.domain.values:  # 5
                                    for personal in HP_PERSONALIZATION.domain.values:
                                        for device_selection in HP_DEVICE_SELECTION.domain.values:
                                            for time_aligned in HP_TIME_ALIGNED.domain.values:
                                                take = 1.0
                                                for data_partitioning in HP_DATA_PARTITIONING.domain.values:
                                                    # if not (time_aligned, personal, device_selection) in conditions:
                                                    if not (time_aligned, personal, device_selection,
                                                            data_partitioning) in conditions:
                                                        continue
                                                    for candidate_lam in HP_CANDIDATE_LAM.domain.values:
                                                        run_name = "run-%d" % int(datetime.datetime.now().timestamp())

                                                        hparams = {
                                                            HP_LR: lr,
                                                            HP_OPTIM: optim,
                                                            HP_BATCH_SIZE: batch_size,
                                                            HP_LOCAL_EPOCHS: local_epochs,
                                                            HP_ROUND: rnd,
                                                            HP_TEST_DEVICES: test_devices,
                                                            HP_MODEL: model,
                                                            HP_TAKE: take,
                                                            HP_SAMPLE_DEVICES_PER: sample_device_per,
                                                            HP_PERSONALIZATION: personal,
                                                            HP_DEVICE_SELECTION: device_selection,
                                                            HP_TIME_ALIGNED: time_aligned,
                                                            HP_DATA_PARTITIONING: data_partitioning,
                                                            HP_CANDIDATE_LAM: candidate_lam,
                                                            HP_DATASET: args.dataset,
                                                        }

                                                        cid = 0

                                                        if time_aligned == 'aligned':
                                                            partitions_path = os.path.join(dataset_path,
                                                                                           "partitioned_devices")
                                                        else:
                                                            partitions_path = os.path.join(dataset_path,
                                                                                           "partitioned_devices_unaligned")
                                                        if data_partitioning == "no_partitioning":
                                                            partitions_path = os.path.join(dataset_path, "no_partitioning")

                                                        if test_devices != 'trained' and test_devices != 'all':
                                                            partitions_path += "_LOO_" + test_devices
                                                        if test_devices[0] == 'S':
                                                            partitions_path = "/mnt/data/mdfl/realworld/" \
                                                                              "partitioned_devices_LOO_" + test_devices

                                                        if not os.path.exists(partitions_path):
                                                            os.mkdir(partitions_path)
                                                            if data_partitioning == "no_partitioning":
                                                                num_clients = len(option["devices"]) * len(option["users"])
                                                                process_nonpartitioned(args.dataset, partitions_path, batch_size)
                                                            else:
                                                                num_users = create_new_devices(args.dataset, partitions_path,
                                                                                   batch_size, test_devices, time_aligned)
                                                                num_clients = len(os.listdir(partitions_path))
                                                                break
                                                        else:
                                                            num_clients = len(os.listdir(partitions_path))

                                                        time = str(datetime.datetime.now())
                                                        print('--- Count: %d.....Starting trial %s at time %s' % (
                                                        session_num, run_name, time))
                                                        hyper = "_".join([str(hparams[h]) for h in hparams])
                                                        print(hyper)

                                                        tf_summary_path = working_directory + '/logs/hparam_tuning_' + start_time + "/" + run_name
                                                        round_results_path = working_directory + '/results/hparam_tuning_' + start_time
                                                        if not os.path.exists(round_results_path):
                                                            os.mkdir(round_results_path)
                                                        round_results_path += "/" + run_name
                                                        if not os.path.exists(round_results_path):
                                                            os.mkdir(round_results_path)

                                                        # Toggle personalization
                                                        if personal == "no_personal":
                                                            user_model_path = "no_personal"
                                                        elif personal == "local_finetuning":
                                                            user_model_path = "local_finetuning"
                                                        else:
                                                            user_model_path = working_directory + '/user_model_parameters/hparam_tuning_' + start_time
                                                            if not os.path.exists(user_model_path):
                                                                os.mkdir(user_model_path)
                                                            user_model_path += "/" + run_name

                                                        writer = tf.summary.create_file_writer(
                                                            working_directory + '/logs/hparam_tuning_' + start_time +
                                                            "/" + run_name)
                                                        with writer.as_default():
                                                            hp.hparams(hparams)  # record the values used in this trial
                                                        writer.flush()

                                                        processes = []
                                                        #
                                                        # num_clients = int(len(users) * len(devices) / take)  # user * device * 1/take
                                                        # num_clients = 3350
                                                        p = subprocess.Popen(["python", "server_ray.py", "--dataset", args.dataset,
                                                                              "--tf_summary_path", tf_summary_path,
                                                                              "--learning_rate", str(lr),
                                                                              "--optimizer", optim,
                                                                              "--batch_size", str(batch_size),
                                                                              "--round", str(rnd),
                                                                              "--local_epochs", str(local_epochs),
                                                                              "--model", model,
                                                                              "--num_clients", str(num_clients),
                                                                              "--take", str(take),
                                                                              "--user_model_path", user_model_path,
                                                                              "--dataset_path", partitions_path,
                                                                              "--device_selection", device_selection,
                                                                              "--time_aligned", time_aligned,
                                                                              "--candidate_lam", str(candidate_lam),
                                                                              "--round_results_path", round_results_path,
                                                                              "--test_devices", test_devices,
                                                                              "--sample_device_per", str(sample_device_per)
                                                                              ])
                                                        processes.append(p)

                                                        for p in processes:
                                                            if p is not None:
                                                                p.communicate()

                                                        session_num += 1
