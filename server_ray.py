from typing import Any, Callable, Dict, List, Optional, Tuple
import os, argparse, datetime, copy
from logging import DEBUG, INFO
import json
import shutil
import glob

import flwr as fl
import tensorflow as tf
import numpy as np
import multiprocessing as mp
from flwr.common import (
    EvaluateRes,
    FitIns,
    FitRes,
    EvaluateRes,
    Parameters,
    Scalar,
    Weights,
    parameters_to_weights,
    weights_to_parameters,
)
from flwr.common.logger import log

from models import DeepConvLSTM, SimCLR, FEMNIST_CNN
import options
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import pickle
import ray
from flwr.server.ray_server.ray_client_proxy import RayClientProxy
from flwr.server.client_manager import ClientManager
from device_selection import Device, select_devices_UAC, compute_metrics
from system_profiler.init_devices import get_device_parameters

# from leaf.models.utils.model_utils import read_data

SAVE_DIR = './results'
FILE_PATH = SAVE_DIR

LSTM_UNITS = 32
CNN_FILTERS = 3
NUM_LSTM_LAYERS = 1

PATIENCE = 20
SEED = 0
F = 32
D = 10

CURR_ROUND = 0

# Global variables that will be replaced by args
DEVICE = "forearm"
LEARNING_RATE = 1e-4
LOCAL_EPOCHS = 10
BATCH_SIZE = 16
ROUND = 5
num_clients = 105
sample_device_per = 0.5
# SEED = 913
SEED = 42

user_model_path = "./checkpoints"
round_results_path = "./results"
writer = None
device_selection = "overall"

# user_parameters = {}
# user_results = {}

WAIT_TIMEOUT = 600

user_list = set()

"""User Centered Personalization Strategy"""
class PersonalizationStrategy(fl.server.strategy.FedAvg):
    def __init__(
            self,
            fraction_fit: float = 0.1,
            fraction_eval: float = 0.1,
            min_fit_clients: int = 1,
            min_eval_clients: int = 1,
            min_available_clients: int = 1,
            eval_fn: Optional[
                Callable[[Weights], Optional[Tuple[float, Dict[str, Scalar]]]]
            ] = None,
            on_fit_config_fn: Optional[Callable[[int], Dict[str, Scalar]]] = None,
            on_evaluate_config_fn: Optional[Callable[[int], Dict[str, Scalar]]] = None,
            initial_parameters: Optional[Parameters] = None,
            num_clients: int = 105,
    ) -> None:
        super().__init__(
            fraction_fit=fraction_fit,
            fraction_eval=fraction_eval,
            min_fit_clients=min_fit_clients,
            min_eval_clients=min_eval_clients,
            min_available_clients=min_available_clients,
            eval_fn=eval_fn,
            on_fit_config_fn=on_fit_config_fn,
            on_evaluate_config_fn=on_evaluate_config_fn,
            initial_parameters=initial_parameters
        )
        try:
            if device_selection != 'none':
                num_device_per_user = len(option['models']) if dataset == "hhar" else len(option['devices'])
                if args.test_devices in option['devices']:
                    num_device_per_user -= 1
                device_configs, device_speeds = get_device_parameters(num_clients, num_device_per_user)
                print(len(device_configs), len(device_speeds))
                self.devices = [Device(device_configs[i], device_speeds[i // num_device_per_user - 1], dataset=dataset)
                                for i in range(num_clients)]
                self.losses = []
                self.round_times = []
                self.sampled_indices = []

                self.init_loss, self.init_drain = compute_metrics(self.devices)

                with open("{}/init.json".format(round_results_path), 'w') as json_file:
                    round_data = {
                        'rnd': 'init',
                        'total_loss': self.init_loss,
                        'dead_devices': self.init_drain,
                        'device_snapshot': [
                            {
                                'device_id': i,
                                'drain': d.drain,
                                't_local': d.t_local
                            } for i, d in enumerate(self.devices)
                        ]
                    }
                    json.dump(round_data, json_file)
        except Exception as e:
            print("Something wrong with Strategy INIT")
            print(e)

    def configure_fit(self,
                      rnd: int,
                      parameters: Parameters,
                      client_manager: ClientManager) -> List[Tuple[RayClientProxy, FitIns]]:
        if device_selection == 'none':
            return super().configure_fit(rnd, parameters, client_manager)
        else:
            try:
                """Configure the next round of training"""

                # Block until `min_num_clients` are available
                sample_size, min_num_clients = self.num_fit_clients(
                    client_manager.num_available()
                )
                success = client_manager.wait_for(
                    num_clients=min_num_clients, timeout=WAIT_TIMEOUT
                )
                if not success:
                    # Do not continue if not enough clients are available
                    log(INFO,
                        "Not enough clients available after timeout %s",
                        WAIT_TIMEOUT
                        )
                    return []

                # Sample clients
                msg = "Round %s, sample %s clients (based on device selection criteria)"
                log(DEBUG, msg, str(rnd), str(sample_size))
                all_clients: Dict[str, RayClientProxy] = client_manager.all()
                cid_idx: Dict[int, str] = {}
                for idx, (cid, _) in enumerate(all_clients.items()):
                    cid_idx[idx] = cid
                    # print("All clients cid: {}, idx: {}".format(cid, idx))

                global CURR_ROUND
                CURR_ROUND = rnd
                if rnd == 1:
                    # sampled_indices = select_devices(rnd, devices=self.devices, strategy='random')
                    sampled_indices = [*range(len(all_clients))]

                else:
                    num_device_per_user = len(option['models']) if dataset == "hhar" else len(option['devices'])
                    if args.test_devices in option['devices']:
                        num_device_per_user -= 1
                    num_sample_devices = int(len(self.devices) * sample_device_per)
                    print("NUM DEVICE PER USER SAMPLED : ", num_device_per_user)
                    sampled_indices = select_devices_UAC(rnd, num_devices=num_sample_devices, devices=self.devices, strategy=device_selection,
                                                         num_device_per_user=num_device_per_user)

                print("sampled indices: ", sampled_indices)
                self.sampled_indices = sampled_indices
                clients = [all_clients[cid_idx[idx]] for idx in sampled_indices]

                round_time = 0
                for idx in sampled_indices:
                    if self.devices[idx].t_local > round_time:
                        round_time = self.devices[idx].t_local

                    self.devices[idx].update_local_state(rnd)
                    self.round_times.append(round_time)

                # Prepare parameters and config
                config = {}
                if self.on_fit_config_fn is not None:
                    # Use custom fit config function if provided
                    config = self.on_fit_config_fn(rnd)

                # Fit instructions
                fit_ins = FitIns(parameters, config)
                # Return client/config pairs
            except Exception as e:
                print("Something wrong with CONFIGURE FIT")
                print(e)
            # return [(client, fit_ins) for client in clients]
            return super().configure_fit(rnd, parameters, client_manager)

    def aggregate_fit(
            self,
            rnd: int,
            results: List[Tuple[RayClientProxy, FitRes]],
            failures: List[BaseException],
    # ) -> Optional[fl.common.Weights]:
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        try:
            print("AGGREGATE FIT IS CALLED")
            # print(f"Results: {results}")
            if not results:
                return None, {}

            global_weights_results = []

            if device_selection != 'none':
                # Update number of samples and loss for selected clients
                # sampled_cid = []
                for client, r in results:
                    if int(client.cid) in self.sampled_indices:
                        self.devices[int(client.cid)].update_num_samples(r.num_examples)
                        self.devices[int(client.cid)].update_loss(r.metrics['loss'])
                        global_weights_results.append((parameters_to_weights(r.parameters), r.num_examples))

            if user_model_path != "no_personal" and user_model_path != "local_finetuning":
                # Dummy model to load weights
                input_shape = (option['seq_len'], option['input_dim'])
                tf.random.set_seed(SEED)
                if args.model == "DeepConvLSTM":
                    user_model = DeepConvLSTM(input_shape=input_shape, num_labels=option['num_class'],
                                              LSTM_units=option['lstm_units'],
                                              num_conv_filters=option['cnn_filters'],
                                              batch_size=args.batch_size, F=option['F'], D=option['D'])
                    user_model.compile(args.optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
                elif args.model == "SimCLR":
                    user_model = SimCLR(input_shape=input_shape, num_labels=option['num_class'])
                    user_model.compile(args.optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
                elif args.model == "cnn":
                    user_model = FEMNIST_CNN()
                    user_model.compile(args.optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])


                if dataset == "hhar":
                    devices = option["models"]
                    params_by_user = {u: [] for u in list(user_list)}
                    for cid in range(num_clients):
                        user, device = hhar_devices[str(cid)]
                        user_model.load_weights(os.path.join(user_model_path, f"{user}_{device}"))
                        with open(os.path.join(user_model_path, f'num_examples_{user}_{device}'), 'r') as f:
                            num_examples = int(f.readline())
                        device_weights = user_model.get_weights()
                        params_by_user[user].append((device_weights, num_examples))

                    for user in params_by_user:
                        if len(params_by_user[user]) > 0:
                            try:
                                user_params = fl.server.strategy.aggregate.aggregate(params_by_user[user])
                                user_model.set_weights(user_params)
                                user_model.save_weights(os.path.join(user_model_path, f"{user}"))
                                print("FIT save weights USER ", user)
                            except Exception as e:
                                print(e)
                                print("user model not found")
                else:
                    devices = option["devices"]
                    test_devices = args.test_devices
                    # For leave-one-device-out experiments
                    if test_devices in devices:
                        devices.remove(test_devices)
                    num_users = int(num_clients / len(devices))
                    print("test_devices: ", test_devices)
                    print("devices: ", devices)
                    print("num_users: ", num_users)
                    for user in range(num_users):
                        device_params = []
                        for device_idx, device in enumerate(devices):
                            cid = user * len(devices) + device_idx
                            if cid in self.sampled_indices:
                                try:
                                    user_model.load_weights(os.path.join(user_model_path, f"{user}_{device}"))
                                    with open(os.path.join(user_model_path, f'num_examples_{user}_{device}'), 'r') as f:
                                        num_examples = int(f.readline())
                                    device_weights = user_model.get_weights()
                                    device_params.append((device_weights, num_examples))

                                    # # Copy temporary device model weights for sampled devices
                                    tmp_path = os.path.join(user_model_path, f'device_{user}_{device}_tmp.*')
                                    for src in glob.glob(tmp_path):
                                        shutil.copyfile(src, src.replace("_tmp", ""))
                                    tmp_path = os.path.join(user_model_path, f'device_num_examples_{user}_{device}_tmp.*')
                                    for src in glob.glob(tmp_path):
                                        shutil.copyfile(src, src.replace("_tmp", ""))
                                except Exception as e:
                                    print("user device weights not found")

                            # if (device_selection == 'none') \
                            #         or (device_selection != 'none' and (cid in self.sampled_indices)):
                        # device_params = [param for param in user_parameters[user].values()]
                        # print(device_params)
                        if len(device_params) > 0:
                            try:
                                user_params = fl.server.strategy.aggregate.aggregate(device_params)
                                user_model.set_weights(user_params)
                                user_model.save_weights(os.path.join(user_model_path, f"{user}"))
                                print("FIT save weights USER ", user)
                            except Exception as e:
                                print("user model not found")
                # for device in option["devices"]:
                #     user_parameters[user][device] = new_user_params
            # aggregated_metrics.update(user_parameters)

            global_weights_prime = fl.server.strategy.aggregate.aggregate(global_weights_results)

            return weights_to_parameters(global_weights_prime), {}

        except Exception as e:
            print("Something wrong with AGG FIT")
            print(e)
            exit()
        # return super().aggregate_fit(rnd, results, failures)

    def aggregate_evaluate(
            self,
            rnd: int,
            results: List[Tuple[RayClientProxy, EvaluateRes]],
            failures: List[BaseException],
    ) -> Optional[float]:
        global device_accuracies, user_accuracies
        try:
            """Aggregate evaluation losses using weighted average"""
            print("AGGREGATE EVALUATE IS CALLED")
            if not results:
                return None
            #
            # global user_results

            # Weigh accuracy of each client by # examples used
            accuracies = [r.metrics["accuracy"] * r.num_examples for _, r in results]
            examples = [r.num_examples for _, r in results]
            losses = [r.loss for _, r in results]

            # if device_selection != 'none':
            #     # Update eval loss of all clients
            #     for client, r in results:
            #         self.devices[int(client.cid)].update_loss(r.loss)

            # Aggregate and print custom metric
            accuracy_aggregated = sum(accuracies) / sum(examples)
            loss_aggregated = np.mean(losses)
            print(f"Round {rnd} accuracy aggregated from client results: {accuracy_aggregated}")

            # User model results aggregation
            if user_model_path != "no_personal" and user_model_path != "local_finetuning":
                # User model performance
                # Weigh accuracy of each client by # examples used
                user_weighted_accuracies = [r.metrics["user_accuracy"] * r.num_examples for _, r in results]
                user_accuracies = [r.metrics["user_accuracy"] for _, r in results]
                user_losses = [r.metrics["user_loss"] for _, r in results]

                # Aggregate all user model test results
                user_weighted_accuracy_agg = sum(user_weighted_accuracies) / sum(examples)
                user_weighted_accuracy_var = np.var(user_weighted_accuracies)
                user_accuracy_agg = np.mean(user_accuracies)
                user_accuracy_var = np.var(user_accuracies)
                user_loss_agg = np.mean(user_losses)

                print(f"USER Round {rnd} WEIGHTED accuracy aggregated from client results: {user_weighted_accuracy_agg}")
                print(f"USER Round {rnd} accuracy aggregated from client results: {user_accuracy_agg}")

                # Device model performance
                # Weigh accuracy of each client by # examples used
                device_weighted_accuracies = [r.metrics["device_accuracy"] * r.num_examples for _, r in results]
                device_accuracies = [r.metrics["device_accuracy"] for _, r in results]
                device_losses = [r.metrics["device_loss"] for _, r in results]

                # Aggregate all user model test results
                device_weighted_accuracy_agg = sum(device_weighted_accuracies) / sum(examples)
                device_weighted_accuracy_var = np.var(device_weighted_accuracies)
                device_accuracy_agg = np.mean(device_accuracies)
                device_accuracy_var = np.var(device_accuracies)
                device_loss_agg = np.mean(device_losses)

                print(
                    f"DEVICE Round {rnd} WEIGHTED accuracy aggregated from client results: {device_weighted_accuracy_agg}")
                print(f"DEVICE Round {rnd} accuracy aggregated from client results: {device_accuracy_agg}")


            # Summary for TensorBoard
            step = rnd if rnd > 0 else ROUND + 1
            total_loss, dead_devices = compute_metrics(self.devices)
            with writer.as_default():
                # Global model loss and accuracy
                tf.summary.scalar("loss", loss_aggregated, step=step)
                tf.summary.scalar("accuracy", accuracy_aggregated, step=step)
                # User model loss and accuracy
                if user_model_path != "no_personal" and user_model_path != "local_finetuning":
                    tf.summary.scalar("user_loss", user_loss_agg, step=step)
                    tf.summary.scalar("user_weighted_accuracy", user_weighted_accuracy_agg, step=step)
                    tf.summary.scalar("user_weighted_accuracy_var", user_weighted_accuracy_var, step=step)
                    tf.summary.scalar("user_accuracy", user_accuracy_agg, step=step)
                    tf.summary.scalar("user_accuracy_var", user_accuracy_var, step=step)
                    tf.summary.scalar("device_loss", device_loss_agg, step=step)
                    tf.summary.scalar("device_weighted_accuracy", device_weighted_accuracy_agg, step=step)
                    tf.summary.scalar("device_weighted_accuracy_var", device_weighted_accuracy_var, step=step)
                    tf.summary.scalar("device_accuracy", device_accuracy_agg, step=step)
                    tf.summary.scalar("device_accuracy_var", device_accuracy_var, step=step)
                    tf.summary.scalar("total_loss", total_loss, step=step)
                    tf.summary.scalar("dead_devices", dead_devices, step=step)
            writer.flush()

            # Summary json
            """
            [
                {
                    rnd: 1,
                    global_loss: loss_aggregated,
                    global_accuracy: accuracy_aggregated,
                    user_loss: user_loss_agg,
                    user_accuracy: user_accuracy_agg,
                    total_loss: total_loss,
                    dead_devices: dead_devices,
                    device_snapshot: [
                        {
                            device_id: 1,
                            stat_util: d.get_stat_utility(),
                            device_util: d.get_device_utility(),
                            time_util: d.get_time_utility(),
                            overall_util: d.get_overall_utility(),
                            drain: d.drain,
                            t_local: d.t_local
                        }
                    ]
                }
            ]
            """
            print("MAX ROUND TIME")
            print(max(self.round_times))
            print("Sampled INDICES")
            print(self.sampled_indices)
            if user_model_path != "no_personal" and user_model_path != "local_finetuning":
                try:
                    with open("{}/{}.json".format(round_results_path, rnd), 'w') as json_file:
                        round_data = {
                            'rnd': rnd,
                            'global_loss': loss_aggregated,
                            'global_accuracy': accuracy_aggregated,
                            'global_accuracies': accuracies,
                            'global_examples': examples,

                            'user_loss': user_loss_agg,
                            'user_weighted_accuracy': user_weighted_accuracy_agg,
                            'user_weighted_accuracy_var': user_weighted_accuracy_var,
                            'user_accuracy': user_accuracy_agg,
                            'user_accuracy_var': user_accuracy_var,
                            'user_accuracies': user_accuracies,
                            'device_weighted_accuracy': device_weighted_accuracy_agg,
                            'device_weighted_accuracy_var': device_weighted_accuracy_var,
                            'device_accuracy': device_accuracy_agg,
                            'device_accuracy_var': device_accuracy_var,
                            'device_accuracies': device_accuracies,
                            'total_loss': total_loss,
                            'dead_devices': dead_devices,
                            'max_round_time': max(self.round_times),
                            'selected_devices': self.sampled_indices,
                            'device_snapshot': [
                                {
                                    'device_id': i,
                                    'stat_util': d.get_stat_utility(),
                                    'device_util': d.get_device_utility(),
                                    'time_util': d.get_time_utility(),
                                    'overall_util': d.get_overall_utility(),
                                    'drain': d.drain,
                                    't_local': d.t_local
                                } for i, d in enumerate(self.devices)
                            ],
                        }

                        json.dump(round_data, json_file)
                except:
                    print("ROUND DATA wrong!!!")
            else:
                with open("{}/{}.json".format(round_results_path, rnd), 'w') as json_file:
                    round_data = {
                        'rnd': rnd,
                        'global_loss': loss_aggregated,
                        'global_accuracy': accuracy_aggregated,
                        'global_accuracies': accuracies,
                        'global_examples': examples,
                        'total_loss': total_loss,
                        'dead_devices': dead_devices,
                        'device_snapshot': [
                            {
                                'device_id': i,
                                'stat_util': d.get_stat_utility(),
                                'device_util': d.get_device_utility(),
                                'time_util': d.get_time_utility(),
                                'overall_util': d.get_overall_utility(),
                                'drain': d.drain,
                                't_local': d.t_local
                            } for i, d in enumerate(self.devices)
                        ],
                        'max_round_time': max(self.round_times),
                        'selected_devices': list(self.sampled_indices),
                    }
                    json.dump(round_data, json_file)
            # if rnd >= ROUND:
            #     self.final_loss, self.final_drain = compute_metrics(self.devices)

            # Call aggregate_evaluate from base class (FedAvg)
        except Exception as e:
            print("Something wrong in AGG EVALUATE")
            print(e)
        return super().aggregate_evaluate(rnd, results, failures)

    # def configure_evaluate(self, rnd, parameters, client_manager):
    #     """Configure the next round of evaluation. Returns None since evaluation is made server side.
    #         You could comment this method if you want to keep the same behaviour as FedAvg."""
    #     return None


def fit_config(rnd: int):
    """
    Return training configuration dict for each round.
    """
    config = {
        "batch_size": BATCH_SIZE,
        # "local_epochs": 1 if rnd < 2 else LOCAL_EPOCHS,
        "local_epochs": LOCAL_EPOCHS,

        "learning_rate": LEARNING_RATE,
    }

    return config


def evaluate_config(rnd: int):
    """
    Return evaluation configuration dict for each round.
    Perform 5 local evaluation steps on each client (i.e., use 5 batches) during rounds 1 to 3,
    then increase to 10 local evaluation steps
    """

    val_steps = 5 if rnd < 4 else 10
    config =  {
        "val_steps": val_steps,
        "batch_size": BATCH_SIZE
    }
    return config

def get_hhar_devices(dataset_path):
    hhar_devices = {}
    files = os.listdir(dataset_path)
    user_list = set()
    for cid, file in enumerate(files):
        file = file.replace('.pickle', '')
        names = file.split('_')
        user = names[0]
        user_list.add(user)
        device = names[1]
        hhar_devices[str(cid)] = (user, device)

    return hhar_devices, user_list

def start_ray_server(args):
    global option, dataset
    if args.dataset == "hhar":
        dataset = "hhar"
        option = options.HHAROpt
        # dataset_path = "./hhar_datas"
    elif args.dataset == "realworld":
        dataset = "realworld"
        option = options.RealWorldOpt
        # dataset_path = "./realworld_data"
    elif args.dataset == "opportunity":
        dataset = "opportunity"
        option = options.OpportunityOpt
    elif args.dataset == "pamap2":
        dataset = "pamap2"
        option = options.PAMAP2Opt
    elif args.dataset == "femnist":
        dataset = "femnist"
        option = options.RealWorldOpt
    # dataset_path = os.path.join(dataset_path, str(args.take))

    global LEARNING_RATE, LOCAL_EPOCHS, BATCH_SIZE, ROUND, hparams, writer, user_model_path, num_clients, \
        device_selection, round_results_path, hhar_devices, user_list, sample_device_per
    # condition, DEVICE = eval(args.condition)
    LEARNING_RATE = args.learning_rate
    LOCAL_EPOCHS = args.local_epochs
    BATCH_SIZE = args.batch_size
    ROUND = args.round
    num_clients = args.num_clients
    sample_device_per = args.sample_device_per

    dataset_path = args.dataset_path
    tf_summary_path = args.tf_summary_path
    user_model_path = args.user_model_path
    if (not os.path.exists(user_model_path)) and (user_model_path != "no_personal" and user_model_path != "local_finetuning"):
        os.mkdir(user_model_path)
    device_selection = args.device_selection
    writer = tf.summary.create_file_writer(tf_summary_path)

    round_results_path = args.round_results_path

    if dataset != "femnist":
        input_shape = (option['seq_len'], option['input_dim'])

        # Initialize parameters and results for personalized user models
        # global user_parameters #, user_results
        # user_parameters = {user: {device: None for device in option['devices']} for user in option['users']}
        # user_results = {user: {device: None for device in option['devices']} for user in option['users']}

        if dataset == "hhar":
            hhar_devices, user_list = get_hhar_devices(dataset_path)
    # else:
    #     clients, groups, train_data, test_data = read_data('leaf/data/femnist/data/train', 'leaf/data/femnist/data/test')

    class HARRayClient(fl.client.NumPyClient):
        def __init__(self, cid: str, fed_dir_data: str):
            try:
                super().__init__()
                print("CLIENT INITT")
                # global user_parameters
                self.cid = cid
                if dataset != "femnist":
                    test_devices = args.test_devices
                    devices = option["models"] if dataset == "hhar" else option["devices"]
                    if test_devices in devices:
                        devices.remove(test_devices)

                    if dataset == "hhar":
                        device_idx = int(len(devices) / args.take)
                        self.user = hhar_devices[cid][0]
                        self.device = hhar_devices[cid][1]
                    else:
                        device_idx = int(len(devices) / args.take)
                        self.user = int(int(cid) / device_idx)
                        self.device = devices[int((int(cid) % device_idx) * args.take)]
                    # print("INIT CID: {}, USER: {}, DEVICE: {}".format(self.cid, self.user, self.device))

                    # Choose model based on args/hparams
                    if args.model == "DeepConvLSTM":
                        self.model = DeepConvLSTM(input_shape=input_shape, num_labels=option['num_class'],
                                                  LSTM_units=option['lstm_units'],
                                                  num_conv_filters=option['cnn_filters'],
                                                  batch_size=args.batch_size, F=option['F'], D=option['D'])
                        self.user_model = DeepConvLSTM(input_shape=input_shape, num_labels=option['num_class'],
                                                       LSTM_units=option['lstm_units'],
                                                       num_conv_filters=option['cnn_filters'],
                                                       batch_size=args.batch_size, F=option['F'], D=option['D'])
                        self.device_model = DeepConvLSTM(input_shape=input_shape, num_labels=option['num_class'],
                                                       LSTM_units=option['lstm_units'],
                                                       num_conv_filters=option['cnn_filters'],
                                                       batch_size=args.batch_size, F=option['F'], D=option['D'])
                        self.model.compile(args.optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
                        self.user_model.compile(args.optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
                        self.device_model.compile(args.optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
                    elif args.model == "SimCLR":
                        self.model = SimCLR(input_shape=input_shape, num_labels=option['num_class'])
                        self.user_model = SimCLR(input_shape=input_shape, num_labels=option['num_class'])
                        self.device_model = SimCLR(input_shape=input_shape, num_labels=option['num_class'])
                        self.model.compile(args.optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
                        self.user_model.compile(args.optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
                        self.device_model.compile(args.optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
                    elif args.model == "cnn":
                        self.model = FEMNIST_CNN()
                        self.user_model = FEMNIST_CNN()
                        self.device_model = FEMNIST_CNN()
                        self.model.compile(args.optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])
                        self.user_model.compile(args.optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])
                        self.device_model.compile(args.optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])


                    # user_parameters[self.user][self.device] = (self.user_model.get_weights(), 0)

                    # Load data by cid
                    print("PLEASE?")
                    file_name = os.path.join(dataset_path, f"{self.user}_{self.device}.pickle")
                    f = open(file_name, 'rb')
                    obj = pickle.load(f)
                    f.close()
                    # print("file name", file_name)
                    # print("obj: ", obj)
                    self.X_train = obj["X_train"]
                    self.y_train = obj["y_train"]
                    self.X_val = obj["X_val"]
                    self.y_val = obj["y_val"]
                    self.X_test = obj["X_test"]
                    self.y_test = obj["y_test"]


            except Exception as e:
                print("CLIENT INIT FAILED")
                print(e)

        def get_parameters(self):
            """Get parameters of the local model."""
            return self.model.get_weights()

        def fit(self, parameters, config):
            try:
                print("FIT!!!")
                """Train parameters on the locally held training set."""
                # Update local model parameters
                # global user_parameters #, user_results
                self.model.set_weights(parameters)
                self.model.save(os.path.join(user_model_path, f"global_model_{CURR_ROUND}"))
                # self.user_model.set_weights(user_parameters[self.user][self.device][0])
                num_examples_train = len(self.X_train)

                # Get hyperparameters for this round
                batch_size: int = config["batch_size"]
                epochs: int = config["local_epochs"]

                #### Global model training ####
                try:
                    ##### POC EFFICIENT #####
                    # num_rows = self.X_train.shape[0]
                    # num_samples = int((num_rows / batch_size) * 0.5) * batch_size
                    # sample_indices = np.random.choice(num_rows, size=num_samples, replace=False)
                    # history = self.model.fit(self.X_train[sample_indices, :], self.y_train[sample_indices, :],
                    #                     epochs=epochs, batch_size=batch_size)
                    #########################
                    log(INFO,
                        "X TRAIN SHAPE BEFORE TRAINING: %s",
                        self.X_train.shape
                        )
                    history = self.model.fit(self.X_train, self.y_train, epochs=epochs, batch_size=batch_size, verbose=3)
                except Exception as e:
                    print("Something wrong with fit", self.user, self.device)
                    # print(f"{self.user}-{self.device} X_train: ", self.X_train)
                    print(e)

                #### User model training ####
                if user_model_path != "no_personal" and user_model_path != "local_finetuning":
                    try:
                        self.user_model.load_weights(os.path.join(user_model_path, f"{self.user}"))
                    except:
                        print("FIT User weights not there", self.user)
                        self.user_model.set_weights(parameters)
                    try:
                        self.device_model.load_weights(os.path.join(user_model_path, f"device_{self.user}_{self.device}"))
                    except:
                        print("FIT Device weights not there", self.user, self.device)
                        self.device_model.set_weights(parameters)

                    candidate_lam = args.candidate_lam

                    train_dataset = tf.data.Dataset.from_tensor_slices((self.X_train, self.y_train)).batch(batch_size)
                    # val_dataset = tf.data.Dataset.from_tensor_slices((self.X_val, self.y_val)).batch(batch_size)
                    for epoch in range(epochs):
                        # Iterate over the batches of the datasets
                        for step, (x_batch_train, y_batch_train) in enumerate(train_dataset):

                            device_weights = self.device_model.get_weights()
                            user_weights = self.user_model.get_weights()
                            global_weights = self.model.get_weights()

                            with tf.GradientTape() as tape:
                                pred = self.user_model(x_batch_train, training=True)
                                loss = self.user_model.compiled_loss(y_batch_train, pred, regularization_losses=self.user_model.losses)
                            gradients = tape.gradient(loss, self.user_model.trainable_weights)
                            self.user_model.optimizer.apply_gradients(zip(gradients, self.user_model.trainable_weights))

                            user_weights_prime = copy.deepcopy(user_weights)
                            for layer in range(len(gradients)):
                                eff_grad = gradients[layer] + candidate_lam * (user_weights[layer] - global_weights[layer])
                                user_weights_prime[layer] = user_weights[layer] - LEARNING_RATE * eff_grad

                            self.user_model.set_weights(user_weights_prime)

                            # Train device models
                            with tf.GradientTape() as tape:
                                pred = self.device_model(x_batch_train, training=True)
                                loss = self.device_model.compiled_loss(y_batch_train, pred,
                                                                     regularization_losses=self.device_model.losses)
                            gradients = tape.gradient(loss, self.device_model.trainable_weights)
                            self.device_model.optimizer.apply_gradients(zip(gradients, self.device_model.trainable_weights))

                            device_weights_prime = copy.deepcopy(device_weights)
                            for layer in range(len(gradients)):
                                eff_grad = gradients[layer] + candidate_lam * (
                                            device_weights[layer] - global_weights[layer])
                                device_weights_prime[layer] = device_weights_prime[layer] - LEARNING_RATE * eff_grad

                            self.device_model.set_weights(device_weights_prime)

                    # # Save updated user model parameters
                    try:
                        self.user_model.save_weights(os.path.join(user_model_path, f'{self.user}_{self.device}'))
                        with open(os.path.join(user_model_path, f'num_examples_{self.user}_{self.device}'), 'w') as f:
                            f.write(str(num_examples_train))

                        self.device_model.save_weights(os.path.join(user_model_path, f'device_{self.user}_{self.device}_tmp'))
                        with open(os.path.join(user_model_path, f'device_num_examples_{self.user}_{self.device}_tmp'), 'w') as f:
                            f.write(str(num_examples_train))
                    except Exception as e:
                        print("NOT SAVED WEIGHTS", self.user, self.device)


                # Return updated model parameters and validation results
                parameters_prime = self.model.get_weights()

                try:
                    results = {
                        "loss": history.history["loss"][0],
                        "accuracy": history.history["accuracy"][0],
                        "val_loss": history.history["val_loss"][0],
                        "val_accuracy": history.history["val_accuracy"][0],
                    }
                    print("RESULTS: ", results)
                except:
                    results = {
                        "loss": history.history["loss"][0],
                        "accuracy": history.history["accuracy"][0],
                    }

            except Exception as e:
                print("Something wrong with FIT")
                print(e)
            return parameters_prime, num_examples_train, results

        def evaluate(self, parameters, config):
            try:
                # global user_parameters #, user_results
                """Evaluate parameters on the locally held test set"""
                # Update local model with global parameters
                self.model.set_weights(parameters)

                ##### Local Fine-Tuning #####
                if CURR_ROUND >= ROUND:
                    if user_model_path == "local_finetuning":
                        history = self.model.fit(self.X_train, self.y_train,
                                             epochs=LOCAL_EPOCHS, batch_size=BATCH_SIZE)
                #############################

                # Evaluate global model parameters on the local test data and return results
                loss, accuracy = self.model.evaluate(self.X_test, self.y_test, config["batch_size"])
                num_example_test = len(self.X_test)

                user_loss, user_accuracy = loss, accuracy
                device_loss, device_accuracy = loss, accuracy

                if user_model_path != "no_personal" and user_model_path != "local_finetuning":
                    try:
                        self.user_model.load_weights(os.path.join(user_model_path, f"{self.user}"))
                    except:
                        print("EVAL User weights not there")
                        self.user_model.set_weights(parameters)
                    try:
                        self.device_model.load_weights(os.path.join(user_model_path, f"device_{self.user}_{self.device}"))
                        if CURR_ROUND >= ROUND:
                            if user_model_path == "local_finetuning":
                                history = self.device_model.fit(self.X_train, self.y_train,
                                                         epochs=LOCAL_EPOCHS, batch_size=BATCH_SIZE)
                    except:
                        print("EVAL Device weights not there")
                        self.device_model.set_weights(parameters)
                    # self.user_model.set_weights(user_parameters[self.user][self.device][0])
                    user_loss, user_accuracy = self.user_model.evaluate(self.X_test, self.y_test, config["batch_size"])
                    device_loss, device_accuracy = self.device_model.evaluate(self.X_test, self.y_test, config["batch_size"])

            except Exception as e:
                print("Something wrong with EVALUATE")
                print(e)
            return loss, num_example_test, {"accuracy": accuracy, "user": self.user,
                                            "user_accuracy": user_accuracy, "user_loss": user_loss,
                                            "device_accuracy": device_accuracy, "device_loss": device_loss}

    if args.model == "DeepConvLSTM":
        model = DeepConvLSTM(input_shape=input_shape, num_labels=option['num_class'],
                                  LSTM_units=option['lstm_units'],
                                  num_conv_filters=option['cnn_filters'],
                                  batch_size=args.batch_size, F=option['F'], D=option['D'])
        model.compile(args.optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
    elif args.model == "SimCLR":
        model = SimCLR(input_shape=input_shape, num_labels=option['num_class'])
        model.compile(args.optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
    elif args.model == "cnn":
        model = FEMNIST_CNN()
        model.compile(args.optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    # strategy = SaveModelStrategy(
    strategy = PersonalizationStrategy(
        fraction_fit=1.0,
        fraction_eval=1.0,
        min_fit_clients=int(args.num_clients),  # Minimum # of clients to be sampled for the next round
        min_eval_clients=int(args.num_clients),  # Minimum # of clients to evaluate in the next round
        min_available_clients=int(args.num_clients),
        # Minimum # of clients that need to be connected to server before a training round can start
        on_fit_config_fn=fit_config,
        on_evaluate_config_fn=evaluate_config,
        initial_parameters=model.get_weights(),
        num_clients=int(args.num_clients)
    )

    fl.server.start_ray_simulation(
        pool_size=args.num_clients,
        data_partitions_dir="",  # path where data partitions for each client exist
        client_resources={'num_cpus': 1, 'num_gpus': 0.1},  #  compute/memory resources for each client
        # client_resources={'num_cpus': 1, 'num_gpus': 0.1},
        client_type=HARRayClient,
        strategy=strategy,
        ray_init_config={"num_cpus": 80, "num_gpus": 8, "log_to_driver": False},
        # ray_init_config={"num_cpus": 80, "num_gpus": 8},
        config={"num_rounds": args.round},
    )


if __name__ == "__main__":
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
    os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"
    os.environ["RAY_OBJECT_STORE_ALLOW_SLOW_STORAGE"] = "1"

    parser = argparse.ArgumentParser(description="Flower")
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument('--learning_rate_decay', default='cosine', type=str, choices=['cosine', 'none'],
                        help='the learning rate decay function')
    parser.add_argument('--gpu_device', default='', type=str,
                        help='set the gpu device')
    parser.add_argument('--learning_rate', default=0.001, type=float,
                        help='the initial learning rate during contrastive learning')
    parser.add_argument('--optimizer', default='adam', type=str, choices=['adam', 'sgd', 'rmsprop'],
                        help='optimizer to use for training')
    parser.add_argument('--take', default=1.0, type=float,
                        help='percentage of training samples to take from the dataset. To use all samples, set 1.0 (100%)')
    parser.add_argument('--round', default=4, type=int,
                        help='number of aggregation rounds')
    parser.add_argument('--batch_size', default=16, type=int,
                        help='batch size')
    parser.add_argument('--local_epochs', default=5, type=int,
                        help='number of epochs for training at client')

    parser.add_argument('--condition', default='hetero', type=str,
                        help="experiment condition")
    # parser.add_argument('--device', default='forearm', type=str,
    #                     help='device/device position to test')

    parser.add_argument('--tf_summary_path', default='./results/logs', type=str,
                        help="tensorboard logs path")

    parser.add_argument('--model', default='DeepConvLSTM', type=str,
                        help='which model architecture')

    parser.add_argument('--num_clients', default=105, type=int,
                        help="Number of available clients")
    parser.add_argument('--user_model_path', default='./checkpoints', type=str,
                        help="Path for saving intermediate user models")
    parser.add_argument('--dataset_path', default='./realworld_data/', type=str,
                        help="Path for partitioned dataset")
    parser.add_argument('--device_selection', default='select_devices', type=str,
                        help="Device selection strategy - overall, device, stat, time, random, none")
    parser.add_argument('--time_aligned', default='time_aligned', type=str,
                        help="aligned or not_aligned")
    parser.add_argument('--candidate_lam', default=1.0, type=float,
                        help='lambda for global model regularization in personalization')
    parser.add_argument('--round_results_path', default='./results', type=str,
                        help='path to save results per round')
    parser.add_argument('--test_devices', default='trained', type=str,
                        help='test devices')
    parser.add_argument('--sample_device_per', default=0.5, type=float,
                        help='percentage of devices to be sampled in each round')

    args = parser.parse_args()

    # start_server(args)
    start_ray_server(args)
