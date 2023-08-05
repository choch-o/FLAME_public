import numpy as np, random, math

class Device:
    def __init__(self, device_config, device_speed, num_samples=-1, init_loss=10.0, init_drain=0.0,
                 dataset="realworld"):
        self.num_samples = num_samples
        self.drain = init_drain
        self.loss = init_loss
        self.t_local = device_config.n_epoch_time + MODEL_SIZE/device_speed[0] + MODEL_SIZE/device_speed[1]
        self.t_ul = MODEL_SIZE/device_speed[1]
        self.t_ud = device_config.n_epoch_time
        '''
        60 J is the estimated energy consumption of UL+DL on a 4G network. 
        Reference:  https://www.mdpi.com/1996-1073/12/1/184/pdf 
        '''
        self.drain_per_round = device_config.energy_drain + 60 
        self.last_round = 0
        self.n_selected = 0

        self.dataset = dataset

    def update_local_state(self, round):
        drain_factor = 1.0
        if self.dataset == "opportunity":
            drain_factor = 0.5
        elif self.dataset == "pamap2":
            drain_factor = 87 / 130

        # drain_factor *= 0.7
        #
        self.last_round = round
        self.drain = self.drain + self.drain_per_round * drain_factor
        self.n_selected += 1

    def update_loss(self, new_loss):
        self.loss = new_loss

    def update_num_samples(self, num_samples):
        self.num_samples = num_samples

    def get_stat_utility(self):
        # print("get_stat_utility: ", self.num_samples)
        if self.num_samples <= 0:
            return 0
        return self.num_samples * math.sqrt(self.loss * self.loss / self.num_samples)

    def get_device_utility(self):
        # print("get_device_utility: ", self.drain)
        if self.drain < MAX_DRAIN:
            return math.log(MAX_DRAIN / self.drain)
        else:
            return 0

    def get_time_utility(self):
        # print("get_time_utility: ", self.t_local)
        if self.t_local <= T_THRESHOLD:
            return 1.0
        else:
            return T_THRESHOLD * ALPHA/self.t_local

    def get_oort_time_utility(self):
        # print("get_time_utility: ", self.t_local)
        if self.t_local <= T_THRESHOLD:
            return 1.0
        else:
            return (T_THRESHOLD / self.t_local)**ALPHA_OORT

    def get_oort_utility(self):
        return self.get_stat_utility() * self.get_oort_time_utility()

    def get_fedcs_utility(self):
        return self.t_ul + self.t_ud

    def get_overall_utility(self):
        return self.get_stat_utility() * self.get_device_utility() * self.get_time_utility()

# def sample_values(device_id):
#     loss = 10000.0 #Set a high initial loss 
#     t_local = <>
#     num_samples = <>
#     drain = <>
#     return num_samples, loss, t_local


def select_devices_UAC(current_round, num_devices=600, strategy='random', devices=[],
                       num_device_per_user=7):
    if strategy=='random':
        # Exclude dead devices
        l = [i for i, d in enumerate(devices) if d.get_device_utility() > 0]
        # l = list(range(0, len(devices)))
        if num_devices < len(l):
            return random.sample(l, num_devices)
        else:
            return l
    elif strategy == 'first_k':
        return list(range(0,num_devices))
    elif strategy == 'stat':
        devices = [(i, d) for i, d in enumerate(devices) if d.get_device_utility() > 0]
        utilities = [(index, d.get_stat_utility()) for index, d in devices]
        utilities.sort(key=lambda x: x[1], reverse=True)
        indices = [u[0] for u in utilities[:num_devices] if u[1]>0]
        if len(indices) < num_devices:
            print("Warning: Less than ", num_devices , " devices found that satisfy the selection criteria")
        return indices
    elif strategy == 'device':
        devices = [(i, d) for i, d in enumerate(devices) if d.get_device_utility() > 0]
        utilities = [(index, d.get_device_utility()) for index, d in enumerate(devices)]
        utilities.sort(key=lambda x: x[1], reverse=True)
        indices = [u[0] for u in utilities[:num_devices] if u[1]>0]
        if len(indices) < num_devices:
            print("Warning: Less than ", num_devices , " devices found that satisfy the selection criteria")
        return indices
    elif strategy == 'time':
        devices = [(i, d) for i, d in enumerate(devices) if d.get_device_utility() > 0]
        utilities = [(index, d.get_time_utility()) for index, d in devices]
        utilities.sort(key=lambda x: x[1], reverse=True)
        indices = [u[0] for u in utilities[:num_devices] if u[1]>0]
        if len(indices) < num_devices:
            print("Warning: Less than ", num_devices , " devices found that satisfy the selection criteria")
        return indices
    elif strategy == 'fedcs':
        devices = [(i, d) for i, d in enumerate(devices) if d.get_device_utility() > 0]
        utilities = [(index, d.get_fedcs_utility()) for index, d in devices]
        utilities.sort(key=lambda x: x[1], reverse=False)
        indices = []
        t = 0
        for u in utilities:
            if t < T_ROUND:
                t += u[1]
                indices.append(u[0])
        return indices
    elif strategy == 'oort':
        devices = [(i, d) for i, d in enumerate(devices) if d.get_device_utility() > 0]
        utilities = [(index, d.get_oort_utility()) for index, d in devices]
        utilities.sort(key=lambda x: x[1], reverse=True)
        indices = [u[0] for u in utilities[:num_devices] if u[1] > 0]
        if len(indices) < num_devices:
            print("Warning: Less than ", num_devices, " devices found that satisfy the selection criteria")
        return indices
    elif strategy == 'average_utility_ours':
        K = 4
        utilities = [d.get_overall_utility() for index, d in enumerate(devices)]
        top_k_utilities_per_user = -np.sort(-np.array(utilities).reshape(-1, 7))[:,:K]
        top_k_indices_per_user = np.argsort(-np.array(utilities).reshape(-1, 7))[:,:K]
        avg_user_utilities = np.mean(top_k_utilities_per_user, axis=1)
        top_users = np.argsort(-np.array(avg_user_utilities))[:num_devices//K]
        top_devices = top_k_indices_per_user[top_users, ]
        device_indices = top_devices + np.expand_dims(top_users * 7, axis=1)
        device_indices = device_indices.flatten()
        device_indices = [index for index in device_indices if utilities[index] > 0]

        if len(device_indices) < num_devices:
            print("Warning: Less than ", num_devices , " devices found that satisfy the selection criteria")
        return device_indices
        
    elif strategy == 'best_utility_ours':
        print(" num device per user in here: ", num_device_per_user)
        K=4
        utilities = [(index, d.get_overall_utility()) for index, d in enumerate(devices)]
        utilities.sort(key=lambda x: x[1], reverse=True)
        selected_devices = [d[0] for d in utilities[:num_devices] if d[1] > 0]
        user_visited = {}

        added_devices = []

        for cid in selected_devices:
            user_idx = cid // num_device_per_user
            if user_idx not in user_visited:
                peer_count = get_peer_count(selected_devices, user_idx, num_device_per_user)
                if peer_count < K - 1: 
                    #need more devices for this user. 
                    how_many = K - 1 - peer_count
                    new_peers = get_peers(utilities[num_devices:], user_idx, how_many, num_device_per_user)
                    if len(new_peers) > 0:
                        del selected_devices[-len(new_peers):]
                    added_devices.extend(new_peers)
                user_visited[user_idx] = True

        selected_devices.extend(added_devices)
        return selected_devices


def get_peer_count(devices, user_idx, num_device_per_user):
    return len([d for d in devices if d // num_device_per_user == user_idx]) - 1

def get_peers(utilities, user_idx, how_many, num_device_per_user):
    return [d[0] for d in utilities if d[0] // num_device_per_user == user_idx][:how_many]


def select_devices(current_round, num_devices=600, strategy='random', devices=[]):
    if strategy=='random':
        l = list(range(0, len(devices)))
        return random.sample(l, num_devices)
    elif strategy == 'first_k':
        return list(range(0,num_devices))
    elif strategy == 'stat':
        utilities = [(index, d.get_stat_utility()) for index, d in enumerate(devices)]
        utilities.sort(key=lambda x: x[1], reverse=True)
        indices = [u[0] for u in utilities[:num_devices] if u[1]>0]
        if len(indices) < num_devices:
            print("Warning: Less than ", num_devices , " devices found that satisfy the selection criteria")
        return indices
    elif strategy == 'device':
        utilities = [(index, d.get_device_utility()) for index, d in enumerate(devices)]
        utilities.sort(key=lambda x: x[1], reverse=True)
        indices = [u[0] for u in utilities[:num_devices] if u[1]>0]
        if len(indices) < num_devices:
            print("Warning: Less than ", num_devices , " devices found that satisfy the selection criteria")
        return indices
    elif strategy == 'time':
        utilities = [(index, d.get_time_utility()) for index, d in enumerate(devices)]
        utilities.sort(key=lambda x: x[1], reverse=True)
        indices = [u[0] for u in utilities[:num_devices] if u[1]>0]
        if len(indices) < num_devices:
            print("Warning: Less than ", num_devices , " devices found that satisfy the selection criteria")
        return indices
    else:
        utilities = [(index, d.get_overall_utility()) for index, d in enumerate(devices)]
        utilities.sort(key=lambda x: x[1], reverse=True)
        indices = [u[0] for u in utilities[:num_devices] if u[1]>0]
        if len(indices) < num_devices:
            print("Warning: Less than ", num_devices , " devices found that satisfy the selection criteria")
        return indices


def compute_metrics(devices):
    total_loss = np.mean([d.loss * d.num_samples for d in devices])
    # total_drain = np.mean([d.drain for d in devices])
    dead_devices = sum(d.drain >= MAX_DRAIN for d in devices)
    return total_loss, dead_devices


random.seed(39)
ALPHA = 0.5
ALPHA_OORT = 2.0
T_THRESHOLD = 30 #seconds
T_ROUND = 300 # seconds = 5min
#NOTE: We can reduce the drain threshold (currently set to 10%) if very few devices are going dead after training. Our model is quite simple and not too power hungry, so it may not have that high power impact. 
DRAIN_THRESHOLD = 0.1
MAX_DRAIN = 3000 * 3.6 * 3.7 * DRAIN_THRESHOLD #mAH battery capacity x joules conversion x voltage x drain threshold

MODEL_SIZE = 150 * 8 * 7/ 1024.0 #total size in Mb. Model is 150 KB for each device and there are 7 devices (in the worst case when all 7 are sampled)
STRATEGIES = ['random', 'first_k',  'stat', 'device', 'time', 'ours']
