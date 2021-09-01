import gym
import sys
import pybulletgym

import pickle as pkl
import json
import os
import numpy as np
import tensorflow as tf
from packaging import version
import tensorflow_probability as tfp

from utils.constants import obs_indices, sorted_neuron_type_indices, \
    e_class_priorities, i_class_priorities
import string
import re
import random
import simmanager


def exp_convolve(tensor, decay=.8, reverse=False, initializer=None, axis=0):
    rank = len(tensor.get_shape())
    perm = np.arange(rank)
    perm[0], perm[axis] = perm[axis], perm[0]
    tensor = tf.transpose(tensor, perm)

    if initializer is None:
        initializer = tf.zeros_like(tensor[0])

    def scan_fun(_acc, _t):
        return _acc * decay + _t

    filtered = tf.scan(scan_fun, tensor, reverse=reverse, initializer=initializer)

    filtered = tf.transpose(filtered, perm)
    return filtered


def getDimsOfTask(name):
    env = gym.make(name)
    # env.reset()
    # env.step(env.action_space.sample())
    obs_len = len(obs_indices[name])
    action_shape = env.action_space.shape
    action_len = 1 if len(action_shape) == 0 else action_shape[0]
    return obs_len, action_len


def getValueRangeOfTask():
        pi_half = np.pi / 2
        pi_quat = np.pi / 4
        mins = np.array([0, -pi_quat, 0, -pi_quat, -0.2 - pi_half, -pi_quat, -0.2 - pi_half, -pi_quat, 0],
                        dtype=np.float32)
        maxs = np.array([0.75, pi_quat, pi_half, pi_quat, -0.2, pi_quat, -0.2, pi_quat, pi_half], dtype=np.float32)
        return mins, maxs


def getSubObs(observation, task_name):
    res = tf.gather(observation, obs_indices[task_name], axis=-1)
    return res

class dotdict(dict):
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

    def flag_values_dict(self):
        return self

def flattern_time_dim(x):
    if len(x.shape) > 2:
        batch_size, n = x.shape[0], x.shape[-1]
        return tf.reshape(x, (batch_size, -1, n))
    else:
        return x


def gen_stats_str(text, mu, sig):
    mu = mu.numpy() if isinstance(mu, tf.Variable) else mu
    sig = sig.numpy() if isinstance(sig, tf.Variable) else sig
    text_len = len(text)
    mu_min, mu_mean, mu_stddev, mu_max = np.min(mu), np.mean(mu), np.std(mu), np.max(mu)
    sig_min, sig_mean, sig_stddev, sig_max = np.min(sig), np.mean(sig), np.std(sig), np.max(sig)
    out_str = f"{text}mu  min/max: [{mu_min:.3f}, {mu_max:.3f}] mean: {mu_mean:.3f}, stddev: {mu_stddev:.3f}\n" \
              f"{'':<{text_len}}sig min/max: [{sig_min:.3f}, {sig_max:.3f}] mean: {sig_mean:.3f}, stddev: {sig_stddev:.3f}"
    return out_str




def linspace(start, end, n):
    if version.parse(tf.__version__) >= version.parse('2.3.0'):
        return tf.linspace(start, end, n)
    else:
        steps = tf.range(n, dtype=tf.float32)
        step_size = (end - start) / (n - 1)
        res = steps[..., None] * step_size + start
        return res


def pop_encode_prop(x, min_values=-1.0, max_values=1.0, n=21, stds=0.1):
    """
    x: input tensor: [batch_size, time, m]
    min_values: tensor:[m]
    max_values: tensor: [m]
    stds: tensor: [m]
    """
    batch_size = tf.shape(x)[0]
    time = tf.shape(x)[1]
    m = tf.shape(x)[2]

    if len(tf.shape(min_values)) == 0:
        min_values = tf.ones(m) * min_values
        max_values = tf.ones(m) * max_values

    if len(tf.shape(stds)) == 1:
        stds = tf.ones((m, n)) * stds[:, None]
    else:
        stds = tf.ones((m, n)) * stds

    centers = tf.transpose(linspace(min_values, max_values, n))

    distr = tfp.distributions.Normal(loc=centers, scale=stds)
    x = tf.repeat(x[..., None], n, axis=-1)
    normalized, _ = tf.linalg.normalize(distr.prob(x), axis=3)  # [batch_size, time, m, n]
    normalized = tf.reshape(normalized, (batch_size, time, m, n))
    return normalized


# @tf.function
def pop_encode(x, t=10, min_values=-1, max_values=1, n=21, k=1):
    """
    x: input tensor: [batch_size, time, m]
    n: number of Gauss curves
    k: number of neurons which are sensitive to a Gauss curve
    mins, maxs: value bounds of the population. Can be scalar or m dimensional.
    """
    x = tf.cast(x, tf.float32)
    if len(tf.shape(x)) == 2:
        x = x[:, None]
    stds = (max_values - min_values) / (2 * (n - 1))
    p = pop_encode_prop(x, min_values, max_values, n, stds)
    p = tf.repeat(p, t, axis=1)  # repeat time dim t times
    p = tf.repeat(p[..., None], k, axis=-1)  # repeat k dim k times
    batch_size, time, m = x.shape
    rand_shape = tf.TensorShape([batch_size, time * t, m, n, k])
    rand = tf.random.uniform(shape=rand_shape)
    z = tf.cast(rand < p, tf.float32)
    z = tf.reshape(z, (batch_size, time * t, m * n * k))
    return z



def rate_decode(z, n_val=1, minval=-1.0, maxval=1.0, tau=10):
    n_types = 2
    batch_size, n_ts, total_n_neurons = z.shape.as_list()

    n_neurons = total_n_neurons // n_types // n_val
    z = tf.reshape(z, (batch_size, n_ts, n_val, n_types, n_neurons))
    time_decay = tf.exp(-tf.range(n_ts, dtype=tf.float32) / tau)[::-1][None, :, None, None]

    # sum over time and average over neurons from the same type
    avg_n_spikes = tf.reduce_mean(z, axis=-1)  # avg over n_pop_out_size
    avg_n_spikes = avg_n_spikes * time_decay  # apply decay
    avg_n_spikes = tf.reduce_sum(avg_n_spikes, axis=1)

    # compute weighted spike rate and then compute sum of both directions
    normalizer = tf.reduce_max(tf.abs(avg_n_spikes), -1)[..., None]
    res = tf.math.divide_no_nan(avg_n_spikes * tf.constant([minval, maxval]), normalizer)
    res = tf.reduce_sum(res, -1)

    return res




def create_rand_seq_with_sum(l, s, minimum=1):
    '''
    Create a random sequence containing l elements which sum to s
    Idea: create random splits and take random length of splits as output
    '''
    assert s // minimum >= l
    possible_splits = np.arange(1, s // minimum) * minimum
    np.random.shuffle(possible_splits)
    rand_split_indices = np.sort(possible_splits[:l - 1])
    rand_split_indices = np.concatenate((np.array([0]), rand_split_indices, np.array([s])))
    res = np.diff(rand_split_indices)
    assert np.sum(res) == s
    return res


def quantize_weights(weights, bits, keep_signs=True, const=1):
    """
    Use keep_signs False if you intent on applying a sign matrix later.
    """
    if bits == 0:
        res = const * tf.ones_like(weights)
    elif bits == 1:
        res = tf.cast(weights > 0, tf.float32) + 1
    else:
        res = tf.quantization.fake_quant_with_min_max_args(tf.abs(weights),
                                                           min=1,
                                                           max=2 ** bits,
                                                           num_bits=bits, narrow_range=False)
    if keep_signs:
        res *= tf.sign(weights)
    return res


# current dir is innate/tasks
def load_neuron_params(path='../neuron_types_data.pkl'):
    if not os.path.exists(path):
        path = path[3:]  # strip away '../', for supercomputer
    with open(path, 'rb') as f:
        d = pkl.load(f)
    return d




def sort_params(params, e_frac=0.75, n_types=20, n_out_types=2):
    n_e = int(e_frac * n_types)
    n_out_types = int(n_out_types)
    n_i = n_types - n_e

    for k, v in params.items():
        params[k] = np.array(v)[sorted_neuron_type_indices]

    # pick e5 with lowest highest firing counf. This way it can have good spike rates.
    e5_out_indices = [i for i, s in enumerate(params['neuron_type_names']) if s.startswith('e5')]
    e5_out_index = e5_out_indices[0]

    def select_diverse(e_i, n):
        # build a dict with name: [ids,]
        class_names = {}
        for i in range(len(params['V_th'])):
            if params['e_i'][i] == e_i:
                current_name = params['neuron_type_names'][i]
                # new classname
                if current_name not in class_names:
                    class_names[current_name] = []
                class_names[current_name].append(i)

        type_indices = []
        if e_i == 1:
            class_priorities = e_class_priorities
        else:
            class_priorities = i_class_priorities

        for i in range(n):
            failcount = 0
            for class_name in class_priorities:
                v = class_names[class_name]
                try:
                    if v[i] == e5_out_index:  # lets skip the one for the output
                        continue
                    type_indices.append(v[i])
                    failcount = 0
                    if len(type_indices) == n:
                        return type_indices
                except IndexError:
                    failcount += 1
                    if failcount == len(class_names):
                        print("Not enough classes")
                        raise ValueError
                    pass

    e_indices = select_diverse(1, n_e)
    i_indices = select_diverse(-1, n_i)
    indices = [*e_indices, *i_indices]

    indices.extend([e5_out_index] * n_out_types)

    # apply new indices inside dict
    for k, v in params.items():
        params[k] = np.array(v)[indices]
    return params


class ObservationDelayBuffer:
    """
    Saves all observations in an array.
    self._buffer[0] is the oldest observation, which will be returned next.
    Meant to be used for Tensors.
    A call to put need to preceed a call to pop
    """

    def __init__(self, length):
        self.length = length
        self._buffer = [None for _ in range(length)]

    def put(self, observation):
        self._buffer.append(observation)

    def get(self):
        try:
            res = self._buffer.pop(0)
            # warm up phase, return first non null entry
            if res is None:
                return next(item for item in self._buffer if item is not None)
            return res
        except IndexError:
            raise UserWarning('First call put(), then get().')

    def clear(self):
        self._buffer.clear()


class RunningSpikeVar:
    def __init__(self, use_avg=True, tau=30, time_axis=1):
        self.decay = np.exp(-1 / tau)
        self.time_axis = time_axis
        self.use_avg = use_avg

        self.m = None
        self.s = None
        self.k = 0

    def _put(self, activity):
        self.k += 1
        if self.k == 1:
            self.m = activity
            self.s = 0
        else:
            new_m = self.m + (activity - self.m) / self.k
            self.s = self.s + (activity - self.m) * (activity - new_m)
            self.m = new_m

    def put(self, z):
        # filter
        if self.use_avg:
            activity = tf.reduce_sum(z, axis=self.time_axis)  # sum is larger than mean,
            self._put(activity)

        else:
            activity = z
            n_ts = activity.shape[self.time_axis]
            for i in range(n_ts):
                activity_i = tf.gather(activity, i, axis=self.time_axis)
                self._put(activity_i)

    def get_var(self):
        if self.k > 1:
            return self.s / (self.k - 1)
        else:
            return None

    def reset(self):
        self.k = 0


class QuietCall:
    """
    Silence a function.
    """

    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout


def get_sparsity(model):
    n_synapses = model.sparsity_info['in_n_synapses'] + model.sparsity_info['rec_n_synapses']
    return n_synapses / (model.n_in_neurons * model.n_rec_neurons + model.n_rec_neurons * model.n_glif_neurons)




def load_flags(f):
    flags_file = os.path.join(f.restore_from, 'data/config.json')
    with open(flags_file, 'r') as file:
        new_f = dotdict(json.load(file))  # restore settings

        # flags to override
        new_f.restore_from = f.restore_from  # some new settings for analyzing have to be kept.
        new_f.eval_only = f.eval_only
        new_f.eval_every = f.eval_every
        new_f.plot_every = f.plot_every
        new_f.save_every = f.save_every
        new_f.save_videos = True
        new_f.comment = f.comment
        new_f.n_epochs = f.n_epochs
        new_f.max_steps = f.max_steps


        # backwards compatibility zone
        if new_f.n_pop_in_size == None:
            new_f.n_pop_in_size = 1
            new_f.n_pop_out_size = 1
        if new_f.less_excitable == None:
            new_f.less_excitable = f.less_excitable

        if f.eval_only:
            new_f.batch_size = f.batch_size
            new_f.offspring_size = f.offspring_size
            new_f.offspring_sample_size = f.offspring_sample_size
        return new_f


def create_new_simmanager(f):
    identifier = ''.join(random.choice(string.ascii_lowercase + string.digits) for _ in range(4))
    sim_name = 'run_{}'.format(identifier)

    root_dir = os.path.join(f.restore_from, 'eval_runs')
    os.makedirs(root_dir, exist_ok=True)
    sm = simmanager.SimManager(sim_name, root_dir, write_protect_dirs=False, tee_stdx_to='output.log')
    return sm


def print_old_logs(tee_print, f):
    with open(os.path.join(f.restore_from, 'logs/print.log'), 'r') as file:
        tee_print('Logs from restored exp:')
        logs = file.read()
        tee_print(logs)
        tee_print('Logs from new exp:')
        n_epochs_re_str = ' (\d+) @'
        n_epochs_regex = re.compile(n_epochs_re_str)
        n_epochs = int(n_epochs_regex.findall(logs)[-1])
        return n_epochs
