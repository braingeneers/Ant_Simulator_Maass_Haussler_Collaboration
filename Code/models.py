import numpy as np
import time
import tensorflow as tf
from copy import copy

import utils.util as util
from utils.constants import E_neuron_params, I_neuron_params, target_n_synapses


def spike_function(v, v_th, dtype=tf.float32):
    z_ = tf.greater(v, v_th)
    z_ = tf.cast(z_, dtype)

    return tf.identity(z_, name="spike_function")


def gather2D(tensor, indices):
    old_shape = indices.shape[:-1]
    tensor = tf.gather_nd(tensor, indices, batch_dims=1)
    return tf.reshape(tensor, old_shape)


class GlifAscPscTrainableCellDense(tf.keras.layers.Layer):
    def __init__(self, neuron_params, flags, dt=1.,
                 n_in_neurons_per_type=None,  # list of number of input neurons per type
                 n_out_neurons_per_type=None,  # list of number of output neurons per type
                 n_receptors=None,
                 observation_length=4):
        super().__init__()
        self.start_time = time.time()

        self.batch_size = flags.batch_size
        self.offspring_size = flags.offspring_size
        self.offspring_sample_size = flags.offspring_sample_size
        self.all_flat_size = self.batch_size * self.offspring_size * self.offspring_sample_size
        self.flags = flags
        self.observation_length = observation_length
        self.max_delay = 3

        # later inits
        self.disconnect_mask = None
        self.type_mask = None
        self.in_weights = None
        self.rec_weights = None

        # init type related variables
        self.n_rec_types = flags.n_rec_types
        self.n_in_neurons_per_type = n_in_neurons_per_type

        self.n_rec_neurons = flags.n_neurons
        self.n_rec_neurons_per_type = np.tile(util.create_rand_seq_with_sum(self.n_rec_types, self.n_rec_neurons)[None],
                                              (self.offspring_size, 1))
        self.n_out_neurons_per_type = n_out_neurons_per_type

        self.n_in_types = len(self.n_in_neurons_per_type)
        self.n_out_types = len(self.n_out_neurons_per_type)
        self.n_types = self.n_in_types + self.n_rec_types + self.n_out_types

        print(f'Total free parameters are {self.n_types * self.n_rec_types + self.n_rec_types + self.n_out_types + 3}')

        self.trained_output_locations = tf.tile(self.middle_linspace(tf.cast(0, tf.float32), flags.spatial_height,
                                                                     self.n_out_types)[None], (self.offspring_size, 1))

        self._params = util.sort_params(neuron_params, n_types=self.n_rec_types, n_out_types=self.n_out_types,
                                        e_frac=flags.e_frac)

        new_params = {}
        el = len(np.argwhere(self._params['e_i'] == 1)) - self.n_out_types
        il = len(np.argwhere(self._params['e_i'] == -1))
        ol = self.n_out_types
        for k, v in E_neuron_params.items():
            e_params = np.repeat(np.array(v), axis=0, repeats=el)
            o_params = np.repeat(np.array(v), axis=0, repeats=ol)
            i_params = np.repeat(np.array(I_neuron_params[k]), axis=0, repeats=il)
            new_params[k] = np.concatenate([e_params, i_params, o_params], axis=0)
        self._params = new_params

        if n_receptors is None:
            self._n_receptors = self._params['tau_syn'].shape[1]
        else:
            self._n_receptors = n_receptors
        self._dt = dt

        # init neuron related variables and parameters
        self.init_neuron_vars()
        self.init_neuron_params()

        # evo inits
        self.init_optimization_parameters(flags.eta_mu, flags.eta_sig_coeff)
        self.init_genome(flags)
        self.init_samples_from_genome()
        self.update_spatial_structure()
        # init masks
        self.strong_first_input_type = True
        self.init_masks()
        self.set_deletion_frac(0)

        self.sparsity_info = {}

    def set_deletion_frac(self, frac):
        _deletion_mask = np.ones(shape=(self.n_glif_neurons))
        num_deletions = int(self.n_glif_neurons * frac)

        del_indices = np.arange(self.n_glif_neurons)
        np.random.shuffle(del_indices)
        del_indices = del_indices[:num_deletions]
        _deletion_mask[del_indices] = 0
        self.deletion_mask.assign(_deletion_mask)

    def middle_linspace(self, start, end, num):
        return tf.linspace(start, end, num + 2)[1:-1]

    def update_spatial_structure(self):
        flags = self.flags
        zero = tf.cast(0, tf.float32)

        n_neurons_per_type = self.n_in_neurons // self.observation_length
        in_x_coord = tf.zeros(shape=(self.offspring_size, self.n_in_neurons))
        in_y_coord = tf.tile(self.middle_linspace(zero, flags.spatial_height, n_neurons_per_type)[None],
                             (self.offspring_size, self.observation_length))

        rec_x_coord_range = tf.range(flags.layer_dist, (self.n_rec_types + 1) * flags.layer_dist, flags.layer_dist)
        rec_y_coord = []
        rec_x_coord = []
        for i in range(self.offspring_size):
            y_rec_neurons = []
            for n in self.n_rec_neurons_per_type[i]:
                y_rec_neurons.append(self.middle_linspace(zero, flags.spatial_height, n))
            rec_x_coord.append(tf.repeat(rec_x_coord_range, self.n_rec_neurons_per_type[i]))
            rec_y_coord.append(tf.concat(y_rec_neurons, 0))
        rec_y_coord = tf.stack(rec_y_coord)
        rec_x_coord = tf.stack(rec_x_coord)

        out_x_coord = tf.constant((self.n_rec_types + 1) * flags.layer_dist,
                                  shape=(self.offspring_size, self.n_out_neurons))
        # distribute output neurons evenly
        if self.flags.train_output_locations:
            out_y_coord = tf.repeat(self.trained_output_locations, repeats=self.n_out_neurons // self.n_out_types,
                                    axis=-1)
        else:
            out_y_coord = tf.tile(self.middle_linspace(zero, flags.spatial_height, self.n_out_neurons)[None],
                                  (self.offspring_size, 1))

        x_coord = tf.concat((in_x_coord, rec_x_coord, out_x_coord), axis=-1)
        y_coord = tf.concat((in_y_coord, rec_y_coord, out_y_coord), axis=-1)
        self.coords = tf.stack((x_coord, y_coord), axis=-1)

        tiled_coords = tf.tile(self.coords[:, :, None], (1, 1, self.n_neurons, 1))
        diff_mat = tiled_coords - tf.transpose(tiled_coords, (0, 2, 1, 3))
        self.distance_matrix = tf.sqrt(tf.square(diff_mat[..., -1]))  # only consider y axis for distances
        self.in_distance_matrix = self.distance_matrix[:, :self.n_in_neurons, self.n_in_neurons:]
        self.rec_distance_matrix = self.distance_matrix[:, self.n_in_neurons:, self.n_in_neurons:]

    def init_neuron_vars(self):
        self.n_i_types = np.sum((self._params['e_i'] == -1).astype(np.int32))
        self.n_e_types = self.n_rec_types - self.n_i_types
        # compute number of input, recurrent and output neurons
        self.n_in_neurons = np.sum(self.n_in_neurons_per_type)
        self.n_out_neurons = np.sum(self.n_out_neurons_per_type)

        # compute vector of types of input, recurrent and output neurons
        self.in_neuron_type_ids = np.concatenate([i * np.ones(n) for i, n in enumerate(self.n_in_neurons_per_type)],
                                                 axis=0).astype(np.int32)

        self.glif_neuron_type_ids = self.n_gilf_neurons_per_type_to_neuron_type_ids(self.n_rec_neurons_per_type)

        self.out_neuron_type_ids = np.concatenate([i * np.ones(n) for i, n in enumerate(self.n_out_neurons_per_type)],
                                                  axis=0).astype(np.int32)
        self.out_neuron_type_ids += self.n_in_types + self.n_rec_types

        self.n_glif_neurons = self.n_rec_neurons + self.n_out_neurons

        self.state_size = (
            self.max_delay * self.n_glif_neurons,  # z
            self.n_glif_neurons,  # v
            self.n_glif_neurons,  # r
            self._n_receptors * self.n_glif_neurons,  # psc rise
            self._n_receptors * self.n_glif_neurons,  # psc
        )
        self.n_neurons = self.n_in_neurons + self.n_rec_neurons + self.n_out_neurons

    def init_neuron_params(self):
        ampere_factor = 1
        voltage_factor = 1
        conductance_factor = ampere_factor / voltage_factor
        time_factor = self._dt
        tau = self._params['C_m'] * ampere_factor / (self._params['g'] * conductance_factor)
        self._decay = np.exp(-self._dt / tau)
        # self.readout_decay = np.exp(-self._dt / self.tau_readout)
        self._current_factor = (1 - self._decay) / (self._params['g'] * conductance_factor)
        self._syn_decay = np.exp(-self._dt / np.array(self._params['tau_syn']))
        self._psc_initial = np.e / np.array(self._params['tau_syn'])

        def _f(_v, full_flat_size=True):
            return tf.Variable(tf.cast(self._gather(_v, full_flat_size), self._compute_dtype), trainable=False)

        # all volatile
        self.v_reset = _f(self._params['V_reset'])  # * voltage_factor
        self.syn_decay = _f(self._syn_decay)
        self.psc_initial = _f(self._psc_initial)  # * ampere_factor
        self.t_ref = _f(self._params['t_ref'])  # * time_factor
        self.param_k = _f(self._params['k'])
        self.v_th = _f(self._params['V_th'])  # * voltage_factor
        self.e_l = _f(self._params['E_L'])  # * voltage_factor
        self.param_g = _f(self._params['g'])  # * conductance_factor
        self.decay = _f(self._decay)
        self.current_factor = _f(self._current_factor)
        self.e_i_signs = _f(self._params['e_i'], full_flat_size=False)

    def init_optimization_parameters(self, eta_mu, eta_sig):
        self.eta_mu = eta_mu
        self.eta_sig_coeff = eta_sig

    def init_genome(self, flags):
        mu_connection_parameter_init = np.random.uniform(size=(self.n_types, self.n_types),
                                                         low=flags.mu_cp_init_min,
                                                         high=flags.mu_cp_init_max)
        if self.flags.spatial_input_to_single_type:
            diag = 8 * np.diag(np.ones(self.n_in_types))
            mu_connection_parameter_init[:self.n_in_types, self.n_in_types:2 * self.n_in_types] += diag

        self.mu_connection_parameter = self.add_weight(
            shape=(self.n_types, self.n_types),
            initializer=tf.keras.initializers.constant(mu_connection_parameter_init),
            dtype=tf.float32,
            name='mu_connection_parameter')
        self.sig_connection_parameter = self.add_weight(
            shape=(self.n_types, self.n_types),
            initializer=tf.keras.initializers.constant(flags.sig_cp_init), dtype=tf.float32,
            name='sig_connection_parameter')

        mu_connection_weights_init = np.array([self.flags.e_in_weight, self.flags.i_in_weight,
                                               self.flags.e_weight, self.flags.i_weight])
        self.mu_connection_weights = self.add_weight(
            shape=(4,),  # w_in, w_e_rec, w_i_rec
            initializer=tf.keras.initializers.constant(mu_connection_weights_init),
            dtype=tf.float32,
            name='mu_connection_weights')
        self.sig_connection_weights = self.add_weight(
            shape=(4,),
            initializer=tf.keras.initializers.constant(flags.sig_w_init), dtype=tf.float32,
            name='sig_connection_weights')

        self.mu_rec_neurons_per_type_parameter = self.add_weight(
            shape=(self.n_rec_types,),
            initializer=tf.keras.initializers.glorot_uniform,
            dtype=tf.float32,
            name='mu_n_rec_neurons_per_type')
        self.sig_rec_neurons_per_type_parameter = self.add_weight(
            shape=(self.n_rec_types,),
            initializer=tf.keras.initializers.constant(flags.sig_rec_n_per_type_para_init_mean),
            dtype=tf.float32,
            name='sig_n_rec_neurons_per_type')

        mu_output_loc_init = 0

        self.mu_trained_output_locations = self.add_weight(
            shape=(self.n_out_types,),
            initializer=tf.keras.initializers.constant(mu_output_loc_init),
            dtype=tf.float32,
            name='mu_trained_output_locations')
        self.sig_trained_output_locations = self.add_weight(
            shape=(self.n_out_types,),
            initializer=tf.keras.initializers.constant(0.1 * self.flags.spatial_height),
            dtype=tf.float32,
            name='sig_trained_output_locations')

        # needs to be a variable..
        self.deletion_mask = self.add_weight(
            shape=(self.n_glif_neurons,),
            initializer=tf.keras.initializers.ones(), dtype=tf.float32,
            name='deletion_mask')

    def init_samples_from_genome(self):
        # define variables for weights (needed for tf.function graph context)
        self.in_weights = self.add_weight(
            shape=(
                self.offspring_size, self.offspring_sample_size, self.n_in_neurons,
                self._n_receptors * self.n_glif_neurons),
            initializer=tf.keras.initializers.zeros(),
            name='in_weights', trainable=False
        )
        time = 2  # used for delay computation
        self.rec_weights = self.add_weight(
            shape=(self.offspring_size, self.offspring_sample_size, time, self.n_glif_neurons,
                   self._n_receptors * self.n_glif_neurons),
            initializer=tf.keras.initializers.zeros(),
            name='rec_weights', trainable=False
        )

        self.freeze()

    def init_masks(self):
        mask = np.diag(np.ones(self.n_glif_neurons, dtype=np.bool))[..., None]
        mask = np.tile(mask, (1, 1, self._n_receptors)).reshape(
            (self.n_glif_neurons, self._n_receptors * self.n_glif_neurons))
        self.disconnect_mask = tf.cast(mask, tf.bool)

        mask = np.ones(shape=(self.n_types, self.n_types), dtype=np.float32)
        mask[:self.n_in_types, -self.n_out_types:] = 0  # no direct connections from input to output
        mask[-self.n_out_types:] = 0  # no connections from output to anywhere

        # also no connections rec -> in (will not get gathered anyway but useful for plotting)
        mask[:, :self.n_in_types] = 0
        self.type_mask = tf.constant(copy(mask))

        # change add constraints to mask to only allow one input variable to be mapped onto one type
        if self.flags.spatial_input_to_single_type:
            assert self.n_rec_types >= self.n_in_types  # impossible to map one input to a type.
            mask = copy(mask)
            diag = np.diag(np.ones(self.n_in_types))
            padding = np.zeros(shape=(self.n_in_types, self.n_rec_types - self.n_in_types), dtype=np.float32)
            diag = np.concatenate((diag, padding), -1)
            mask[:self.n_in_types, self.n_in_types:self.n_in_types + self.n_rec_types] = diag
            self.input_restriction_type_mask = mask

        self.w_out_coeff_mask = tf.concat((tf.ones(shape=(self.n_glif_neurons, self._n_receptors * self.n_rec_neurons)),
                                           self.flags.w_out_coeff * tf.ones(
                                               shape=(self.n_glif_neurons, self._n_receptors * self.n_out_neurons))),
                                          -1)

    def n_gilf_neurons_per_type_to_neuron_type_ids(self, n_neurons_per_type):
        neuron_type_ids = np.zeros(shape=(self.offspring_size, self.n_rec_neurons), dtype=np.int32)
        for i, o_n_neurons_per_type in enumerate(n_neurons_per_type):
            neuron_type_ids[i] = np.concatenate(
                [j * np.ones(n, dtype=np.int32) for j, n in enumerate(o_n_neurons_per_type)])
        neuron_type_ids += self.n_in_types

        # # add output neuron type ids
        out_start = tf.reduce_max(neuron_type_ids) + 1
        out_neuron_type_ids = tf.tile(tf.range(out_start, out_start + self.n_out_types)[None], (self.offspring_size, 1))
        out_neuron_type_ids = tf.repeat(out_neuron_type_ids, self.n_out_neurons // self.n_out_types, -1)
        neuron_type_ids = tf.concat((neuron_type_ids, out_neuron_type_ids), axis=-1)
        return neuron_type_ids

    def update_params(self):
        self.transform_type_params_to_numbers()
        self.glif_neuron_type_ids = self.n_gilf_neurons_per_type_to_neuron_type_ids(self.n_rec_neurons_per_type)

        def _f(_v, full_flat_size=True):
            return tf.cast(self._gather(_v, full_flat_size), self._compute_dtype)

        self.v_reset.assign(_f(self._params['V_reset']))  # * voltage_factor
        self.syn_decay.assign(_f(self._syn_decay))
        self.psc_initial.assign(_f(self._psc_initial))  # * ampere_factor
        self.t_ref.assign(_f(self._params['t_ref']))  # * time_factor
        self.param_k.assign(_f(self._params['k']))
        self.v_th.assign(_f(self._params['V_th']))  # * voltage_factor
        self.e_l.assign(_f(self._params['E_L']))  # * voltage_factor
        self.param_g.assign(_f(self._params['g']))  # * conductance_factor
        self.decay.assign(_f(self._decay))
        self.current_factor.assign(_f(self._current_factor))
        self.e_i_signs.assign(_f(self._params['e_i'], full_flat_size=False))

        self.n_i_neurons = np.sum((self.e_i_signs.numpy() == -1).astype(np.int32), -1)
        self.n_e_neurons = self.n_rec_neurons - self.n_i_neurons
        self.n_i_types = np.sum((self._params['e_i'] == -1).astype(np.int32))
        self.n_e_types = self.n_rec_types - self.n_i_types
        self.n_glif_types = self.n_types - self.n_in_types
        pass

    def build(self, input_shape):
        super().build(input_shape)

    def zero_state(self, dtype=tf.float32):
        z_rec0 = tf.zeros((self.all_flat_size, self.max_delay * self.n_glif_neurons), dtype)
        v_rec0 = tf.ones((self.all_flat_size, self.n_glif_neurons), dtype) * tf.cast(
            self.v_th[0] * self.flags.voltage_init + (1. - self.flags.voltage_init) * self.v_reset[0], dtype)
        r0 = tf.zeros((self.all_flat_size, self.n_glif_neurons), dtype)
        psc_rise0 = tf.zeros((self.all_flat_size, self.n_glif_neurons * self._n_receptors), dtype)
        psc0 = tf.zeros((self.all_flat_size, self.n_glif_neurons * self._n_receptors), dtype)
        return z_rec0, v_rec0, r0, psc_rise0, psc0

    def compute_connection_probabilities(self):
        dm = self.distance_matrix  # * 200/3  # use for of old params
        lamb = self.flags.lamb  # * 80
        distance_factor = tf.exp(-dm ** 2 / lamb ** 2)
        prob = tf.nn.sigmoid(self.connection_parameter)
        prob = tf.quantization.fake_quant_with_min_max_args(prob, min=0, max=1, num_bits=self.flags.n_prob_bits)
        return prob, distance_factor

    def expand_ei_weights(self, weights):
        # expands weights to extra time dim to implement different transmission delay
        # for e and i neurons
        e = tf.nn.relu(weights)
        i = -tf.nn.relu(-weights)
        return tf.stack([i, e], axis=2)


    def sample_params(self):
        # update params
        self.update_params()
        # update spatial structure
        self.update_spatial_structure()
        connection_probabilities, distance_factor = self.compute_connection_probabilities()

        # apply connection constraints mask (no in->out, out->rec and out->out)
        self.connection_parameter *= self.type_mask
        connection_probabilities *= self.type_mask
        if self.flags.spatial_input_to_single_type:
            connection_probabilities *= self.input_restriction_type_mask

        # compute in_weights indices
        # select right section of dist matrix
        in_dist_fact = distance_factor[:, :self.n_in_neurons, self.n_in_neurons:]
        in_weights = self.create_weight_matrix(connection_probabilities, in_dist_fact,
                                               self.in_neuron_type_ids, self.n_in_neurons,
                                               self.glif_neuron_type_ids, self.n_glif_neurons,
                                               n_receptors=self._n_receptors, quantize=True,
                                               position="in") * self.flags.w_in_coeff
        self.in_weights.assign(in_weights)

        # compute rec weights indices
        rec_dist_fact = distance_factor[:, self.n_in_neurons:, self.n_in_neurons:]
        rec_weights = self.create_weight_matrix(connection_probabilities, rec_dist_fact,
                                                self.glif_neuron_type_ids, self.n_glif_neurons,
                                                self.glif_neuron_type_ids, self.n_glif_neurons,
                                                n_receptors=self._n_receptors, use_e_i=True,
                                                quantize=True)
        rec_weights *= self.w_out_coeff_mask
        rec_weights = self.expand_ei_weights(rec_weights)
        self.rec_weights.assign(rec_weights)

    # override, as we have n_receptors here
    def create_weight_matrix(self, connection_probabilities, distance_factor, pre_unit_types,
                             n_pre_neurons, post_unit_types, n_post_neurons, n_receptors=1,
                             use_e_i=False, quantize=False, position="rec"):
        n_synapses = target_n_synapses  # self.flags.n_receptors_coeff
        # n_pre_types or offspring x n_pre_types
        if len(pre_unit_types.shape) == 1:  # pre is input, no offspring dimension
            pre_type_indices = tf.tile(pre_unit_types[None, :, None],
                                       (self.offspring_size, 1, n_post_neurons))
        else:  # pre is recurrent
            pre_type_indices = tf.tile(pre_unit_types[:, :, None],
                                       (1, 1, n_post_neurons))

        if len(post_unit_types.shape) == 1:  # post is output
            post_type_indices = tf.tile(post_unit_types[None, :, None],
                                        (self.offspring_size, 1, n_pre_neurons))
        else:  # post is output
            post_type_indices = tf.tile(post_unit_types[:, :, None],
                                        (1, 1, n_pre_neurons))

        indices = tf.stack((pre_type_indices, tf.transpose(post_type_indices, (0, 2, 1))),
                           axis=-1)  # offspring_size x n_pre_neurons x n_post_neurons x 2

        weights_connection_prob = gather2D(connection_probabilities, indices)

        # apply distance factor
        weights_connection_prob = distance_factor * weights_connection_prob

        # pruning small connection probs
        weights_connection_prob = tf.where(weights_connection_prob > self.flags.prob_pruning_thr,
                                           weights_connection_prob,
                                           tf.zeros_like(weights_connection_prob))

        # extend for offspring_sample_size
        weights_connection_prob = tf.tile(weights_connection_prob[:, None],
                                          (1, self.offspring_sample_size, 1, 1))


        weights_connection_prob = weights_connection_prob[..., None]

        # repeat n_receptors_coeff
        weights_connection_prob = tf.repeat(weights_connection_prob[..., None], n_synapses, axis=-1)

        # sample connections from distributions
        weights_is_connected = tf.random.uniform(weights_connection_prob.shape,
                                                 maxval=1, dtype=tf.float32) < weights_connection_prob


        dense_weights = self.build_weights(position)

        synapse_mask = self.compute_synapse_mask(dense_weights)

        # extend for n_receptors
        dense_weights = dense_weights[..., None]  # tf.repeat(dense_weights, n_receptors, axis=-1)


        n_sampled_synapses = tf.reduce_sum(tf.cast(weights_is_connected, tf.float32), -1)  # sum over synapses

        # save sparsity params
        realized_n_synapses = tf.reduce_sum(tf.cast(n_sampled_synapses > 0, tf.float32), axis=[2, 3, 4])
        average_n_synapses = tf.reduce_mean(realized_n_synapses).numpy()
        self.sparsity_info[f'{position}_n_synapses'] = average_n_synapses
        total_n_connections = self.n_rec_neurons * (self.n_glif_neurons if position == 'rec' else self.n_in_neurons)
        sparsity = tf.reduce_mean(average_n_synapses / total_n_connections).numpy()
        self.sparsity_info[f'{position}_sparsity'] = sparsity
        dm = self.rec_distance_matrix if position == 'rec' else self.in_distance_matrix
        dm = dm[:, None]  # add oss
        wire_lens = tf.where(n_sampled_synapses[..., 0] > 0, dm, tf.zeros_like(dm)).numpy().sum((-2, -1)).mean()
        self.sparsity_info[f'{position}_wire_len'] = wire_lens

        # weight perturbation:
        if self.flags.weight_perturbation > 0:
            weight_perturbation_noise = tf.random.uniform(n_sampled_synapses.shape,
                                                          minval=-self.flags.weight_perturbation,
                                                          maxval=self.flags.weight_perturbation)
            weight_perturbation_noise = tf.round(weight_perturbation_noise * n_sampled_synapses)
            n_sampled_synapses += weight_perturbation_noise
            n_sampled_synapses = tf.nn.relu(n_sampled_synapses)

        weights = dense_weights * n_sampled_synapses

        if self.flags.save_w_intermediate_steps:
            if position == 'in':
                self.realized_weights_in = weights
            else:
                self.realized_weights_rec = weights
        weights *= synapse_mask  # apply synapse mask enforcing e->e, i->e, e->i, i->i

        current_shape = weights.shape.as_list()
        target_shape = current_shape[:-2] + [current_shape[-2] * 4]
        weights = tf.reshape(weights, target_shape)

        # rescale if it used a different number of synapses
        weights *= (self.flags.n_receptors_coeff / target_n_synapses)
        return weights

    def build_weights(self, position):
        if position == 'in':
            n_pre = self.n_in_neurons
            n_post = 1
            e_weight = self.connection_weights[..., 0][:, None, None, None]
            i_weight = self.connection_weights[..., 1][:, None, None, None]
        else:
            n_pre = self.n_glif_neurons
            n_post = 1  # self.n_glif_neurons
            e_weight = self.connection_weights[..., 2][:, None, None, None]
            i_weight = self.connection_weights[..., 3][:, None, None, None]

        weights = tf.abs(e_weight) * tf.ones(shape=(self.offspring_size, self.offspring_sample_size, n_pre, n_post))

        if position == 'rec':
            is_inhibitory = tf.tile(self.e_i_signs[:, None, :, None] < 0, (1, self.offspring_sample_size, 1, n_post))
            weights = tf.where(is_inhibitory,
                               -tf.abs(i_weight) * tf.ones_like(weights),  # negative sign gets added here
                               weights)
        return weights

    def compute_synapse_mask(self, weights):
        _, _, n_pre_neurons, _ = weights.shape.as_list()
        target_e_i = tf.tile(self.e_i_signs[:, None, None], (1, self.offspring_sample_size, n_pre_neurons, 1))
        ones = tf.ones(shape=(self.offspring_size, self.offspring_sample_size, n_pre_neurons, self.n_glif_neurons))
        zeros = tf.zeros_like(ones)

        ee = tf.where(tf.logical_and(weights > 0, target_e_i > 0), ones, zeros)
        ie = tf.where(tf.logical_and(weights < 0, target_e_i > 0), ones, zeros)
        ei = tf.where(tf.logical_and(weights > 0, target_e_i < 0), ones, zeros)
        ii = tf.where(tf.logical_and(weights < 0, target_e_i < 0), ones, zeros)

        mask = tf.stack([ee, ie, ei, ii], -1)
        mask = tf.reshape(mask,
                          (self.offspring_size, self.offspring_sample_size, n_pre_neurons, self.n_glif_neurons, 4))
        return mask

    def compute_external_current_from_input_spikes(self, inp):
        tf_shp = tf.unstack(tf.shape(inp))
        shp = inp.shape.as_list()
        for i, a in enumerate(shp):
            if a is None:
                shp[i] = tf_shp[i]
        input_current = tf.einsum('bosti, osih->bosth', inp,
                                  self.in_weights)  # self.input_layer(inp) * self._input_gradient_scale
        input_current = tf.reshape(
            input_current, (self.all_flat_size, -1, self._n_receptors * self.n_glif_neurons))
        return input_current

    def _gather(self, prop, full_flat_size=True):
        res = tf.gather_nd(prop, self.glif_neuron_type_ids[..., None] - self.n_in_types)
        if full_flat_size:
            new_shape = (self.n_glif_neurons,) if len(prop.shape) == 1 else (self.n_glif_neurons, prop.shape[1])
            tile_shape = (self.batch_size, 1, self.offspring_sample_size) + tuple([1] * len(prop.shape))
            res = tf.tile(res[None, :, None], tile_shape)
            res = tf.reshape(res, (self.all_flat_size,) + new_shape)
        return res

    def make_flat_shape(self, tensor):
        flat_shape = (self.all_flat_size, -1)
        return tf.reshape(tensor, flat_shape)

    def _compute_internal_currents(self, z_rec, rec_external_current, batch_size):
        w_rec = tf.where(self.disconnect_mask, tf.zeros_like(self.rec_weights), self.rec_weights)

        z_rec = tf.reshape(z_rec, (self.batch_size, self.offspring_size, self.offspring_sample_size, 2, -1))

        i_rec = self.make_flat_shape(tf.einsum('bosti,ostij->bosj', z_rec, w_rec))

        rec_inputs = tf.cast(i_rec, self._compute_dtype)
        rec_inputs = tf.reshape(rec_inputs + rec_external_current, (batch_size, self.n_glif_neurons, self._n_receptors))
        return rec_inputs

    def call(self, inputs, state, constants=None):
        batch_size = inputs.shape[0]
        if batch_size is None:
            batch_size = tf.shape(inputs)[0]
        external_current = inputs

        z, v, r, psc_rise, psc = state

        psc_rise = tf.reshape(psc_rise, (batch_size, self.n_glif_neurons, self._n_receptors))
        psc = tf.reshape(psc, (batch_size, self.n_glif_neurons, self._n_receptors))

        z_buf = tf.reshape(z, (self.all_flat_size, self.max_delay, -1))  # self.make_flat_shape(z)
        z = z_buf[:, 0]  # z contains the spike from the prev time step
        z_rec = z_buf[:, 1:]
        inputs = self._compute_internal_currents(z_rec, external_current, batch_size)

        new_psc_rise = self.syn_decay * psc_rise + inputs * self.psc_initial
        new_psc = psc * self.syn_decay + self._dt * self.syn_decay * new_psc_rise

        new_r = tf.nn.relu(r + z * self.t_ref - self._dt)

        input_current = tf.reduce_sum(psc, -1)

        if constants != None:
            input_current = constants[0]

        decayed_v = self.decay * v

        gathered_g = self.param_g * self.e_l

        c1 = input_current + gathered_g
        new_v = decayed_v + self.current_factor * c1

        new_z = spike_function(new_v, self.v_th)
        if self.flags.less_excitable > 0.0:
            excitation_mask = 1 - (1 - self.deletion_mask) * np.random.binomial(1,
                                                                                1 - self.flags.less_excitable,
                                                                                np.shape(self.deletion_mask))
        else:
            excitation_mask = self.deletion_mask
        new_z = excitation_mask * new_z  # apply neuron deletion

        old_new_v = tf.where(z > 0.5, self.v_reset, v)  # v_rec + reset_current_rec
        new_v = tf.where(new_r > 0., old_new_v, new_v)
        new_z = tf.where(new_r > 0., tf.zeros_like(new_z), new_z)

        new_psc = tf.reshape(new_psc, (batch_size, self.n_glif_neurons * self._n_receptors))
        new_psc_rise = tf.reshape(new_psc_rise, (batch_size, self.n_glif_neurons * self._n_receptors))

        # the last neurons are the output neurons
        new_z_out = new_z[..., -self.n_out_neurons:]
        outputs = (new_z_out, dict(v_rec=new_v[..., :-self.n_out_neurons],
                                   v_out=new_v[..., -self.n_out_neurons:],
                                   z_rec=new_z[..., :-self.n_out_neurons],
                                   z_out=new_z_out))

        # add new time step to beginning buffer and drop last
        new_z_buf = tf.concat((new_z[:, None], z_buf[:, :-1]), 1)
        new_z_buf = tf.reshape(new_z_buf, (self.all_flat_size, -1))
        new_state = (new_z_buf, new_v, new_r, new_psc_rise, new_psc)

        return outputs, new_state

    def freeze(self):

        z_shape = (self.offspring_size, self.n_types, self.n_types)

        # compute connection_parameter
        self.connection_parameter = self.mu_connection_parameter + self.sig_connection_parameter * tf.zeros(
            shape=z_shape, dtype=tf.float32)

        #  compute connection_weight
        self.connection_weights = self.mu_connection_weights + self.sig_connection_weights * tf.zeros(
            shape=(self.offspring_size, 4), dtype=tf.float32)  # to ensure right shape

        self.rec_neurons_per_type_parameter = self.mu_rec_neurons_per_type_parameter + \
                                              self.sig_rec_neurons_per_type_parameter * tf.zeros(
            shape=(self.offspring_size, self.n_rec_types),
            dtype=tf.float32)
        self.trained_output_locations = self.mu_trained_output_locations + \
                                        self.sig_trained_output_locations * tf.zeros(
            shape=(self.offspring_size, self.n_out_types),
            dtype=tf.float32)


    def transform_type_params_to_numbers(self):
        fractions = tf.math.softmax(self.rec_neurons_per_type_parameter, -1)
        neuron_numbers_float = fractions * self.n_rec_neurons
        neuron_numbers_int = tf.cast(tf.floor(neuron_numbers_float), tf.int32).numpy()
        differences = neuron_numbers_float - neuron_numbers_int
        differences_sorted_indices = tf.argsort(differences, axis=-1, direction='DESCENDING').numpy()
        overlap = self.n_rec_neurons - tf.cast(np.sum(neuron_numbers_int, -1), tf.int32).numpy()
        for i in range(self.offspring_size):
            for j in range(overlap[i]):
                neuron_numbers_int[i, differences_sorted_indices[i, j]] += 1

        self.n_rec_neurons_per_type = neuron_numbers_int


    def gen_stats_str(self):
        return util.gen_stats_str('Connection param: ', self.mu_connection_parameter,
                                  self.sig_connection_parameter)

