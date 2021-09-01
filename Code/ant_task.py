import tensorflow as tf
import os
import datetime as dt
import utils.util as util
import numpy as np
import gym
from gym.vector.tests.utils import make_env
from gym.vector import AsyncVectorEnv

from data_analysis.ploty import video_pca_spikes_plot
from data_analysis.ploty_control import plot_z_y_control
from models import GlifAscPscTrainableCellDense
from utils.util import getDimsOfTask, getValueRangeOfTask, pop_encode, flattern_time_dim
from utils.constants import task_name_env_map
import absl
import json


class ControlTask:
    def __init__(self, model, results_dir, flags):
        # Maybe we need offspring_sample_size
        self.envs = []
        # init flat array containing all envs
        # if flags.save_videos we wrap the first env with a Monitor to save the videos
        self.n_envs = flags.batch_size * flags.offspring_size * flags.offspring_sample_size
        self.fat_shape = (flags.batch_size, flags.offspring_size, flags.offspring_sample_size)

        name = task_name_env_map[flags.task]

        if flags.save_videos:
            monitor_env = gym.make(name)  # .env
            vid_callable = (lambda x: True) if flags.eval_only else (
                lambda x: (x + 1) % flags.plot_every == 0)
            self.monitor_env = gym.wrappers.Monitor(monitor_env, results_dir, video_callable=vid_callable, force=True)
            if flags.eval_only:
                self.monitor_env.render()
            env_fns = [make_env(name, i) for i in range(self.n_envs - 1)]

        else:
            env_fns = [make_env(name, i) for i in range(self.n_envs)]
        self.envs = AsyncVectorEnv(env_fns)

        self.train_fitness_results = []
        self.environment_runs = 0
        self.acc_rewards = np.zeros(self.n_envs)
        self.max_steps = flags.max_steps
        self.model = model
        self.flags = flags
        self.task_name = task_name_env_map[flags.task]
        self.obs_len, self.action_len = getDimsOfTask(self.task_name)
        self.keys = ['v_rec', 'z_rec', 'z_out', 'v_out', 'y_pred']
        self.flattern_time_keys = ['v_rec', 'z_rec', 'z_out', 'v_out', 'y_pred', 'z_in', 'observation']
        self.min_values, self.max_values = getValueRangeOfTask()
        self.n_train_step = 0
        self.compute_reg = self.flags.model == 'glif3'

        self.obs_delay_buf = util.ObservationDelayBuffer(flags.n_obs_delay)
        self.run_var = util.RunningSpikeVar()

    def reset_envs(self):
        rewards = np.zeros(self.n_envs)
        dones = np.zeros(self.n_envs, dtype=bool)
        self.acc_rewards = np.zeros(self.n_envs)
        if self.flags.save_videos:
            monitor_obs = self.monitor_env.reset()
            obs = self.envs.reset()
            obs = np.concatenate((monitor_obs[None], obs), axis=0)
        else:
            obs = self.envs.reset()
        obs = util.getSubObs(obs, self.task_name)
        return obs, rewards, dones

    def step(self, actions):
        if self.flags.save_videos:
            observations, rewards, new_dones, _ = self.envs.step(actions[1:])
            try:
                monitor_obs, monitor_reward, monitor_new_done, _ = self.monitor_env.step(actions[0])
                self.monitor_env.render()
            except gym.error.ResetNeeded:
                monitor_obs, monitor_reward, monitor_new_done = observations[0], 0., True

            observations = np.concatenate((monitor_obs[None], observations), axis=0)
            rewards = np.concatenate((np.array(monitor_reward)[None], rewards), axis=0)
            new_dones = np.concatenate((np.array(monitor_new_done)[None], new_dones), axis=0)
        else:
            observations, rewards, new_dones, _ = self.envs.step(actions)
        observations = util.getSubObs(observations, self.task_name)  # extract important part from observations
        return observations, rewards, new_dones

    def compute_reg_loss(self):
        reg_loss = tf.abs(tf.reduce_mean(self.model.in_weights)) + \
                   tf.abs(tf.reduce_mean(self.model.rec_weights))
        reg_loss *= self.flags.reg_coeff
        return reg_loss

    @tf.function
    def select_actions(self, model, observations, state, action_len):
        inputs = tf.reshape(observations, (model.batch_size, model.offspring_size,
                                           model.offspring_sample_size, self.flags.n_ts_per_env_step,
                                           model.n_in_neurons))

        inputs = model.compute_external_current_from_input_spikes(inputs)
        rnn = tf.keras.layers.RNN(model, return_sequences=True, return_state=True)

        rnn_returns = rnn(inputs, state)
        output = rnn_returns[0]
        new_state = tuple(rnn_returns[1:])
        z_buf_out, tensors = output


        logits = util.rate_decode(z_buf_out, n_val=action_len)
        y_pred = tf.reshape(logits, (model.all_flat_size, action_len))
        if action_len == 1:
            y_pred = tf.cast(y_pred > 0, tf.int32)
        tensors['z_out'] = z_buf_out
        tensors['y_pred'] = tf.repeat(y_pred[:, None, None], self.flags.n_ts_per_env_step, axis=1)
        return y_pred, new_state, tensors

    def simulate_one_episode(self, dump_data=False):
        # apply the weights
        if dump_data:
            tensor_arrays = {k: [] for k in self.keys}
            tensor_arrays['z_in'] = []
            tensor_arrays['observation'] = []
            tensor_arrays['video'] = []

        observations, rewards, dones = self.reset_envs()
        # observation delay
        self.obs_delay_buf.put(observations)
        observations = self.obs_delay_buf.get()

        self.environment_runs += 1
        model_state = self.model.zero_state()
        z_sum = tf.zeros(shape=(self.n_envs, self.model.n_rec_neurons))
        v_reg_sum = tf.zeros(shape=(self.n_envs, self.model.n_rec_neurons))
        actions_hist = []

        for t in range(self.max_steps):
            obs = pop_encode(observations, t=self.flags.n_ts_per_env_step, n=self.flags.n_pop_in_size,  # was n=self.flags.n_pop_in
                             min_values=self.min_values, max_values=self.max_values, k=1)  # k=self.flags.n_pop_in_size
            actions, model_state, tensors = self.select_actions(self.model, obs, model_state, self.action_len)

            # regularization
            # rate
            z_sum = z_sum + tf.reduce_sum(tensors['z_rec'], axis=1)  # sum over time
            # variance
            z_glif = tf.concat((tensors['z_rec'], tensors['z_out']), -1)
            self.run_var.put(z_glif)

            # v_reg
            if self.compute_reg:
                dist_v_el = tf.abs(tf.abs(tensors['v_rec']) - tf.abs(self.model.e_l[:, None, :-self.model.n_out_neurons]))
                tolerance = tf.abs(1.0 * (tf.abs(self.model.v_th[:, None]) - tf.abs(self.model.e_l[:, None])))[...,
                            :-self.model.n_out_neurons]
                v_reg_sum = v_reg_sum + tf.reduce_mean(tf.square(tf.nn.relu(dist_v_el - tolerance)), axis=1)

            if dump_data:
                [tensor_arrays[k].append(tensors[k][0][None]) for k in self.keys]
                tensor_arrays['z_in'].append(obs[0][None])
                tensor_arrays['observation'].append(observations[0][None])
                if self.flags.save_videos:
                    tensor_arrays['video'].append(self.monitor_env.render(mode='rgb_array'))

            actions = tf.squeeze(actions).numpy()
            actions_hist.append(actions)

            observations, rewards, new_dones = self.step(actions)
            self.obs_delay_buf.put(observations)
            observations = self.obs_delay_buf.get()
            self.acc_rewards = np.where(dones == False,
                                        self.acc_rewards + rewards,
                                        self.acc_rewards)

            dones = np.logical_or(dones, new_dones)

            if all(dones) or t + 1 == self.max_steps or (dump_data and dones[0]):
                if self.flags.save_videos:
                    self.monitor_env.stats_recorder.save_complete()
                    self.monitor_env.stats_recorder.done = True
                break

        # compute rate loss
        if self.compute_reg:
            z_sum = tf.reduce_mean(tf.reshape(z_sum, self.fat_shape + (-1,)), axis=[0, 2])
            rate_loss = self.flags.rate_reg_coeff * tf.reduce_mean(
                tf.square(z_sum / ((t + 1) * self.flags.n_ts_per_env_step) - self.flags.target_rate), axis=1)
            # compute voltage loss
            v_reg_loss = self.flags.voltage_reg_coeff * tf.reduce_mean(
                tf.reshape(v_reg_sum, self.fat_shape + (-1,)), axis=[0, 2, 3]) / ((t + 1) * self.flags.n_ts_per_env_step)

            # compute action variance loss
            variance = tf.reshape(self.run_var.get_var(), self.fat_shape + (self.model.n_glif_neurons,))
            self.run_var.reset()
            action_var_loss = self.flags.action_var_loss_coeff * tf.nn.relu(0.5 - variance)
            action_var_loss = tf.reduce_mean(action_var_loss,
                                             axis=[0, 2, 3])  # batch, offspring_sample_size, action_len

        self.obs_delay_buf.clear()
        if dump_data:
            tensors = {k: flattern_time_dim(tf.stack(tensor_arrays[k], axis=1)) for k in self.flattern_time_keys}
            tensors['video'] = tensor_arrays['video']
            return self.acc_rewards, rate_loss, v_reg_loss, action_var_loss, tensors
        return self.acc_rewards, rate_loss, v_reg_loss, action_var_loss

    def eval_step(self):
        self.model.freeze()
        self.model.sample_params()
        fitness, rate_loss, v_reg_loss, action_var_loss, tensors = self.simulate_one_episode(dump_data=True)
        reg_loss = self.compute_reg_loss()
        self.test_fitness_results = fitness

        fitness = tf.reduce_mean(fitness)
        reg_loss = tf.reduce_mean(reg_loss)
        rate_loss = tf.reduce_mean(rate_loss)
        v_reg_loss = tf.reduce_mean(v_reg_loss)
        action_var_loss = tf.reduce_mean(action_var_loss)
        return fitness, reg_loss, rate_loss, v_reg_loss, action_var_loss, tensors


def init_input_nr(f):
    n_inputs, n_outputs = getDimsOfTask(task_name_env_map[f.task])
    n_inputs = n_inputs * f.n_pop_in
    n_outputs = n_outputs * f.n_pop_out
    n_in_neurons_per_type = [f.n_pop_in_size] * n_inputs
    n_out_neurons_per_type = [f.n_pop_out_size] * n_outputs
    return n_inputs, n_in_neurons_per_type, n_out_neurons_per_type


def loop(f, model, sm, tee_print, saver):
    # Prepare metrics
    train_fitness_hist = []
    test_fitness_hist = []
    train_fitness_reg_hist = []

    task = ControlTask(model=model, results_dir=sm.paths.results_path, flags=f)

    if f.restore_from != "" and not f.eval_only:
        start = util.print_old_logs(tee_print, f)
    else:
        start = 0

    for epoch in range(start, f.n_epochs):
        if not f.eval_only:
            train_fitness, train_fitness_reg = task.train_step()
            train_fitness_hist.append(train_fitness)
            train_fitness_reg_hist.append(train_fitness_reg)

        if (epoch + 1) % f.eval_every == 0 or f.eval_only:
            eval_fitness, reg_loss, rate_loss, v_reg_loss, action_var_loss, tensors = task.eval_step()
            test_fitness_hist.append(eval_fitness)

            date_str = dt.datetime.now().strftime('%H:%M:%S %d-%m-%y')

            # print logs
            tee_print(f'Run: {sm.sim_name} Gen: {epoch + 1:>5} @ {date_str} ')

            if f.eval_only:
                tee_print(f'  - Test Fitness Average: {eval_fitness:.3f}')
            else:
                tee_print(
                    f'  - Train Fitness Average: {np.mean(train_fitness_hist[-f.eval_every:]):.3f} '
                    f'(with reg: {np.mean(train_fitness_reg_hist[-f.eval_every:]):.3f}) \n'
                    f'  - Test Fitness: {eval_fitness:.3f} ')
                tee_print(model.gen_stats_str())
                if f.model == 'glif3':
                    tee_print(f'(weight reg: -{reg_loss:.3f}, rate reg: -{rate_loss:.3f}, '
                              f'voltage reg: -{v_reg_loss:.3f}, action variance loss: -{action_var_loss:.3f})')
                    tee_print(f'  - Weights: e_in_weight: {model.mu_connection_weights[0]:.3f} '
                              f'e_weight: {model.mu_connection_weights[1]:.3f} '
                              f'i_weight: {model.mu_connection_weights[2]:.3f} \n')

            if (epoch + 1) % f.plot_every == 0 or f.eval_only:
                # Plotting
                plot_path = sm.paths.results_path
                plot_z_y_control(tensors, plot_path, f.task, epoch + 1, model)
                video_pca_spikes_plot(tensors, plot_path, epoch + 1, model, f)

        if (epoch + 1) % f.save_every == 0:
            tee_print(f"Saving at iteration {epoch + 1}\n")
            saver.save()



def main(_argv):
    f = absl.flags.FLAGS

    print(f'GPU available: {tf.config.list_physical_devices("GPU")}')
    tf.keras.backend.set_floatx('float32')

    sm = util.create_new_simmanager(f)

    with sm:
        if f.restore_from != "":
            new_f = util.load_flags(f)
            f = new_f


        n_inputs, n_in_neurons_per_type, n_out_neurons_per_type = init_input_nr(f)

        model = GlifAscPscTrainableCellDense(util.load_neuron_params(), n_in_neurons_per_type=n_in_neurons_per_type,
                                             n_out_neurons_per_type=n_out_neurons_per_type,
                                             observation_length=n_inputs,
                                             flags=f)

        # restore probabilistic skeleton from checkpoint
        ckpt = tf.train.Checkpoint(model=model.trainable_variables)
        ckpt_path = sm.paths.results_path
        saver = tf.train.CheckpointManager(ckpt, ckpt_path, max_to_keep=1)
        if f.restore_from != '':
            load_checkpoint_dir = os.path.join(f.restore_from, 'results')
            try:
                ckpt.restore(tf.train.latest_checkpoint(load_checkpoint_dir))
                print(f'Restored from {f.restore_from}')
            except:
                pass

        print(f'Starting experiment {sm.sim_name}')
        data_path = sm.paths.data_path
        with open(os.path.join(data_path, 'config.json'), 'w') as data_file:
            json.dump(f.flag_values_dict(), data_file, indent=4)

        tee_file_path = os.path.join(sm.paths.log_path, 'print.log')
        with open(tee_file_path, 'w') as tee_file:
            def tee_print(_str):
                print(_str)
                tee_file.write(_str)
                tee_file.write('\n')
                tee_file.flush()

            tee_print(f"Comment: {f.comment}")

            model.freeze()
            model.sample_params()
            loop(f, model, sm, tee_print, saver)


if __name__ == '__main__':

    # General flags
    absl.flags.DEFINE_integer('n_epochs', 1000, 'number of epochs')
    absl.flags.DEFINE_integer('eval_every', 1, '')
    absl.flags.DEFINE_integer('plot_every', 1, '')

    absl.flags.DEFINE_integer('batch_size', 1, '')
    absl.flags.DEFINE_integer('offspring_size', 2, '')
    absl.flags.DEFINE_integer('offspring_sample_size', 1, '')

    absl.flags.DEFINE_integer('save_every', 50, '')
    absl.flags.DEFINE_integer('max_steps', 250, '')
    absl.flags.DEFINE_boolean('eval_only', True, '')
    absl.flags.DEFINE_string('restore_from', 'ant',
                             'Path to exp which should be restored')
    absl.flags.DEFINE_string('comment', '', 'comments')
    absl.flags.DEFINE_enum('task', 'ant', ['ant'], '')

    absl.app.run(main)