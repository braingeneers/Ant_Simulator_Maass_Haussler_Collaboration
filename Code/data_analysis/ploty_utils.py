import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from sklearn.decomposition import PCA

import utils.util as util
import tensorflow as tf
import numpy as np
import colorsys

from models import GlifAscPscTrainableCellDense
from utils.constants import task_to_tasktype


plt.style.use('seaborn-dark-palette')



class NeuronTypeColorMap:
    def __init__(self, model: GlifAscPscTrainableCellDense, without_alpha=False, n_input_types=0, min_val=0.0):
        self.n_input_types = n_input_types
        self.n_types = model.n_types
        e_i = model._params['e_i']
        self.e_i = e_i


        self.in_map = plt.get_cmap('cool')  # last third
        self.e_map = plt.get_cmap('winter')  # inverted
        self.i_map = plt.get_cmap('autumn')
        self.o_map = plt.get_cmap('copper')  # second half

        self.n_e_types = model.n_e_types
        self.n_i_types = model.n_i_types
        self.n_o_types = model.n_out_types
        self.min_val = min_val
        self.without_alpha = without_alpha

    def __getitem__(self, item):
        mv = self.min_val
        # input type
        if item < self.n_input_types:
            mv = 2 / 3
            val = mv + (1 - mv) * item / self.n_input_types
            if self.without_alpha:
                return self.in_map(val)[:3]
            return self.in_map(val)
        else:
            item -= self.n_input_types

        # rec types
        if item < self.n_e_types:
            cm = self.e_map
            val = mv + (1 - mv) * (item / self.n_e_types)
            val = 1 - val  # invert
        elif item - self.n_e_types < self.n_i_types:
            cm = self.i_map
            val = mv + (1 - mv) * (item - self.n_e_types) / self.n_i_types
        else:
            mv = 0.5
            cm = self.o_map
            val = mv + (1 - mv) * (item - self.n_e_types - self.n_i_types) / self.n_o_types
        if self.without_alpha:
            return cm(val)[:3]
        return cm(val)


def raster_plot(ax, spikes, linewidth=0.8, max_num_spikes=100000, spike_height=1, **kwargs):
    n_t, n_n = spikes.shape
    event_times, event_ids = np.where(spikes)
    event_times = event_times
    event_ids = event_ids
    v_line_list = []
    for n, t in zip(event_ids, event_times):
        line = ax.vlines(t, n + 0., n + spike_height, linewidth=linewidth, **kwargs)
        v_line_list.append(line)
        # if t == max_num_spikes:
        #     break
    ax.set_ylim([0.5, n_n + .5])
    ax.set_xlim([0, n_t])
    # ax.set_yticks([0, n_n])
    ax.set_yticks([])
    return v_line_list


def fade_plot(ax, x, y, alpha_max=0.2, fade_type='lin', *args, **kwargs):
    n_lines = len(x)
    for i in range(n_lines - 2):
        if fade_type == 'lin':
            alpha = alpha_max * i / n_lines
        else:
            alpha = np.clip(alpha_max * 0.99 ** (n_lines - i) + 0.05, a_min=0, a_max=1)
        alpha = np.clip(alpha + 0.2, a_min=0, a_max=1)
        plot(ax, x[i:i + 2], y[i:i + 2], *args, alpha=alpha, **kwargs)
    pass


def plot(ax, *args, **kwargs):
    # kwargs['c'] = kwargs['c'] if 'c' in kwargs else matplotlib.cm.get_cmap('seaborn-dark-palette')
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')
    # ax.spines['bottom'].set_linewidth(2)
    # ax.spines['left'].set_linewidth(2)
    ax.plot(*args, **kwargs)


def scatter(ax, *args, **kwargs):
    # kwargs['c'] = kwargs['c'] if 'c' in kwargs else matplotlib.cm.get_cmap('seaborn-dark-palette')
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')
    # ax.spines['bottom'].set_linewidth(2)
    # ax.spines['left'].set_linewidth(2)
    ax.scatter(*args, **kwargs)


def bar(ax, *args, **kwargs):
    # kwargs['c'] = kwargs['c'] if 'c' in kwargs else matplotlib.cm.get_cmap('seaborn-dark-palette')
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')
    ax.bar(*args, **kwargs)


def get_color(red_to_green):
    assert 0 <= red_to_green <= 1
    # in HSV, red is 0 deg and green is 120 deg (out of 360);
    # divide red_to_green with 3 to map [0, 1] to [0, 1./3.]
    hue = red_to_green / 3.0
    r, g, b = colorsys.hsv_to_rgb(hue, 1, 1)
    return map(lambda x: int(255 * x), (r, g, b))


def plot_v(v_ax, tensors, n=-1, cax=None):
    # plot membrane potential
    v = tf.concat((tensors['v_rec'][0], tensors['v_out'][0]), -1).numpy().T[::-1]  # combine rec and out v
    v_max = np.max(v)
    v_min = np.min(v)
    pcm = v_ax.pcolormesh(v, cmap='seismic', vmax=v_max, vmin=v_min)
    if cax:
        cb = plt.colorbar(cax=cax, mappable=pcm)
    else:
        cb = plt.colorbar(ax=v_ax, mappable=pcm)
    # cb.ax.set_ylabel('membrane potential in mv')
    v_ax.set_title('Membrane potential')
    # v_ax.set_xlabel('time (in ms)')
    v_ax.set_ylabel('Neuron ID')
    v_ax.set_yticks([0, v.shape[0]])


def plot_x_pop(ax, z, model):
    raster_plot(ax, z)
    obs_size = model.observation_length
    n_input_neurons_per_obs = model.n_in_neurons // obs_size
    n_ts = len(z)

    ys = [i for i in range(n_input_neurons_per_obs, model.n_in_neurons, n_input_neurons_per_obs)]
    ax.hlines(y=ys, xmin=0, xmax=n_ts, colors='grey', alpha=0.6, linestyles=':')

    ax.set_title('Input spikes')
    ax.set_ylabel('Input neuron ID')


def plot_z(z_ax, tensors, model, n=-1, t_start=0, add_white_rows=True, add_input_spikes=True,
           eio_padding=5, padding=1, offset_to_left=0, index=0, frac=1.0, linewidth=1.0, spike_height=1.0):
    z_ax.spines['left'].set_visible(False)

    z = tf.concat((tensors['z_rec'][index], tensors['z_out'][index]), -1)  # combine rec and out spikes
    if add_input_spikes:
        z = tf.concat((tensors['z_in'][index], z), -1)  # combine rec and out spikes

    nnpt = np.concatenate((model.n_rec_neurons_per_type[0], model.n_out_neurons_per_type), axis=0)
    if add_input_spikes:
        nnpt = np.concatenate((model.n_in_neurons_per_type, nnpt), axis=0)

    # apply subsection
    z_sections = []
    new_nnpt = []
    start = 0
    i = 0
    n_e_neurons = 0
    n_i_neurons = 0
    n_out_neurons = 0
    n_in_neurons = 0
    for n in nnpt:
        n_frac = int(n * frac)
        new_nnpt.append(n_frac)
        z_sections.append(z[..., start:start+n_frac])

        # correct for input
        if i < model.n_in_types:
            n_in_neurons = n_frac
        else:
            j = i - model.n_in_types
            if j < model.n_e_types:
                n_e_neurons += n_frac
            elif j < model.n_e_types + model.n_i_types:
                n_i_neurons += n_frac
            else:
                n_out_neurons += n_frac
        start += n
        i += 1
    nnpt = np.array(new_nnpt)
    z = np.concatenate(z_sections, axis=-1)
    n_glif_neurons = n_e_neurons + n_i_neurons + n_out_neurons
    z = z.T[:, t_start:][::-1]  # invert such that the plot will show E/I/Output from top to bottom
    t_max = z.shape[-1]

    if add_white_rows:
        # add spacing between types
        nnpt = nnpt[::-1]
        idx = np.cumsum(nnpt)
        idx = np.repeat(idx, repeats=padding)
        e_idx = n_glif_neurons - n_e_neurons
        i_idx = n_out_neurons

        eio_n_idx = np.concatenate(([e_idx] * eio_padding, [i_idx] * eio_padding), axis=0)
        if add_input_spikes:
            in_index = n_glif_neurons  # model.n_neurons - model.n_in_neurons
            eio_n_idx = np.concatenate(([in_index] * eio_padding, eio_n_idx), axis=0)

        idx = np.concatenate((eio_n_idx, idx), axis=0)
        z = np.insert(z, idx, 0, axis=0)


    raster_plot(z_ax, z.T, color='black', linewidth=linewidth, spike_height=spike_height)
    plot_type_shading(z_ax, model, start=-offset_to_left, end=t_max, alpha=0.4, add_white_rows=add_white_rows,
                      add_input_spikes=add_input_spikes, eio_padding=eio_padding, padding=padding,
                      nnpt=nnpt)


def plot_type_shading(ax, model: GlifAscPscTrainableCellDense, start=-10, end=0, alpha=None,
                      add_white_rows=False, add_input_spikes=False,
                      eio_padding=5, padding=1, nnpt=None):
    n_input_types = model.n_in_types if add_input_spikes else 0
    cmap = NeuronTypeColorMap(model, n_input_types=n_input_types)
    rect_width = end - start
    y = 0
    eio_padding_pos = [model.n_out_types, model.n_out_types + model.n_i_types]
    if add_input_spikes:
        eio_padding_pos = eio_padding_pos + [model.n_glif_types]

    for i, rect_height in enumerate(nnpt):
        ri = len(nnpt) - i - 1
        xy = (start, y)
        rect = Rectangle(xy, rect_width, rect_height, color=cmap[ri], clip_on=False, alpha=alpha)
        ax.add_patch(rect)
        y += rect_height
        if add_white_rows:
            y += padding
        if i + 1 in eio_padding_pos:
            y += eio_padding




def compute_z_pca(z, decay=0.95, n_components=2):
    z = util.exp_convolve(z, decay=decay).numpy()
    pca = PCA(n_components=n_components)
    z_pca = pca.fit_transform(z)
    if n_components == 1:
        return z_pca
    x, y = z_pca.T
    return x, y


def compute_z_dpca(z, labels, decay=0.95, n_components=2):
    from dPCA.dPCA import dPCA
    bs, n_ts, n_neurons = z.shape
    n_labels = np.max(labels) + 1
    trial_averaged_responses = np.zeros((n_neurons, n_ts, n_labels))

    label_counts = np.zeros((n_labels,))
    for l in range(n_labels):
        combination_count = 0
        for i in range(bs):
            if (labels[i] == l).all:
                combination_count += 1
                trial_averaged_responses[..., l] += z[i].reshape(n_neurons, -1)
        label_counts[l] = combination_count
        trial_averaged_responses[..., l] /= combination_count

    min_full_data = np.min(label_counts).astype(np.int32)
    trial_data = np.zeros((min_full_data, n_neurons, n_ts, n_labels))

    for l in range(n_labels):
        combination_count = 0
        for i in range(bs):
            if (l == labels[i]).all:
                trial_data[combination_count, :, :, l] = z[i].T
                combination_count += 1
            if combination_count == min_full_data:
                break

    with util.QuietCall():
        pca = dPCA(labels='lt', n_components=n_components, regularizer='auto', )
        pca.protect = ['t']

        trial_averaged_responses = trial_data.mean(0)
        trial_averaged_responses -= trial_averaged_responses.reshape((n_neurons, -1)).mean(-1)[:, None, None]
        trial_averaged_responses = trial_averaged_responses.transpose((0, 2, 1))
        trial_data = trial_data.transpose((0, 1, 3, 2))

        pca.fit(trial_averaged_responses, trial_data)

        pca_res = pca.transform(trial_averaged_responses)
    l_pc = pca.D['l']
    res = np.einsum('btn,nk->btk', z, l_pc)
    return res


def plot_out(y_ax, tensors, task, t_start=0):
    y_ax.axis('off')
    if task_to_tasktype[task] == 'control':
        pcm = y_ax.pcolormesh(tensors['y_pred'][0].numpy().T[:, t_start:], cmap='seismic')
        cbar = plt.colorbar(ax=y_ax, mappable=pcm, ticks=[-1, 1])
        return pcm

def plot_in(y_ax, tensors, task, t_start=0):
    y_ax.axis('off')
    if task_to_tasktype[task] == 'control':
        obs = np.repeat(tensors['observation'][0].numpy().T, axis=1, repeats=20)
        pcm = y_ax.pcolormesh(obs[:, t_start:], cmap='seismic')
        plt.colorbar(ax=y_ax, mappable=pcm, ticks=[-1, 1])
        return pcm

def compute_nnpt(model, add_input_types=False):
    nnpt = np.concatenate((model.n_rec_neurons_per_type[0], model.n_out_neurons_per_type), axis=0)
    if add_input_types:
        nnpt = np.concatenate((model.n_in_neurons_per_type, nnpt), axis=0)
    return nnpt




def compute_type_bounds(model):
    return [model.n_in_types, model.n_in_types + model.n_e_types, model.n_in_types + model.n_rec_types]


def replace_x_axis_with_scale_bar(z_ax, s_ax, ms=50, font_size=10, size_vertical=1, pad=0.1, pos='lower left'):
    from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar
    import matplotlib.font_manager as fm
    fontprops = fm.FontProperties(size=font_size)

    z_ax.spines['right'].set_visible(False)
    z_ax.spines['top'].set_visible(False)
    z_ax.spines['bottom'].set_visible(False)
    z_ax.set_xticks([])

    scalebar = AnchoredSizeBar(z_ax.transData,
                               ms, f'{ms} ms', pos,
                               pad=pad,
                               color='black',
                               frameon=False,
                               size_vertical=size_vertical,
                               fontproperties=fontprops)
    s_ax.axis('off')
    s_ax.add_artist(scalebar)


def greyify(x):
    # make color array more grey
    grey = [.5, .5, .5]
    if x.shape[-1] == 4:
        grey = grey + [1]
    grey = np.array(grey)

    res = np.where((np.sum(x, -1) < 3.9)[..., None], (x + grey) / 2, x)

    return res


def get_type_names(model):
    n_in_types = model.n_in_types
    n_e_types = model.n_e_types
    n_i_types = model.n_i_types
    n_out_types = model.n_out_types
    n_types = [n_in_types, n_e_types, n_i_types, n_out_types]
    names = ["In ", "E ", "I ", "Out "]
    types_names = []
    for name, n in zip(names, n_types):
        for i in range(n):
            types_names.append(name + str(i+1))
    return types_names
