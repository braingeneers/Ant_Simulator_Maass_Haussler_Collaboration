from matplotlib import patches, gridspec
import os

from data_analysis.ploty_utils import *



def plot_obs_out(tensors, path, i=0, task=None):
    gs = gridspec.GridSpec(3, 1, height_ratios=[40, 40,  1])
    fig = plt.figure(figsize=(8, 4), dpi=300)
    in_ax, out_ax, s_ax = [plt.subplot(gs[i]) for i in range(3)]

    t_start = 1000

    plot_in(in_ax, tensors, task, t_start=t_start)
    plot_out(out_ax, tensors, task, t_start=t_start)
    # plt.colorbar(ax=out_ax, mappable=pcm)

    replace_x_axis_with_scale_bar(out_ax, s_ax, 500, font_size=12, size_vertical=0.1, pad=-1.0,
                                  pos='lower center')

    filepath = os.path.join(path, f"obs_out_{i}.png")
    plt.savefig(filepath)
    plt.close(fig)
    pass


def plot_z_y_control(tensors, path, task, i=0, model: GlifAscPscTrainableCellDense = None, n=-1):
    if n == -1:
        n = model.n_glif_neurons

    d = 300
    t_start = tensors['z_rec'].numpy().shape[1] - d

    gs = gridspec.GridSpec(2, 1, height_ratios=[40,  1])
    fig = plt.figure(figsize=(6, 6), dpi=300)
    z_ax, s_ax = [plt.subplot(gs[i]) for i in range(2)]
    plot_z(z_ax, tensors, model, n, eio_padding=10, padding=5, t_start=t_start,
           linewidth=2.0, spike_height=2.0)

    replace_x_axis_with_scale_bar(z_ax, s_ax, 50, font_size=12)

    filepath = os.path.join(path, f"activity_plot_{d}ms{i}.png")
    plt.savefig(filepath)
    plt.close(fig)


def plot_x_v_z_y_control(tensors, path, task, i=0, model: GlifAscPscTrainableCellDense = None, n=-1):
    if n == -1:
        n = model.n_glif_neurons

    gs = gridspec.GridSpec(5, 1, height_ratios=[5, 5, 5, 3, 1])
    fig = plt.figure(figsize=(12, 12))
    x_ax, v_ax, z_ax, z_ax_zoom, y_ax = [plt.subplot(gs[i]) for i in range(5)]
    # plot x
    plot_x_pop(x_ax, tensors['z_in'][0], model)
    # plot v
    plot_v(v_ax, tensors, n)

    # plot spikes
    # plot_spike_rate(z_ax, tensors, model, n)
    plot_z(z_ax, tensors, model, z_ax_zoom, n)

    # plot output
    pcm = plot_out(y_ax, tensors, task)

    if pcm:
        plt.colorbar(ax=x_ax, mappable=pcm).remove()
        plt.colorbar(ax=z_ax, mappable=pcm).remove()
        plt.colorbar(ax=z_ax_zoom, mappable=pcm).remove()

    filepath = os.path.join(path, f"activity_plot{i}.png")
    # plt.tight_layout()
    plt.savefig(filepath)
    plt.close(fig)


def compute_avg_rel_weight_distance(model: GlifAscPscTrainableCellDense, path):

    res = model.calculate_deviations_from_each_other()
    print(res)
    # rec_weights = model.realized_weights_rec[..., 0]

    # rec_weights_prob = model.connection_strengths_rec
    # exp_weight = rec_weights_prob[0, 0] * model.flags.n_receptors_coeff * model.mu_connection_weights[2]

    # rel_deviation = tf.reduce_mean(tf.math.divide_no_nan(tf.abs(rec_weights[0] - exp_weight]), exp_weight))


    pass


def plot_prob_table_control(model: GlifAscPscTrainableCellDense, path):
    cmap = 'bwr'
    model.freeze()
    prob, distance = model.compute_connection_probabilities()
    prob *= model.type_mask
    if model.flags.spatial_input_to_single_type:
        prob *= model.input_restriction_type_mask
    prob = prob[0].numpy()

    n_types = model.n_types  # model.n_rec_types + model.n_out_types

    cm = NeuronTypeColorMap(model, n_input_types=model.n_in_types)
    prob_cm = plt.get_cmap(cmap)
    colors = [cm[i] for i in range(n_types)]
    prob_color = np.array([[prob_cm(prob[i, j]/2+0.5) for j in range(n_types)] for i in range(n_types)])
    prob = np.around(prob, 2).astype(np.str)
    prob = np.where(prob == '0.0', '', prob)

    type_numbers = [str(i+1) for i in range(n_types)]
    type_numbers_rows = [f"{i+1}" for i in range(n_types)]
    col_widths = [1/model.n_types for _ in range(n_types+1)]
    type_bounds = [model.n_e_types]

    gs = gridspec.GridSpec(2, 1, height_ratios=[5, 1])  # , width_ratios=[25, 1])
    fig = plt.figure(figsize=(8, 8), dpi=300)
    prob_ax, frac_ax = [plt.subplot(gs[i, 0]) for i in range(2)]
    prob_ax.spines['right'].set_visible(False)
    prob_ax.spines['top'].set_visible(False)
    prob_ax.spines['left'].set_visible(False)
    prob_ax.spines['bottom'].set_visible(False)
    prob_ax.set_xticks([])
    prob_ax.set_yticks([])
    prob_color = greyify(prob_color)

    prob_color = prob_color[model.n_in_types:-model.n_out_types, model.n_in_types:-model.n_out_types]
    prob = prob[model.n_in_types:-model.n_out_types, model.n_in_types:-model.n_out_types]
    type_numbers = get_type_names(model)[model.n_in_types:-model.n_out_types]  # type_numbers[model.n_in_types:-model.n_out_types]
    col_colors = colors[model.n_in_types:-model.n_out_types]
    row_colors = colors[model.n_in_types:-model.n_out_types]
    prob_ax.axis([-2, len(prob_color)-2, len(prob_color), -1])

    prob_table = prob_ax.table(rowColours=row_colors,
                               rowLabels=type_numbers,
                               rowLoc='center',
                               colColours=col_colors,
                               colLabels=type_numbers,
                               cellColours=prob_color,
                               cellText=prob,
                               colWidths=col_widths,
                               loc='center',
                               cellLoc='center',
                               bbox=[0, 0, 1, 1])

    prob_table.set_fontsize(24)
    cells = prob_table.get_celld()
    for k, v in cells.items():
        v._edgecolor = (0.7, 0.7, 0.7, 1)
        v._linewidth = 1.5
    lw = 3
    for b in [0] + type_bounds:
        vline = patches.Polygon(((b, -1), (b, model.n_rec_types)), clip_on=False, lw=lw, edgecolor='black')
        prob_ax.add_patch(vline)
        hline = patches.Polygon(((-0.88, b), (model.n_rec_types, b)), clip_on=False, lw=lw, edgecolor='black')
        prob_ax.add_patch(hline)
    # prob_ax.set_title('target type', fontsize=8)
    prob_ax.set_xlabel('target type', labelpad=10)
    prob_ax.xaxis.set_label_position('top')
    prob_ax.set_ylabel('source type', labelpad=30)

    # fraction
    nnpt = compute_nnpt(model, add_input_types=True)

    type_bounds = [model.n_in_types, model.n_in_types + model.n_e_types, model.n_in_types + model.n_rec_types]
    plot_percentage_bar(fig, frac_ax, nnpt, colors, white_locs=type_bounds)
    # frac_ax.set_title('Neuron prevalance', loc='left', pad=20)

    filepath = os.path.join(path, f"prob_table.svg")
    plt.savefig(filepath)
    plt.close(fig)
    pass

