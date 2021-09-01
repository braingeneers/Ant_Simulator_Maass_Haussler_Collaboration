import inspect
import os
import sys

from matplotlib import gridspec

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)
from data_analysis.ploty_utils import *

plt.style.use('seaborn-dark-palette')


def plot_v_z(tensors, path, i=0):
    fig, (v_ax, z_ax) = plt.subplots(2, 1, dpi=300)
    # plot membrane potential
    pcm = v_ax.pcolormesh(tensors['v_rec'][0].numpy().T, cmap='seismic')
    plt.colorbar(ax=v_ax, mappable=pcm)

    # plot spikes
    raster_plot(z_ax, tensors['z_rec'][0].numpy())
    cb = plt.colorbar(ax=z_ax, mappable=pcm)
    cb.remove()

    filepath = os.path.join(path, f"activity_plot{i}.png")
    plt.savefig(filepath)
    pass




def plot_v_z_y(tensors, path, task, i=0, model: GlifAscPscTrainableCellDense = None, n=-1):
    if n == -1:
        n = model.n_glif_neurons

    gs = gridspec.GridSpec(4, 1, height_ratios=[5, 5, 3, 1])
    fig = plt.figure(figsize=(12, 12))
    v_ax, z_ax, z_ax_zoom, y_ax = plt.subplot(gs[0]), plt.subplot(gs[1]), plt.subplot(gs[2]), plt.subplot(gs[3])
    # plot v
    plot_v(v_ax, tensors, n)

    # plot spikes
    plot_z(z_ax, tensors, model, z_ax_zoom, n)

    # plot output
    pcm = plot_out(y_ax, tensors, task)

    if pcm:
        plt.colorbar(ax=z_ax, mappable=pcm).remove()
        plt.colorbar(ax=z_ax_zoom, mappable=pcm).remove()

    filepath = os.path.join(path, f"activity_plot{i}.png")
    # plt.tight_layout()
    plt.savefig(filepath)
    plt.close(fig)


def plot_z_y(tensors, path, task, i=0, model: GlifAscPscTrainableCellDense = None, n=-1):
    if n == -1:
        n = model.n_glif_neurons

    gs = gridspec.GridSpec(3, 1, height_ratios=[5, 2, 1])
    fig = plt.figure(figsize=(12, 12))
    z_ax, z_ax_zoom, y_ax = plt.subplot(gs[0]), plt.subplot(gs[1]), plt.subplot(gs[2])

    # plot spikes
    plot_z(z_ax, tensors, model, z_ax_zoom, n)

    pcm = plot_out(y_ax, tensors, task)
    if pcm:
        plt.colorbar(ax=z_ax, mappable=pcm).remove()
        plt.colorbar(ax=z_ax_zoom, mappable=pcm).remove()

    filepath = os.path.join(path, f"activity_plot{i}.png")
    plt.savefig(filepath)
    plt.close(fig)
    pass


def plot_v_z(tensors, path, i=0):
    gs = gridspec.GridSpec(2, 1, height_ratios=[5, 5])
    fig = plt.figure(figsize=(12, 12))
    v_ax, z_ax = plt.subplot(gs[0]), plt.subplot(gs[1])

    # plot membrane potential
    v_max = np.max(tensors['v_rec'][0].numpy())
    v_min = np.min(tensors['v_rec'][0].numpy())
    pcm = v_ax.pcolormesh(tensors['v_rec'][0].numpy().T, cmap='seismic', vmax=v_max, vmin=v_min)
    plt.colorbar(ax=v_ax, mappable=pcm)
    v_ax.set_title('Membrane potential')
    # v_ax.set_xlabel('time (in ms)')
    v_ax.set_ylabel('Neuron ID')
    v_ax.set_yticks([0, tensors['v_rec'][0].numpy().shape[-1]])

    # plot spikes
    raster_plot(z_ax, tensors['z_rec'][0].numpy())
    # z_ax.set_xlabel('time (in ms)')
    z_ax.set_ylabel('Neuron ID')

    cb = plt.colorbar(ax=z_ax, mappable=pcm)
    z_ax.set_title('Spikes')
    cb.remove()

    filepath = os.path.join(path, f"activity_plot{i}.png")
    # plt.tight_layout()
    plt.savefig(filepath)
    plt.close(fig)
    pass



def plot_z_pca(tensors, path, i, decay=0.95, start_time=0):
    fig, (ax1, ax2) = plt.subplots(2, 1, dpi=300, figsize=(4, 8))
    # z_rec
    z = tensors['z_rec'][0, start_time:].numpy()
    z = util.exp_convolve(tf.constant(z), decay=decay).numpy()
    pca = PCA(n_components=2)
    z_pca = pca.fit_transform(z)
    x, y = z_pca.T
    fade_plot(ax1, x, y, color='blue')
    ax1.set_title('PCA spiking activity recurrent neurons')

    # z_out
    z = tensors['z_out'][0, start_time:].numpy()
    z = util.exp_convolve(tf.constant(z), decay=decay).numpy()
    pca = PCA(n_components=2)
    z_pca = pca.fit_transform(z)
    x, y = z_pca.T
    fade_plot(ax2, x, y, color='blue')
    ax2.set_title('PCA spiking activity output neurons')

    filepath = os.path.join(path, f"pca_{i}.png")
    plt.savefig(filepath)
    plt.close(fig)


def plot_video_frame(ax, image, task):
    img = image[80:-20, 50:-50]
    img = ax.imshow(img)
    ax.axis('off')
    return img


def video_pca_spikes_plot(tensors, path, j, model, flags, n=-1, t_start=1000):
    if n == -1:
        n = model.n_glif_neurons
    n_images = 8
    n_rows = 3

    # gs = gridspec.GridSpec(4, 1, height_ratios=[5, 3, 4, 2])
    fig = plt.figure(figsize=(14, 8))

    # plot video:
    tensors['video'] = tensors['video'][t_start // flags.n_ts_per_env_step:]
    select_every = len(tensors['video']) // n_images
    images = tensors['video'][::select_every]
    # images = np.concatenate(np.array(images), axis=-1)
    for i in range(n_images):
        vid_ax = plt.subplot2grid((n_rows, n_images), (0, i), colspan=1)
        plot_video_frame(vid_ax, images[i], flags.task)

    # plot pca
    pca_x, pca_y = compute_z_pca(tensors['z_rec'][0, t_start:])
    for i in range(n_images):
        pca_ax = plt.subplot2grid((n_rows, n_images), (1, i), colspan=1)
        # plot old pca
        end = (i) * select_every * flags.n_ts_per_env_step
        fade_plot(pca_ax, pca_x[:end], pca_y[:end], color='blue', alpha_max=0.15)

        # plot new part
        start = end
        end = (i + 1) * select_every * flags.n_ts_per_env_step
        fade_plot(pca_ax, pca_x[start:end], pca_y[start:end], color='red', alpha_max=0.8)
        pca_ax.axis('off')
        pca_ax.set_xlim([np.min(pca_x), np.max(pca_x)])
        pca_ax.set_ylim([np.min(pca_y), np.max(pca_y)])

    # plot spikes
    z_ax = plt.subplot2grid((n_rows, n_images), (2, 0), colspan=n_images)
    plot_z(z_ax, tensors, model, n=n,  t_start=t_start)

    filepath = os.path.join(path, f"vid_pca_z_plot_{j}.png")
    plt.tight_layout()
    plt.savefig(filepath)
    plt.close(fig)

