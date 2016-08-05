import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import rgb_to_hsv
from utils import AttributeDict
from matplotlib.colors import hsv_to_rgb, rgb2hex

GROUP_COLORS = np.array([[0.122, 0.031, 0.408], [0.592, 0.016, 0], [0, 0.467, 0.031], [0.592, 0.49, 0]])


def colorize_groups(figure, group=None, saturations_map=None):
    """Colorize groups representations."""
    nr_colors = figure.shape[0]

    assert nr_colors in [4], 'color map assumes four groups.'
    color_conv = 1 - GROUP_COLORS

    if group is not None:
        new_figure = np.dot(color_conv[[group]].T, figure.reshape(nr_colors, -1)[[group]]).reshape((3,) + figure.shape[1:])
    else:
        new_figure = np.dot(color_conv.T, figure.reshape(nr_colors, -1)).reshape((3, ) + figure.shape[1:])

    if saturations_map is not None:
        new_figure_hsv = new_figure.transpose(1, 2, 0)
        new_figure_hsv = rgb_to_hsv(new_figure_hsv.reshape(-1, 3))

        new_figure_hsv[:, 1] = new_figure_hsv[:, 1] * (saturations_map[[group]]).reshape(new_figure_hsv.shape[0])

        new_figure = hsv_to_rgb(new_figure_hsv)
        new_figure = new_figure.transpose(1, 0).reshape((3, ) + figure.shape[1:])

    new_figure = np.clip(new_figure, 0, 1)

    return new_figure


def show_mat(m, ax, title='', lim=(0, 1), cmap='Greys', **kwargs):
    """Show a single matrix without pesky xticks and yticks."""
    if len(m.shape) == 3 and m.shape[0] == 3:
        m = np.transpose(m, (1, 2, 0))

    if len(m.shape) == 3 and m.shape[-1] == 3:
        cmd = ax.imshow
    else:
        cmd = ax.matshow

    cmd(m, cmap=cmap, vmin=lim[0], vmax=lim[1], **kwargs)
    ax.set_title(title)
    ax.set_yticks([])
    ax.set_xticks([])


def show_reconstr(z, m, ax, channel=None, title=None, saturations_map=None):
    if m is None:
        rec = z
    else:
        rec = z * m

    rec = np.clip(rec, 0, 1)

    rec_plot = colorize_groups(rec, group=channel, saturations_map=saturations_map)
    rec_plot = rec_plot.transpose(1, 2, 0)

    ax.imshow(1 - rec_plot, vmin=0, vmax=1, interpolation='nearest')

    ax.set_xticks([])
    ax.set_yticks([])
    if title:
        ax.set_title(title)


def create_fig_layout(groups, time, n_left, n_extra, extra_col=False):
    """Create a complex grid for visualizing the image"""

    horizontal_grid_spec = 2 * groups + 1
    vertical_grid_spec = time + 2

    figw = 2.6 * vertical_grid_spec
    figh = 2.25 * horizontal_grid_spec

    f = plt.figure(figsize=(figw, figh))

    gs0 = gridspec.GridSpec(2, 2, width_ratios=[2, time + 0.5], height_ratios=[2 * groups, 3.45])
    if extra_col is True:
        gs1 = gridspec.GridSpecFromSubplotSpec(2 * groups + 1, time + 1, gs0[:, 1])
    else:
        gs1 = gridspec.GridSpecFromSubplotSpec(2 * groups + 1, time, gs0[:, 1])
    gs3 = gridspec.GridSpecFromSubplotSpec(n_left, 2, gs0[0, 0])
    gs4 = gridspec.GridSpecFromSubplotSpec(n_extra, 2, gs0[1, 0])

    def cleanse_tick(ax):
        plt.setp(ax.get_xticklabels(), visible=False)
        plt.setp(ax.get_yticklabels(), visible=False)

    def generate_axes(grid_spec, merge_extra_col=False):
        """Generate axes from the grid specs"""
        (gs_i, gs_j) = grid_spec.get_geometry()

        print (gs_i, gs_j)

        axes = []

        for i in range(gs_i):
            axes_j = []
            for j in range(gs_j):
                if j + 1 == gs_j:
                    if merge_extra_col is True:
                        if 2 * i + 1 >= gs_i > 1:
                            continue
                        elif gs_i == 1:
                            continue
                        else:
                            ax = plt.Subplot(f, grid_spec[2*i+1:2*i+3, j])
                    else:
                        ax = plt.Subplot(f, grid_spec[i, j])
                else:
                    ax = plt.Subplot(f, grid_spec[i, j])
                cleanse_tick(ax)
                f.add_subplot(ax)
                axes_j.append(ax)
            axes.append(axes_j)

        return axes

    gs1_axes = generate_axes(gs1, merge_extra_col=extra_col)
    gs3_axes = generate_axes(gs3, merge_extra_col=False)
    gs4_axes = generate_axes(gs4, merge_extra_col=False)

    total_axes = {
        'right': gs1_axes,
        'left': gs3_axes,
        'left_b': gs4_axes
    }

    return f, total_axes, [gs1, gs3, gs4]


def analyze_plot_publication(sample, ims_results, acts, mb, S=(28, 28), specials=None, saturations_in_z=False,
                             plot_classification=False):
    """Plot colored visualization of the groups and masks.

    Assumes 4 slots.
    """
    font = {
        'family': 'sans-serif',
        'weight': 'bold',
        'size': 20
    }

    tagger_output = AttributeDict(
        z_hat=acts.clean.z,
        mask=acts.clean.m,
        ami_cost_per_sample=acts.clean.ami_score_per_sample)

    TIME, GROUPS = tagger_output.z_hat.shape[:2]

    n_extra = len(specials)
    n_left = len(ims_results)

    if plot_classification is True:
        f, axes, gss = create_fig_layout(GROUPS, TIME, n_left, n_extra, extra_col=True)
    else:
        f, axes, gss = create_fig_layout(GROUPS, TIME, n_left, n_extra)

    # plot the normal results.
    for im_index, im in enumerate(ims_results):
        orig = mb['features_unlabeled'][im].reshape(*S)
        z = tagger_output.z_hat[:, :, im].reshape(TIME, GROUPS, *S)
        m = tagger_output.mask[:, :, im].reshape(TIME, GROUPS, *S)

        show_mat(orig, axes['left'][im_index][0])

        axes['left'][im_index][0].set_ylabel('${0:.2f}$'.format(tagger_output.ami_cost_per_sample[-1, im]), fontdict=font)
        show_reconstr(z[-1], m[-1], axes['left'][im_index][1])

    if specials is not None:
        # plot the special results.
        for im_index, im in enumerate(specials):
            # path_x is the key to plot the digit removal example in Freq20-2MNIST experiment.
            if 'path_x' in im.keys():
                path_data = np.load(im['path_x'])
                orig = path_data['x']
                z = path_data['z']
                m = path_data[im['type']]

                show_mat(orig[0], axes['left_b'][im_index][0])
                show_reconstr(z, m, axes['left_b'][im_index][1])
            else:
                orig = im['mb']['features_unlabeled'][im['index']].reshape(*S)

                TIME_temp, GROUPS_temp = im['acts'].clean.z.shape[:2]

                z = im['acts'].clean.z[:, :, im['index']].reshape(TIME_temp, GROUPS_temp, *S)
                m = im['acts'].clean.m[:, :, im['index']].reshape(TIME_temp, GROUPS_temp, *S)

                show_mat(orig, axes['left_b'][im_index][0])

                show_reconstr(z[-1], m[-1], axes['left_b'][im_index][1])

    axes['left_b'][-1][0].set_xlabel('$original$', fontdict=font)
    axes['left_b'][-1][1].set_xlabel('$reconst.$', fontdict=font)

    z = tagger_output.z_hat[:, :, sample].reshape(TIME, GROUPS, *S)
    m = tagger_output.mask[:, :, sample].reshape(TIME, GROUPS, *S)

    for j in range(TIME):
        for i in range(GROUPS):
            if saturations_in_z is True:
                show_reconstr(z[j], None, axes['right'][i * 2 + 1][j], channel=i, saturations_map=m[j])
            else:
                show_reconstr(z[j], None, axes['right'][i * 2 + 1][j], channel=i)
            show_reconstr(m[j], None, axes['right'][i * 2 + 2][j], channel=i)
        show_reconstr(z[j], m[j], axes['right'][0][j])

    color_conv = map(rgb2hex, map(lambda x: x, GROUP_COLORS))

    if plot_classification is True:
        classification = acts.clean.pred[-1, :, sample]

        axes['right'][0][-1].set_title("$Class$", fontdict=font, fontsize=24)

        for i in range(m.shape[1]):
            # here we plot per group
            # print i+1
            axes['right'][i][-1].bar(np.arange(10)+0.1, classification[i])
            axes['right'][i][-1].set_ylim(0., 0.8)
            axes['right'][i][-1].yaxis.set_label_position("right")

            class_label = classification[i].argsort()[::-1][0]
            if np.max(classification[i]) > 0.2:
                axes['right'][i][-1].set_ylabel('$Pred.: {}$'.format(class_label), fontdict=font, fontsize=24)
            else:
                axes['right'][i][-1].set_ylabel('$Pred.: no\ class$', fontdict=font, fontsize=24)
        # then plot all group

    # Set the titles or x labels
    for j in range(TIME):
        axes['right'][0][j].set_title('$i={}$'.format(j), fontdict=font, fontsize=24)
        axes['right'][0][j].spines['top'].set_linewidth(3)
        axes['right'][0][j].spines['left'].set_linewidth(3)
        axes['right'][0][j].spines['right'].set_linewidth(3)
        axes['right'][0][j].spines['bottom'].set_linewidth(3)

    thickness = 2
    for j in range(TIME):
        for i in range(GROUPS):
            axes['right'][2 * i + 1][j].spines['top'].set_linewidth(thickness)
            axes['right'][2 * i + 1][j].spines['bottom'].set_linewidth(thickness)
            axes['right'][2 * i + 1][j].spines['left'].set_linewidth(thickness)
            axes['right'][2 * i + 1][j].spines['right'].set_linewidth(thickness)
            axes['right'][2 * i + 1][j].spines['top'].set_color(color_conv[i])
            axes['right'][2 * i + 1][j].spines['left'].set_color(color_conv[i])
            axes['right'][2 * i + 1][j].spines['right'].set_color(color_conv[i])
            axes['right'][2 * i + 1][j].spines['bottom'].set_color(color_conv[i])

            axes['right'][2 * i + 2][j].spines['top'].set_linewidth(thickness)
            axes['right'][2 * i + 2][j].spines['bottom'].set_linewidth(thickness)
            axes['right'][2 * i + 2][j].spines['left'].set_linewidth(thickness)
            axes['right'][2 * i + 2][j].spines['right'].set_linewidth(thickness)
            axes['right'][2 * i + 2][j].spines['bottom'].set_color(color_conv[i])
            axes['right'][2 * i + 2][j].spines['left'].set_color(color_conv[i])
            axes['right'][2 * i + 2][j].spines['right'].set_color(color_conv[i])
            axes['right'][2 * i + 2][j].spines['top'].set_color(color_conv[i])

    # Print y labels
    axes['right'][0][0].set_ylabel("$reconst.$", fontdict=font, fontsize=24)

    for i in range(GROUPS):
        axes['right'][2*i+1][0].set_ylabel("$z_{}$".format(i), fontdict=font, fontsize=24)
        axes['right'][2*i+2][0].set_ylabel("$m_{}$".format(i), fontdict=font, fontsize=24)

    f.tight_layout()

    return f
