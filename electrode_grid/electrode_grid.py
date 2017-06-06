import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import brewer2mpl as b2mpl
from tqdm import tqdm
import seaborn.apionly as sns
from .FS_colorLUT import get_lut

#TODO: make freesurfer colors default

def electrode_grid(data, plot_func=plt.plot, align_window=None, labels=None, elects_to_plot=None, anatomy=None,
                   yscale='shared', anat_colors=None, xlabel_elect=None, ylabel_elect=256, anat_legend=True,
                   elect_num_pos='ul', clinical_num_pos=None, grid_orientation='lr', gridx=16, gridy=16,
                   xlabel='time (s)', ylabel='HG', channel_order=None, fig=None,
                   anat_legend_kwargs = {}, clinical_number_kwargs={}, elec_number_kwargs={}):
    """
    Plots any function on an electrode grid

    Inputs:
    data: np.array(elects x time) or np.array(elects x time x case) or None
    align_window: time before and after stimulus in seconds
    elects_to_plot: list of electrodes by number indexed from 0 or boolean array
    yscale:
        individual: Each of the electrodes as its own y-scale
        {shared}:   The y-scale is the same across all electrodes. The max and min of the entire set is taken
        range:      Given a defined list of min and max, will scale all electrodes to that
    anatomy: np.array of anatomy_ID for each electrode. Used to color background of each axis.
    co_anat: name of seaborn colormap to use for anatomy.
    labels: Labels for the data legend.
    anat_legend=True show anatomy legend?
    elect_num_pos='ul' ("upper left") other options: 'ur', 'll', 'lr', None (do not show)
    clinical_num_pos=None, options same as elect_num_pos
    grid_orientation='lr': Where is electrode 1?
    gridx=16, gridy=16: number of electrodes on grid
    elect_num_color='k'
    xlabel='time (s)'
    ylabel='$HG$'

    Returns:
        fig
        axs
    """

    nelect = gridx * gridy

    if fig is None:
        fig = plt.figure(figsize=(gridx, gridy))

    clinical_electrodes = np.hstack([np.arange(1, 16, 2) + x for x in np.arange(0, 256, 32)])

    if elects_to_plot is None:
        elects_to_plot = np.arange(nelect)
    elects_to_plot = np.array(elects_to_plot)
    if elects_to_plot.dtype == np.bool:
        elects_to_plot = np.where(elects_to_plot)[0]

    if align_window is None:
        if data is not None:
            align_window = (0, data.shape[-1])
        else:
            align_window = (0, 1)

    if data is not None:
        tt = np.linspace(align_window[0], align_window[1], data.shape[1])

    if xlabel_elect is None:
        if elects_to_plot is None:
            xlabel_elect = 241
        else:
            rows = np.arange(256, 0, -16)
            for row in rows:
                col = np.arange(row - 16, row)
                elecs = [x for x in elects_to_plot if x in col]
                if len(elecs):
                    xlabel_elect = min(elecs) + 1
                    ylabel_elect = max(elecs) + 1
                    break

    # Define axes face color to illustrate anatomy
    anat_color_array = np.array([0.9, 0.9, 0.9] * nelect)  # initialize to light gray
    if anatomy is None and anat_colors is not None:
        raise Warning('anat colors set without anatomy specified.')
    else:
        if anat_colors is None:
            anat_colors = get_lut()
        anatomy = anatomy[elects_to_plot]
        regions, region_indices = np.unique(anatomy, return_inverse=True)
        anat_legend_h = []
        for i, region in enumerate(regions):
            color = anat_colors['ctx-rh-' + region] / 255.
            anat_color_array[region_indices == i] = color
            anat_legend_h.append(mpl.patches.Patch(color=color, label=region))

        # create smart anat legend
        true_anat_legend_kwargs = {'loc': 'upper_left'}
        if len(regions) > 4:
            true_anat_legend_kwargs.update(ncol=2)
        true_anat_legend_kwargs.update(**anat_legend_kwargs)
        if anat_legend:
            fig.legend(handles=anat_legend_h, labels=regions, **anat_legend_kwargs)

    if channel_order == None:
        # Rotate grid by indexing
        if grid_orientation == 'lr':
            channel_order = np.rot90(np.array(range(1, nelect + 1)).reshape(gridx, gridy).transpose(), 2).reshape(nelect)
        elif grid_orientation == 'll':
            channel_order = np.rot90(np.array(range(1, nelect + 1)).reshape(gridx, gridy).transpose(), 1).reshape(nelect)
        elif grid_orientation == 'ur':
            channel_order = np.rot90(np.array(range(1, nelect + 1)).reshape(gridx, gridy).transpose(), 3).reshape(nelect)
        elif grid_orientation == 'straight':
            channel_order = np.arange(nelect) + 1

    # set electrode number properties
    true_elec_number_kwargs = {'fontsize': 10, 'color': 'k'}
    true_elec_number_kwargs.update(**elec_number_kwargs)

    true_clinical_number_kwargs = {'fontsize': 10, 'color': 'r'}
    true_clinical_number_kwargs.update(**clinical_number_kwargs)

    # Add subplot for each electrode
    a = []  # axes list
    ylims = []
    for ielect, (elect_number, this_anat_color) in tqdm(enumerate(zip(elects_to_plot, anat_colors)),
                                                        total=len(elects_to_plot), desc='Drawing electrode grid'):

        ax = fig.add_subplot(gridy, gridx, channel_order[elect_number])

        if data is not None:
            h_lines = plot_func(tt, data[elect_number])

        ylims.append(ax.get_ylim())
        ax.set_xlim(align_window)
        ax.set_axis_bgcolor(this_anat_color)
        for spine in ax.spines.values():
            spine.set_color(this_anat_color)
        a.append(ax)

        # Add electrode number
        if elect_num_pos:
            corner_text(str(elect_number + 1), elect_num_pos, ax, **true_elec_number_kwargs)

        if clinical_num_pos:
            if (elect_number + 1) in clinical_electrodes:
                clinical_number = np.where(clinical_electrodes == (elect_number + 1))[0] + 1
                corner_text(str(clinical_number), clinical_num_pos, ax, **true_clinical_number_kwargs)

        # Add axes labels only on corner electrodes
        ax.set_xticks([align_window[0], 0, align_window[1]])

        if elect_number == xlabel_elect - 1:
            ax.set_yticklabels([])
            ax.set_xlabel(xlabel)
        elif (elect_number == ylabel_elect - 1) and yscale != 'individual':
            ax.set_xticklabels([])
            ax.set_ylabel(ylabel)
        else:
            ax.set_yticklabels([])
            ax.set_xticklabels([])

    if labels is not None:
        fig.legend(h_lines, labels, loc='upper right')

    # Scale using min and max values from all axes
    if not isinstance(yscale, str):
        shared_ylim = yscale
        for axi in a:
            axi.set_ylim(shared_ylim)

    elif yscale == 'shared':
        all_min = min(x[0] for x in ylims)
        all_max = max(x[1] for x in ylims)
        shared_ylim = [all_min, all_max]
        for axi in a:
            axi.grid(True)
            axi.set_ylim(shared_ylim)
            if (shared_ylim[0] < 0) and (shared_ylim[1] > 0):
                axi.set_yticks((shared_ylim[0], 0, shared_ylim[1]))
            else:
                axi.set_yticks(shared_ylim)

    elif yscale == 'individual':
        pass

    else:
        print('yscale setting not recognized')

    # for axi in a:
    #    plt.plot([0,0],axi.get_ylim(),color='black', axes=axi)

    plt.subplots_adjust(left=.07, right=.99, bottom=.1, top=.86, hspace=0.1, wspace=0.1)

    return fig, a


def show_erps(Ds, labels=None, align_window=None, show_sem=True, co_data=None, elects_to_plot=None, gridx=16, gridy=16,
              yscale=(-.5, 2), **kwargs):
    """
    :param Ds: data; np.array.shape = (electrodes, time, trials)
    :param labels:
    :param align_window:

    Optional (show_erps):
    :param show_sem: bool

    Optional (electrodeGrid):
    elects_to_plot = None,
    anatomy = None
    yscale = 'shared'
    co_anat = 'Pastel2'
    x_axis_label_elect=None (really 241 or 1)
    y_axis_label_elect=256
    figsize=10


    :return: fig, axs
    """
    if len(align_window) is not 2:
        raise Exception("align_window must have a length of 2")

    nelect = gridx * gridy
    if elects_to_plot is None:
        elects_to_plot = np.arange(nelect)
    elects_to_plot = np.array(elects_to_plot)
    if elects_to_plot.dtype == np.bool:
        elects_to_plot = np.where(elects_to_plot)[0]

    if co_data is None:
        co_data = b2mpl.get_map('Set1', 'Qualitative', 4).mpl_colors[1:]

    fig, axs = electrode_grid(None, align_window=align_window, elects_to_plot=elects_to_plot, gridx=gridx, gridy=gridy,
                             yscale=yscale, **kwargs)

    #if co_data is not None:
    #    [ax.set_prop_cycle(cycler('color',co_data)) for ax in axs]

    h_lines = []
    for D, color in zip(Ds,co_data):
        D = D[np.array(elects_to_plot).astype(int)]
        mean_erp = np.ma.mean(D, axis=2)
        tt = np.linspace(align_window[0], align_window[1], mean_erp.shape[1])
        for ax, data in tqdm(zip(axs, mean_erp), desc='Drawing data'):
            h_line, = ax.plot(tt, data, color=color, linewidth=1)
        h_lines.append(h_line)

        if show_sem:
            sem_erp = np.ma.std(D, axis=2) / np.sqrt(D.shape[2])
            for ax, data, err in tqdm(zip(axs, mean_erp, sem_erp), desc='Drawing sem'):
                ax.fill_between(tt, data - err, data + err, alpha=.4, facecolor=color, edgecolor=color)

    for ax in axs:
        yl = ax.get_ylim()
        ax.set_yticks((yl[0], 0, yl[1]))
        ax.grid(True)
    #axs[-1].set_yticklabels(axs[-1].get_yticks())

    #axs[240].set_xticks((np.min(tt),0,np.max(tt)))
    #axs[240].xaxis.set_ticklabels((np.min(tt),0,np.max(tt)))
    #axs[240].set_xlabel('times (s)')
    if labels is not None:
        fig.legend(h_lines, labels, loc='upper right', ncol=2)

    return fig, axs


def get_channel_order(subject):
    if subject == 'EC141':
        co1 = np.rot90((np.arange(128) + 129).reshape(8, 16).T, 3).ravel()
        co2 = np.rot90((np.arange(128) + 1).reshape(8, 16).T, 3).ravel()
        channel_order = np.concatenate((co1, co2))
    else:
        channel_order = None

    return channel_order


def compute_ERP(Ds):
    data_list = []
    for iD, D in enumerate(Ds):
        this_means = np.mean(D, axis=2)
        this_stes = np.std(D, axis=2) / np.sqrt(D.shape[2])
        D2 = np.concatenate((np.expand_dims(this_means, 1), np.expand_dims(this_stes, 1)), 1)
        data_list.append(D2)
    data_list = np.array(data_list)
    data_list = np.swapaxes(data_list, 0, 1)
    data_list = np.swapaxes(data_list, 1, 3)
    data_list = np.swapaxes(data_list, 2, 3)

    return data_list


def plotSingleERP(tt, data, labels=None, alpha=.5, co=b2mpl.get_map('Set1', 'Qualitative', 8).mpl_colors, ax=None):
    # expects data of size ntime x ncases x 1 (for just mean) or 2 (for mean and standard error)
    data = np.swapaxes(data, 0, 2)
    data = np.swapaxes(data, 0, 1)

    if ax is None:
        ax = plt.gca()
    # pdb.set_trace()
    if not labels:
        labels = [' '] * len(data)

    for (di, label, c) in zip(data, labels, co):
        if di.shape[0] <= 2:
            ax.plot(tt, di[0], color=c, label=label)
            if len(di) == 2:
                ax.fill_between(tt, di[0] - di[1], di[0] + di[1], color=c, alpha=alpha)
        else:
            # pdb.set_trace()
            ax.plot(tt, di, color=c, label=label)

    ylim = ax.get_ylim()
    ax.plot([0, 0], [-9999999, 999999999], color='black', alpha=.4)
    # plt.gca().set_yscale("log", nonposx='clip')
    ax.set_ylim(ylim)
    ax.set_xlim(np.min(tt), np.max(tt))

    sns.despine(ax=ax)

    return plt.gca()


def corner_text(text, spec, ax, **kwargs):
    """
    Puts `text` in the corner of `ax` specified by `spec`. additional arguments are passed to plt.text.

    Parameters
    ----------
    s: str
        string to placed
    spec: str
        specifies which corner. e.g. ul = upper left, lr = lower right
    ax: plt.axes
    kwargs: optional arguments get passed to plt.text.

    Returns
    -------
    h: plt.text

    """
    if type(spec) is not str or spec[0] not in ('u', 'l') or spec[1] not in ('l', 'r'):
        raise ValueError('spec must be string: [u|l][l|r]')
    x = 0.
    y = 1.
    va = 'top'
    ha = 'left'
    if spec[0] == 'l':
        y = 0.0
        va = 'bottom'
    if spec[1] == 'r':
        x = 1.0
        ha = 'right'

    kwargs = {'va': va, 'ha': ha}.update(**kwargs)
    h = plt.text(x, y, str(text), transform=ax.transAxes, **kwargs)

    return h
