from copy import copy
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import brewer2mpl as b2mpl
from tqdm import tqdm
import seaborn.apionly as sns
from .FS_colorLUT import get_lut
from seaborn.utils import set_hls_values


def electrode_grid(gridx=16, gridy=16, elects_to_plot=None, anatomy=None, xlims=(-1., 1.), ylims=(-.5, 2.),
                   anat_colors=None, xlabel_elect=None, ylabel_elect=256, anat_legend=True, elect_num_pos='ul',
                   clinical_num_pos=None, grid_orientation='lr', xlabel='time (s)', ylabel='HG', channel_order=None,
                   fig=None, anat_legend_kwargs=None, clinical_number_kwargs=None, elec_number_kwargs=None):
    """
    Create electrod grid for plotting.

    Parameters
    ----------
    gridx, gridy: int
        How many electrodes are there in the x and y dimensions of the grid? (Default: 16)
    elects_to_plot: list
        List of electrodes by number indexed from 0 or boolean array
    anatomy: np.array
        List of anatomy_ID for each electrode. Used to color background of each axis.
    xlims, ylims: tuple
        axes are set using these xlims and ylims
    anat_colors: None o| dict
        region (str) -> r,g,b (as 0>x>1). Default is the freesurfer colormap, but any of the rois colors can be replaced
        here.
    xlabel_elect, ylabel_elect: int
        Which electrode gets the xlabel and ylabel
    anat_legend: bool
        whether to show the anatomy legend
    elect_num_pos: str or None
        where to put the electrode number. If None, no electrode number is shown. Otherwise, use
        'ul': upper left (Default)
        'll': lower left
        'ur': upper right
        'lr': lower right
    clinical_num_pos: str or None
        Same as elect_num_pos. 1 in ever 4 electrodes is clinical. Default = None
    grid_orientation: str
        Where is electrode #1? Follows elect_num_pos rules.
    xlabel: str
        Default: 'time (s)'
    ylabel: str
        Default: 'HG'
    channel_order: list
        Should be used if channels are in an arrangement that cannot be captured by grid_orientation. Supercedes
        grid_orientation.
    fig: plt.Figure
        Default: None
    anat_legend_kwargs: None | dict
    clinical_number_kwargs: None | dict
    elec_number_kwargs: None | dict

    Returns
    -------
    fig: plt.Figure
    axs: list(plt.Axes)
    elects_to_plot: np.array()

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
    anat_color_array = np.array([[0.9, 0.9, 0.9]] * nelect)  # initialize to light gray
    if anatomy is None and anat_colors is not {}:
        raise Warning('anat colors set without anatomy specified.')
    else:
        # set up anatomy colors
        input_anat_colors = copy(anat_colors)
        anat_colors = get_lut()  # use freesurfer defaults
        if input_anat_colors:
            anat_colors.update(**input_anat_colors)

        anatomy = anatomy[elects_to_plot]
        regions, region_indices = np.unique(anatomy, return_inverse=True)
        anat_legend_h = []
        for i, region in enumerate(regions):
            color = np.array(anat_colors['ctx-rh-' + region]) / 255. # mpl takes colors as (0≤x≤1)
            anat_color_array[region_indices == i,:] = color
            anat_legend_h.append(mpl.patches.Patch(color=color, label=region))

        # create smart anat legend
        true_anat_legend_kwargs = {'loc': 'upper left'}
        if len(regions) > 4:
            true_anat_legend_kwargs.update(ncol=2)
        if anat_legend_kwargs:
            true_anat_legend_kwargs.update(**anat_legend_kwargs)
        if anat_legend:
            fig.legend(handles=anat_legend_h, labels=regions, **true_anat_legend_kwargs)

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
    true_elec_number_kwargs = {'fontsize': 10}
    if elec_number_kwargs:
        true_elec_number_kwargs.update(**elec_number_kwargs)

    true_clinical_number_kwargs = {'fontsize': 10, 'color': 'r'}
    if clinical_number_kwargs:
        true_clinical_number_kwargs.update(**clinical_number_kwargs)

    # Add subplot for each electrode
    a = []  # axes list
    for ielect, (elect_number, this_anat_color) in tqdm(enumerate(zip(elects_to_plot, anat_color_array)),
                                                        total=len(elects_to_plot), desc='Drawing electrode grid'):

        ax = fig.add_subplot(gridy, gridx, channel_order[elect_number])
        ax.set_xlim(xlims)
        ax.set_ylim(ylims)
        ax.set_axis_bgcolor(set_hls_values(this_anat_color, l=.95))  # lighten color
        for spine in ax.spines.values():
            spine.set_color(this_anat_color)
        a.append(ax)

        # Add electrode numbers
        if elect_num_pos:
            corner_text(str(elect_number + 1), elect_num_pos, ax, **true_elec_number_kwargs)
        if clinical_num_pos:
            if (elect_number + 1) in clinical_electrodes:
                clinical_number = np.where(clinical_electrodes == (elect_number + 1))[0] + 1
                corner_text(str(clinical_number), clinical_num_pos, ax, **true_clinical_number_kwargs)

        # Add axes labels only on corner electrodes
        ax.set_xticks((xlims[0], 0, xlims[1]))

        # set axes labels
        if elect_number == xlabel_elect - 1:
            ax.set_xlabel(xlabel)
        else:
            ax.set_xticklabels([])
        if elect_number == ylabel_elect - 1:
            ax.set_ylabel(ylabel)
        else:
            ax.set_yticklabels([])

    fig.subplots_adjust(left=.05, right=.99, bottom=.1, top=.9, hspace=0.1, wspace=0.1)

    return fig, a, elects_to_plot


def show_erps(Ds, align_window, labels=None, show_sem=True, co_data=None, **kwargs):
    """
    Use plot ERPs on electrode_grid
    Parameters
    ----------
    Ds: list
        list of D tensors (electrodes x time x trials)
    align_window: tuple
        time before and after stim in seconds
    labels: tuple
        Optional. labels for data legend
    show_sem: bool
    co_data: list
        List of RGB (0<x<1) values for the data colors. Default: cbrewer Set1
    kwargs: see electrode_grid

    Returns
    -------
    fig: plt.Figure
    axs: list(plt.Axes)
    """

    if co_data is None:
        co_data = b2mpl.get_map('Set1', 'Qualitative', 4).mpl_colors[1:]

    fig, axs, elects_to_plot = electrode_grid(xlims=align_window, **kwargs)

    h_lines = []
    for D, color in zip(Ds, co_data):
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

    true_kwargs = {'va': va, 'ha': ha}
    true_kwargs.update(**kwargs)
    h = plt.text(x, y, str(text), transform=ax.transAxes, **true_kwargs)

    return h
