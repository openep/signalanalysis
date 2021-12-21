import numpy as np
import matplotlib.pyplot as plt
from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from signalanalysis.egm import Egm

plt.style.use('seaborn')


def plot_signal(data: "Egm",
                i_plot: Optional[int] = None,
                plot_bipolar_square: bool = False,
                plot_markers: bool = False,
                plot_peaks: bool = False,
                plot_at: bool = False,
                plot_rt: bool = False):
    """General use function to plot EGM data with flags for various markers as required

    Will plot the unipolar and bipolar EGM signals on the same figure, and if requested, also the RMS trace at
    the same time.

    Parameters
    ----------
    data : signalanalysis.egm.Egm
        EGM data
    i_plot : int, optional
        Which signal from the data to plot, default=random
    plot_bipolar_square : bool, optional
        Whether to plot the squared bipolar data on the figure as well, default=False
    plot_markers : bool, optional
        Whether to plot all the various markers available, i.e. will set plot_peaks, plot_at, etc., to True,
        default=False
    plot_peaks, plot_at, plot_rt : bool, optional
        Whether to plot the points of the bipolar peak/AT/RT on the figure, default=False

    Returns
    -------
    fig, ax
        Handles to the figure and axis data, respectively
    """

    if plot_markers:
        plot_peaks = True
        plot_at = True
        plot_rt = True

    # Pick a random signal to plot as an example trace (making sure to not pick a 'dead' trace)
    if i_plot is None:
        i_plot = data.n_beats.sample().index[0]
        while data.n_beats[i_plot] == 0:
            i_plot = data.n_beats.sample().index[0]
    else:
        if data.n_beats[i_plot] == 0:
            raise IOError("No beats detected in specified trace")

    fig = plt.figure()
    fig.suptitle('Trace {}'.format(i_plot))
    ax = dict()
    ax_labels = ['Unipolar', 'Bipolar']
    plot_data = [data.data_uni, data.data_bi]
    if plot_bipolar_square:
        ax_labels.append('Bipolar^2')
        plot_data.append(np.square(data.data_bi))

    for i_ax, ax_data in enumerate(plot_data):
        ax[ax_labels[i_ax]] = fig.add_subplot(len(plot_data), 1, i_ax + 1)
        ax[ax_labels[i_ax]].plot(ax_data.loc[:, i_plot], color='C0')
        ax[ax_labels[i_ax]].set_ylabel(ax_labels[i_ax])

        if plot_peaks:
            ax[ax_labels[i_ax]].scatter(data.t_peaks[i_plot].dropna(),
                                        ax_data.loc[:, i_plot][data.t_peaks[i_plot].dropna()],
                                        label='Peaks',
                                        marker='o', edgecolor='tab:orange', facecolor='none', linewidths=2)
            if ax_labels[i_ax] == 'Bipolar^2':
                ax[ax_labels[i_ax]].axhline(data.n_beats_threshold * ax_data.loc[:, i_plot].max(),
                                            color='tab:orange', linestyle='--')

        if plot_at:
            ax[ax_labels[i_ax]].scatter(data.at[i_plot].dropna(),
                                        ax_data.loc[:, i_plot][data.at[i_plot].dropna()],
                                        label='AT',
                                        marker='d', edgecolor='tab:green', facecolor='none', linewidths=2)

        if plot_rt:
            try:
                ax[ax_labels[i_ax]].scatter(data.rt[i_plot].dropna(),
                                            ax_data.loc[:, i_plot][data.rt[i_plot].dropna()],
                                            label='RT',
                                            marker='s', edgecolor='tab:red', facecolor='none', linewidths=2)
            except KeyError:
                pass

    # Add legend to top axis
    ax[ax_labels[0]].legend()

    return fig, ax
