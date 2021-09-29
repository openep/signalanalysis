import matplotlib.pyplot as plt  # type: ignore
import pandas as pd
from typing import Union, Optional, List, Tuple

import tools.python


def plot(ecgs: Union[List[pd.DataFrame], pd.DataFrame],
         legend_ecg: Optional[List[str]] = None,
         linewidths_ecg: float = 2,
         limits: Union[list, float, None] = None,
         legend_limits: Optional[List[str]] = None,
         plot_sequence: Optional[List[str]] = None,
         single_fig: bool = True,
         colours_ecg: Union[List[str], List[List[float]], List[Tuple[float]], None] = None,
         linestyles_ecg: Optional[List[str]] = '-',
         colours_limits: Union[List[str], List[List[float]], List[Tuple[float]], None] = None,
         linestyles_limits: Optional[List[str]] = None,
         fig: Optional[plt.figure] = None,
         ax=None) -> tuple:
    """
    Plot and label the ECG data from simulation(s). Optional to add in QRS start/end boundaries for plotting

    Parameters
    ----------
    ecgs : pd.DataFrame or list of pd.DataFrame
        Dataframe or list of dataframes for ECG data, with keys corresponding to the trace name and index to the time
        data
    legend_ecg : list of str, optional
        List of names for each given set of ECG data e.g. ['BCL=300ms', 'BCL=600ms'], default=None
    linewidths_ecg : float, optional
        Width to use for plotting lines, default=3
    limits : float or list of float or pd.DataFrame, optional
        Optional temporal limits (e.g. QRS limits) to add to ECG plots. Can add multiple limits, which will be
        plotted identically on all axes. If provided as a dataframe, will plot the limits on the relevant axis
    legend_limits : list of str, optional
        List of names for each given set of limits e.g. ['QRS start', 'QRS end'], default=None
    plot_sequence : list of str, optional
        Sequence in which to plot the ECG traces. Will default to: V1, V2, V3, V4, V5, V6, LI, LII, LIII, aVR, aVL, aVF
    single_fig : bool, optional
        If true, will plot all axes on a single figure window. If false, will plot each axis on a separate figure
        window. Default is True
    colours_ecg : str or list of str or list of list/tuple of float, optional
        Colours to be used to plot ECG traces. Can provide as either string (e.g. 'b') or as RGB values (floats). Will
        default to ca.get_plot_colours()
    linestyles_ecg : str or list, optional
        Linestyles to be used to plot ECG traces. Will default to ca.get_plot_lines()
    colours_limits : str or list of str or list of list/tuple of float, optional
        Colours to be used to plot limits. Can provide as either string (e.g. 'b') or as RGB values (floats). Will
        default to ca.get_plot_colours()
    linestyles_limits : str or list, optional
        Linestyles to be used to plot limits. Will default to ca.get_plot_lines()
    fig : optional
        If given, will plot data on existing figure window
    ax: optional
        If given, will plot data using existing axis handles

    Returns
    -------
    fig
        Handle to output figure window, or dictionary to several handles if traces are all plotted in separate figure
        windows (if single_fig=False)
    ax : dict
        Dictionary to axis handles for ECG traces

    Raises
    ------
    AssertionError
        Checks that various list lengths are the same
    TypeError
        If input argument is given in an unexpected format
    """

    # Prepare axes and inputs
    if plot_sequence is None:
        plot_sequence = ['V1', 'V2', 'V3', 'V4', 'V5', 'V6', 'LI', 'LII', 'LIII', 'aVR', 'aVL', 'aVF']
    if fig is None and ax is None:
        fig, ax = __init_axes(plot_sequence, single_fig)
    else:
        assert ((fig is not None) and (ax is not None)), 'Fig and ax handles must be passed simultaneously'

    if not isinstance(ecgs, list):
        ecgs = [ecgs]
    n_ecgs = len(ecgs)

    if limits is None:
        n_limits = 1
    else:
        n_limits = len(limits)

    legend_ecg = tools.python.convert_input_to_list(legend_ecg, n_list=n_ecgs)
    legend_limits = tools.python.convert_input_to_list(legend_limits, n_list=n_limits)

    colours_ecg = tools.python.convert_input_to_list(colours_ecg, n_list=n_ecgs, default_entry='colour')
    colours_limits = tools.python.convert_input_to_list(colours_limits, n_list=n_limits, default_entry='colour')
    if linestyles_ecg is None:
        linestyles_ecg = tools.python.convert_input_to_list(linestyles_ecg, n_list=n_ecgs, default_entry='line')
    else:
        linestyles_ecg = tools.python.convert_input_to_list(linestyles_ecg, n_list=n_ecgs, default_entry=linestyles_ecg)
    linestyles_limits = tools.python.convert_input_to_list(linestyles_limits, n_list=n_limits, default_entry='line')
    linewidths_ecg = tools.python.convert_input_to_list(linewidths_ecg, n_list=n_ecgs)

    for (ecg, label, colour, linestyle, linewidth) in zip(ecgs, legend_ecg, colours_ecg, linestyles_ecg,
                                                          linewidths_ecg):
        for key in plot_sequence:
            ax[key].plot(ecg.index, ecg[key])

    # Add limits, if supplied
    if limits is not None:
        if isinstance(limits, pd.DataFrame):
            for key in ax:
                ax[key].axvline(limits[key].values)
        else:
            if isinstance(limits, float):
                limits = [limits]

            # Cycle through each limit provided, e.g. QRS start, QRS end...
            for (limit, label, colour, linestyle) in zip(limits, legend_limits, colours_limits, linestyles_limits):
                if isinstance(limit, pd.Series):
                    limit = limit.values
                for key in ax:
                    ax[key].axvline(limit, label=label, color=colour, alpha=0.5, linestyle=linestyle)

    # Add legend, title and axis labels
    if legend_ecg[0] is not None or legend_limits[0] is not None:
        plt.rc('text', usetex=True)
        plt.legend(bbox_to_anchor=(1.04, 1.1), loc="center left")

    return fig, ax


def __init_axes(plot_sequence: List[str],
                single_fig: bool = True):
    """
    Initialise figure and axis handles

    Based on the required plot_sequence (order in which to plot ECG leads), and whether or not it is required to have
    all the plots on a single figure or on separate figures, will return the required figure and axis handles

    Parameters
    ----------
    plot_sequence : list of str
        Sequence in which to plot the ECG leads (only really important if plotted on single figure rather than
        separate figures)
    single_fig : bool, optional
        Whether or not to plot all ECG data on a single figure, or whether to plot each lead data in a separate figure

    Returns
    -------
    fig
        Handle to figure window(s)
    ax
        Handle to axes
    """

    if single_fig:
        fig = plt.figure()
        i = 1
        ax = dict()
        for key in plot_sequence:
            ax[key] = fig.add_subplot(2, 6, i)
            ax[key].set_title(key)
            i += 1
    else:
        fig = dict()
        ax = dict()
        for key in plot_sequence:
            fig[key] = plt.figure()
            ax[key] = fig[key].add_subplot(1, 1, 1)
            ax[key].set_title(key)
    return fig, ax


def __plot_limits(ax,
                  limits: Union[List[float], float],
                  colours: Union[List[List[float]], List[Tuple[float]], List[str]],
                  linestyles: List[str]) -> None:
    """
    Add limit markers to a given plot (e.g. add line marking start of QRS complex)

    Parameters
    ----------
    ax
        Handle to axis
    limits: list of float or float
        Limits to plot on the axis
    colours : list of list/tuple of float or list of str
        RGB values of colours for the individual limits to plot. For plotting n limits, then should be given as
        [[R1, G1, B1], [R2, G2, B2], ... [Rn, Gn, Bn]]
    linestyles : list of str
        Linestyles to plot for each limit.
    """

    if not isinstance(limits, list):
        limits = [limits]
    assert len(limits) <= len(colours), "Incompatible length of limits to plot and colours"
    assert len(limits) <= len(linestyles), "Incompatible length of limits to plot and linestyles"
    for (sim_limit, sim_colour, sim_linestyle) in zip(limits, colours, linestyles):
        for key in ax:
            ax[key].axvline(sim_limit, color=sim_colour, alpha=0.5, linestyle=sim_linestyle)

    return None

