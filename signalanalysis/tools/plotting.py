import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import warnings
from mpl_toolkits.mplot3d import Axes3D
from typing import List, Tuple, Union, Optional


def get_plot_colours(n: int = 10, colourmap: Optional[str] = None) -> List[Tuple[float]]:
    """
    Return iterable list of RGB colour values that can be used for custom plotting functions

    Returns a list of RGB colours values, potentially according to a specified colourmap. If n is low enough, will use
    the custom 'tab10' colourmap by default, which will use alternating colours as much as possible to maximise
    visibility. If n is too big, then the default setting is 'viridis', which should provide a gradation of colour from
    first to last.

    Parameters
    ----------
    n : int, optional
        Number of distinct colours required, default=10
    colourmap : str
        Matplotlib colourmap to base the end result on. Will default to 'tab10' if n<11, 'viridis' otherwise

    Returns
    -------
    cmap : list of tuple
        List of RGB values
    """
    if colourmap is None:
        if n < 11:
            colourmap = 'tab10'
        elif n < 13:
            colourmap = 'Set3'
        elif n < 21:
            colourmap = 'tab20'
        else:
            colourmap = 'viridis'

    if n < 11:
        cmap = cm.get_cmap(colourmap, lut=10)
        return [cmap(i) for i in range(n)]
    else:
        cmap = cm.get_cmap(colourmap, lut=n)
        return [cmap(i) for i in range(n)]


def get_plot_lines(n: int = 4) -> Union[List[tuple], List[str]]:
    """Returns different line-styles for plotting

    Parameters
    ----------
    n : int, optional
        Number of different line-styles required

    Returns
    -------
    lines : list of str or list of tuple
        List of different line-styles
    """

    if n <= 4:
        basic_linestyles = ['-', '--', '-.', ':']
        return [basic_linestyles[i] for i in range(n)]
    elif n < 15:
        lines = list()
        lines.append('-')
        dash_gap = 2
        i_lines = 0
        while i_lines < 5:
            # First few iterations to be '-----', '-.-.-.', '-..-..-..-',...
            lines.append((0, tuple([5, dash_gap]+[1, dash_gap]*i_lines)))
            i_lines += 1
        while i_lines < 10:
            # Following iterations to be '--.--', '--..--'. '--...---',...
            lines.append((0, tuple([5, dash_gap, 5, dash_gap, 1, dash_gap]+[1, dash_gap]*(i_lines-5))))
            i_lines += 1
        while i_lines < 15:
            # Following iterations to be '---.---', '---..---', '---...---',...
            lines.append((0, tuple([5, dash_gap, 5, dash_gap, 5, dash_gap, 1, dash_gap]+[1, dash_gap]*(i_lines-10))))
            i_lines += 1
        return lines
    else:
        raise Exception('Unsure of how effective this number of different linestyles will be...')


def write_colourmap_to_xml(start_data: float,
                           end_data: float,
                           start_highlight: float,
                           end_highlight: float,
                           opacity_data: float = 1,
                           opacity_highlight: float = 1,
                           n_tags: int = 20,
                           colourmap: str = 'viridis',
                           outfile: str = 'colourmap.xml') -> None:
    """
    Create a Paraview friendly colourmap useful for highlighting a particular range

    Creates a colourmap that is entirely gray, save for a specified region of interest that will vary according to the
    specified colourmap

    Input parameters (required):
    ----------------------------
    start_data                      start value for overall data (can't just use data for region of interest -
                                    Paraview will scale)
    end_data                        end value for overall data
    start_highlight                 start value for region of interest
    end_highlight                   end value for region of interest

    Input parameters (optional):
    ----------------------------

    opacity_data        1.0                 overall opacity to use for all data
    opacity_highlight   1.0                 opacity for region of interest
    colourmap           'viridis'           colourmap to use
    outfile             'colourmap.xml'     filename to save .xml file under

    Output parameters:
    ------------------
    None
    """

    cmap = get_plot_colours(n_tags, colourmap=colourmap)

    # Get values for x, depending on start and end values
    x_offset = 0.2      # Value to provide safespace round x values
    x_maintain = 0.01   # Value to maintain safespace round n_tag values
    cmap_x_data = np.linspace(start_data, end_data, 20)
    cmap_x_data = np.delete(cmap_x_data, np.where(np.logical_and(cmap_x_data > start_highlight-x_offset,
                                                                 cmap_x_data < end_highlight+x_offset)))
    cmap_x_highlight = np.linspace(start_highlight-x_offset, end_highlight+x_offset, n_tags)

    # Extract colourmap name from given value for outfile
    if outfile.endswith('.xml'):
        name = outfile[:-4]
    else:
        name = outfile[:]
        outfile = outfile+'.xml'

    # Write to file
    with open(outfile, 'w') as pFile:
        # pFile.write('<ColorMaps>\n'.format(name))
        pFile.write('\t<ColorMap name="{}" space="RGB">\n'.format(name))

        # Write non-highlighted data values
        for x in cmap_x_data:
            pFile.write('\t\t<Point x="{}" o="{}" r="0.5" g="0.5" b="0.5"/>\n'.format(x, opacity_data))
        pFile.write('\t\t<Point x="{}" o="{}" r="0.5" g="0.5" b="0.5"/>\n'.format(start_highlight-3*x_offset,
                                                                                  opacity_data))
        pFile.write('\t\t<Point x="{}" o="{}" r="0.5" g="0.5" b="0.5"/>\n'.format(end_highlight+3*x_offset,
                                                                                  opacity_data))

        # Write highlighted data values
        for (rgb, x) in zip(cmap, cmap_x_highlight):
            pFile.write('\t\t<Point x="{}" o="{}" r="{}" g="{}" b="{}"/>\n'.format(x-x_maintain, opacity_highlight,
                                                                                   rgb[0], rgb[1], rgb[2]))
            pFile.write('\t\t<Point x="{}" o="{}" r="{}" g="{}" b="{}"/>\n'.format(x, opacity_highlight,
                                                                                   rgb[0], rgb[1], rgb[2]))
            pFile.write('\t\t<Point x="{}" o="{}" r="{}" g="{}" b="{}"/>\n'.format(x+x_maintain, opacity_highlight,
                                                                                   rgb[0], rgb[1], rgb[2]))

        pFile.write('\t</ColorMap>\n')
        pFile.write('</ColorMaps>')

    return None


def set_axis_limits(ax,
                    data: Optional[pd.DataFrame] = None,
                    unit_min: bool = True,
                    axis_limits: Optional[Union[List[float], float]] = None,
                    pad_percent: float = 0.01) -> None:
    """Set axis limits (not automatic for line collections, so needs to be done manually)

    Parameters
    ----------
    ax
        Handles to the axes that need to be adjusted
    data : pd.DataFrame, optional
        Data that has been plotted, default=None
    unit_min : bool, optional
        Whether to have the axes set to, as a minimum, unit length, default=True
    axis_limits : list of float or float, optional
        Min/max values for axes, either as one value (i.e. min=-max), or two separate values. Same axis limits will
        be applied to all dimensions
    pad_percent : float, optional
        Percentage 'padding' to add to the ranges, to try and ensure that the edges of linewidths are not cut off,
        default=0.01
    """
    assert 0 < pad_percent < 0.1, "pad_percent is set to 'unusual' values..."

    if axis_limits is None:
        if data is not None:
            # noinspection PyArgumentList
            ax_min = data.min().min()
            # noinspection PyArgumentList
            ax_max = data.max().max()
        else:
            ax_min = min([np.amin(temp_line.get_xydata()) for temp_line in ax.get_lines()])
            ax_max = max([np.amax(temp_line.get_xydata()) for temp_line in ax.get_lines()])
        if abs(ax_min) > abs(ax_max):
            ax_max = -ax_min
        else:
            ax_min = -ax_max
        if unit_min:
            if ax_max < 1:
                ax_min = -1
                ax_max = 1
    else:
        if isinstance(axis_limits, list):
            ax_min = axis_limits[0]
            ax_max = axis_limits[1]
        else:
            if axis_limits < 0:
                axis_limits = -axis_limits
            ax_min = -axis_limits
            ax_max = axis_limits
    pad_value = (ax_max-ax_min)*pad_percent
    ax.set_xlim(ax_min-pad_value, ax_max+pad_value)
    ax.set_ylim(ax_min-pad_value, ax_max+pad_value)
    if data.shape[1] == 3:
        ax.set_zlim(ax_min, ax_max)
        ax.set_aspect('auto', adjustable='box')
    else:
        ax.set_aspect('equal', adjustable='box')
    return None


def add_colourbar(limits: List[float],
                  fig: Optional[plt.figure] = None,
                  colourmap: str = 'viridis',
                  n_elements: int = 100) -> None:
    """Add arbitrary colourbar to a figure, for instances when an automatic colorbar isn't available

    Parameters
    ----------
    limits : list of float
        Numerical limits to apply
    fig : plt.figure, optional
        Figure on which to plot the colourbar. If not provided (default=None), then will pick up the figure most
        recently available
    colourmap : str, optional
        Colourmap to be used, default='viridis'
    n_elements : int, optional
        Number of entries to be made in the colourmap index, default=100

    Notes
    -----
    This is useful for instances such as when LineCollections are used to plot line that changes colour during the
    plotting process, as LineCollections do not enable an automatic colorbar to be added to the plot. This function
    adds a dummy colorbar to replace that.
    """

    if fig is None:
        fig = plt.gcf()

    cmap = plt.get_cmap(colourmap, n_elements)
    # noinspection PyUnresolvedReferences
    norm = mpl.colors.Normalize(vmin=limits[0], vmax=limits[1])
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array(np.ndarray([]))
    fig.colorbar(sm)
    return None


def add_xyz_axes(fig: plt.figure,
                 ax: Axes3D,
                 axis_limits: Optional[Union[float, List[float], List[List[float]]]] = None,
                 symmetrical_axes: bool = False,
                 equal_limits: bool = False,
                 unit_axes: bool = False,
                 sig_fig: int = None) -> None:
    """ Plot dummy axes (can't move splines in 3D plots)

    Parameters
    ----------
    fig : plt.figure
        Figure handle
    ax : Axes3D
        Axis handle
    axis_limits : float or list of float or list of list of float, optional
        Axis limits, either same for all dimensions (min=-max), or individual limits ([min, max]), or individual limits
        for each dimension
    symmetrical_axes : bool, optional
        Apply same limits to x, y and z axes
    equal_limits : bool, optional
        Set axis minimum to minus axis maximum (or vice versa)
    unit_axes : bool, optional
        Apply minimum of -1 -> 1 for axis limits
    sig_fig : int, optional
        Maximum number of decimal places to be used on the axis plots (e.g., if set to 2, 0.12345 will be displayed
        as 0.12). Used to avoid floating point errors, default=None (no adaption made)
    """

    """ Construct dummy 3D axes - make sure they're equal sizes """
    # Extract all current axis properties before we start plotting anything new and changing them!
    x_min, x_max = ax.get_xlim()
    y_min, y_max = ax.get_ylim()
    z_min, z_max = ax.get_zlim()
    ax_min = min([x_min, y_min, z_min])
    ax_max = max([x_max, y_max, z_max])
    if equal_limits:
        x_min, x_max = set_symmetrical_axis_limits(x_min, x_max, unit_axes=unit_axes)
        y_min, y_max = set_symmetrical_axis_limits(y_min, y_max, unit_axes=unit_axes)
        z_min, z_max = set_symmetrical_axis_limits(z_min, z_max, unit_axes=unit_axes)
        ax_min, ax_max = set_symmetrical_axis_limits(ax_min, ax_max, unit_axes=unit_axes)
    if symmetrical_axes:
        x_min = ax_min
        y_min = ax_min
        z_min = ax_min
        x_max = ax_max
        y_max = ax_max
        z_max = ax_max

    # Adjust axis limits if requested
    if axis_limits is not None:
        if not isinstance(axis_limits, list):
            if axis_limits < 0:
                axis_limits = -axis_limits
            if -axis_limits > min([x_min, y_min, z_min]):
                warnings.warn('Lower limit provided greater than automatic.')
            if axis_limits < max([x_max, y_max, z_max]):
                warnings.warn('Upper limit provided less than automatic.')
            x_min = -axis_limits
            x_max = axis_limits
            y_min = -axis_limits
            y_max = axis_limits
            z_min = -axis_limits
            z_max = axis_limits
        elif not isinstance(axis_limits[0], list):
            # If same axis limits applied to all 3 dimensions
            if axis_limits[0] > min([x_min, y_min, z_min]):
                warnings.warn('Lower limit provided greater than automatic.')
            if axis_limits[1] < max([x_max, y_max, z_max]):
                warnings.warn('Upper limit provided less than automatic.')
            x_min = axis_limits[0]
            x_max = axis_limits[1]
            y_min = axis_limits[0]
            y_max = axis_limits[1]
            z_min = axis_limits[0]
            z_max = axis_limits[1]
        else:
            # Different axis limits provided for each dimension
            x_min = axis_limits[0][0]
            x_max = axis_limits[0][1]
            y_min = axis_limits[1][0]
            y_max = axis_limits[1][1]
            z_min = axis_limits[2][0]
            z_max = axis_limits[2][1]
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_zlim(z_min, z_max)
    x_ticks = ax.get_xticks()
    y_ticks = ax.get_yticks()
    z_ticks = ax.get_zticks()
    x_ticks = x_ticks[(x_ticks >= x_min) & (x_ticks <= x_max)]
    y_ticks = y_ticks[(y_ticks >= y_min) & (y_ticks <= y_max)]
    z_ticks = z_ticks[(z_ticks >= z_min) & (z_ticks <= z_max)]

    # Plot splines - limit them to the min/max tick rather than the automatic axes, as this is slightly prettier
    ax.plot([0, 0], [0, 0], [x_ticks[0], x_ticks[-1]], 'k', linewidth=1.5)
    ax.plot([0, 0], [y_ticks[0], y_ticks[-1]], [0, 0], 'k', linewidth=1.5)
    ax.plot([z_ticks[0], z_ticks[-1]], [0, 0], [0, 0], 'k', linewidth=1.5)

    # Import tick markers (use only those tick markers for the longest axis, as the changes are made to encourage a
    # square set of axes)
    x_tick_range = (x_max-x_min)/100
    y_tick_range = (y_max-y_min)/100
    z_tick_range = (z_max-z_min)/100
    for x_tick in x_ticks:
        ax.plot([x_tick, x_tick], [-x_tick_range, x_tick_range], [0, 0], 'k', linewidth=1.5)
    for y_tick in y_ticks:
        ax.plot([-y_tick_range, y_tick_range], [y_tick, y_tick], [0, 0], 'k', linewidth=1.5)
    for z_tick in z_ticks:
        ax.plot([0, 0], [-z_tick_range, z_tick_range], [z_tick, z_tick], 'k', linewidth=1.5)

    # Label tick markers (only at the extremes, to prevent a confusing plot)
    if sig_fig is not None:
        x_ticks = [round(x_tick, sig_fig) for x_tick in x_ticks]
        y_ticks = [round(y_tick, sig_fig) for y_tick in y_ticks]
        z_ticks = [round(z_tick, sig_fig) for z_tick in z_ticks]
    ax.text(x_ticks[0], -x_tick_range*12, 0, x_ticks[0], None)
    ax.text(x_ticks[-1], -x_tick_range*12, 0, x_ticks[-1], None)
    ax.text(y_tick_range*4, y_ticks[0], 0, y_ticks[0], None)
    ax.text(y_tick_range*4, y_ticks[-1], 0, y_ticks[-1], None)
    ax.text(z_tick_range*4, 0, z_ticks[0], z_ticks[0], None)
    ax.text(z_tick_range*4, 0, z_ticks[-1], z_ticks[-1], None)

    # Import axis labels
    ax.text(x_max+x_tick_range, 0, 0, ax.get_xlabel(), None)
    ax.text(0, y_max+y_tick_range, 0, ax.get_ylabel(), None)
    ax.text(0, 0, z_max+z_tick_range*4, ax.get_zlabel(), None)

    # Remove original axes, and eliminate whitespace
    ax.set_axis_off()
    plt.subplots_adjust(left=-0.4, right=1.4, top=1.4, bottom=-0.4)
    if len(fig.axes) > 1:
        # Assume the extra axis is for a colourbar we wish to preserve
        cax = plt.axes([0.9, 0.1, 0.03, 0.8])
        plt.colorbar(mappable=fig.axes[1].collections[0], cax=cax)
    return None


def set_symmetrical_axis_limits(ax_min: float,
                                ax_max: float,
                                unit_axes: bool = False) -> Tuple[float, float]:
    """Sets symmetrical limits for a series of axes

    TODO: fold functionality into set_axis_limits to avoid redundant functions

    Parameters
    ----------
    ax_min : float
        Minimum value for axes
    ax_max : float
        Maximum value for axes
    unit_axes : bool, optional
        Whether to apply a minimum axis range of [-1,1]

    Returns
    -------
    ax_min, ax_max : float
        Symmetrical axis limits, where ax_min=-ax_max
    """
    if abs(ax_min) > abs(ax_max):
        ax_max = -ax_min
    else:
        ax_min = -ax_max

    if unit_axes:
        if ax_max < 1:
            ax_max = 1
            ax_min = -1
    return ax_min, ax_max

