import matplotlib.pyplot as plt
from matplotlib import gridspec
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Line3DCollection
from matplotlib.collections import LineCollection
import numpy as np
import pandas as pd
from math import sin, cos, acos, atan2
import warnings
from typing import List, Tuple, Optional, Union

import tools.python
import tools.plotting
import signalanalysis.vcg

plt.style.use('seaborn')

__all__ = ['Axes3D']    # Workaround to prevent Axes3D import statement to be labelled as unused


def plot_xyz_components(vcgs: Union[pd.DataFrame, List[pd.DataFrame]],
                        legend: Optional[List[str]] = None,
                        colours: Optional[List[List[float]]] = None,
                        linestyles: Optional[List[str]] = None,
                        legend_location: Optional[str] = None,
                        limits: Optional[List[List[float]]] = None,
                        limits_legend: Optional[List[str]] = None,
                        limits_colours: Optional[List[List[float]]] = None,
                        limits_linestyles: Optional[List[str]] = None,
                        limits_legend_location: str = 'lower right',
                        layout: str = 'grid') -> tuple:
    """Plot x, y, z components of VCG data

    Multiple options given for layout of resulting plot

    Parameters
    ----------
    vcgs : list of pd.DataFrame or pd.DataFrame
        List of vcg data: [vcg_data1, vcg_data2, ...]
    legend : list of str, optional
        Legend names for each VCG trace, default=None
    colours : list of list of float or list of str, optional
        Colours to use for plotting, default=common_analysis.get_plot_colours
    linestyles : list of str, optional
        Linestyles to use for plotting, default='-'
    legend_location : str, optional
        Location to plot the legend. Default=None, which will translate to 'best' if no legend is required for
        limits, or 'upper right' if legend is needed for limits
    limits: list of list of float, optional
        QRS limits to plot on axes, default=None
        To be presented in form [[qrs_start1, qrs_starts, ...], [qrs_end1, qrs_end2, ...], ...]
    limits_legend : list of str, optional
        Legend to apply to the limits plotted, default=None
    limits_colours : list of list of float or list of str, optional
        Colours to use when plotting limits, default=common_analysis.get_plot_colours
    limits_linestyles : list of str, optional
        Linestyles to use when plotting limits, default='-'
    limits_legend_location : str, optional
        Location to use for the legend containing the limits data
    layout : {'grid', 'figures', 'combined', 'row', 'column', 'best'}, optional
        Layout of resulting plot
            grid        x,y,z plots are arranged in a grid (like best, but more rigid grid)
            figures     Each x,y,z plot is on a separate figure
            combined    x,y,z plots are combined on a single set of axes
            row         x,y,z plots are arranged on a horizontal row in one figure
            column      x,y,z plots are arranged in a vertical column in one figure
            best        x,y,z plots are arranged to try and optimise space (nb: figures not equal sizes...)

    Returns
    -------
    fig, ax
        Handle for resulting figure(s) and axes
    """
    if not isinstance(vcgs, list):
        vcgs = [vcgs]
    n_vcgs = len(vcgs)
    legend = tools.python.convert_input_to_list(legend, n_list=n_vcgs)

    if layout.lower() == 'combined':
        n_colours = len(vcgs) * 3
    else:
        n_colours = len(vcgs)
    colours = tools.python.convert_input_to_list(colours, n_list=n_colours, default_entry='colour')
    linestyles = tools.python.convert_input_to_list(linestyles, n_list=n_colours, default_entry='line')
    if legend_location is None:
        if limits is None:
            legend_location = 'best'
        else:
            legend_location = 'upper right'

    if limits is not None:
        limits = tools.python.convert_input_to_list(limits, n_list=n_vcgs, list_depth=2)
        n_limits = len(limits)
        limits_legend = tools.python.convert_input_to_list(limits_legend, n_list=n_limits, n_list2=n_limits)
        if n_vcgs == 1:
            # If plotting only one VCG trace, greater flexibility in colours and linestyles, so use the full range
            # (excluding the first colour/linestyle as reserved for the VCG trace itself)
            colours_base = tools.python.convert_input_to_list(limits_colours, n_list=n_limits+1,
                                                              default_entry='colour')[1:]
            limits_colours = [[colour_base] for colour_base in colours_base]
            lines_base = tools.python.convert_input_to_list(limits_linestyles, n_list=n_limits+1,
                                                            default_entry='line')[1:]
            limits_linestyles = [[line_base] for line_base in lines_base]
        else:
            # If plotting multiple VCGs, need to be more circumspect...
            if n_limits == 1:
                # If plotting multiple VCGs with single limits for each VCG, have colours and linestyles matching
                # between VCG and limit. Need to pass the arguments through twice to get the correct format.
                limits_colours = [tools.python.convert_input_to_list(None, n_list=n_vcgs, default_entry='colour')]
                limits_linestyles = [tools.python.convert_input_to_list(None, n_list=n_vcgs, default_entry='line')]
            else:
                # If plotting multiple VCGs with multiple limits each VCG, need to split the load. In this instance,
                # assign different colours to the VCGs with identical linestyles (which thus need to be redefined
                # here), and different linestyles to the limits, with the colour of the VCG matching the colour of the
                # limits
                linestyles = ['-' for _ in range(n_vcgs)]
                limits_colours = [colours for _ in range(n_limits)]
                lines_base = tools.python.convert_input_to_list(None, n_list=n_limits, default_entry='line')
                limits_linestyles = [[line for _ in range(n_vcgs)] for line in lines_base]

    plt.rc('text', usetex=True)

    # Prepare figure and axis handles and layout, depending on requirement
    if layout.lower() == 'figures':
        fig = [plt.figure() for _ in range(3)]
        fig_h = fig
    else:
        fig = plt.figure()
        fig_h = [fig for _ in range(3)]

    ax = dict()
    if layout.lower() == 'figures':
        ax['x'] = fig[0].add_subplot(1, 1, 1, ylabel='x')
        ax['y'] = fig[1].add_subplot(1, 1, 1, ylabel='y', sharex=ax['x'], sharey=ax['x'])
        ax['z'] = fig[2].add_subplot(1, 1, 1, ylabel='z', sharex=ax['x'], sharey=ax['x'])
    elif layout.lower() == 'combined':
        ax['x'] = fig.add_subplot(1, 1, 1)
        ax['y'] = ax['x']
        ax['z'] = ax['x']
    elif layout.lower() == 'row':
        gs = gridspec.GridSpec(3, 1)
        ax['x'] = fig_h[0].add_subplot(gs[0], ylabel='x')
        ax['y'] = fig_h[1].add_subplot(gs[1], ylabel='y', sharex=ax['x'], sharey=ax['x'])
        ax['z'] = fig_h[2].add_subplot(gs[2], ylabel='z', sharex=ax['x'], sharey=ax['x'])
        plt.setp(ax['y'].get_yticklabels(), visible=False)
        plt.setp(ax['z'].get_yticklabels(), visible=False)
        gs.update(wspace=0.025, hspace=0.05)
    elif layout.lower() == 'column':
        gs = gridspec.GridSpec(1, 3)
        ax['x'] = fig_h[0].add_subplot(gs[0], ylabel='x')
        ax['y'] = fig_h[1].add_subplot(gs[1], ylabel='y', sharex=ax['x'], sharey=ax['x'])
        ax['z'] = fig_h[2].add_subplot(gs[2], ylabel='z', sharex=ax['x'], sharey=ax['x'])
        plt.setp(ax['y'].get_xticklabels(), visible=False)
        plt.setp(ax['z'].get_xticklabels(), visible=False)
        gs.update(wspace=0.025, hspace=0.05)
    elif layout.lower() == 'best':
        gs = gridspec.GridSpec(2, 6)
        ax['x'] = fig_h[0].add_subplot(gs[0, :3], ylabel='x')
        ax['y'] = fig_h[1].add_subplot(gs[0, 3:], ylabel='y', sharex=ax['x'], sharey=ax['x'])
        ax['z'] = fig_h[2].add_subplot(gs[1, 2:4], ylabel='z', sharex=ax['x'], sharey=ax['x'])
        plt.setp(ax['y'].get_yticklabels(), visible=False)
        gs.update(wspace=0.025, hspace=0.05)
    elif layout.lower() == 'grid':
        gs = gridspec.GridSpec(2, 2)
        ax['x'] = fig_h[0].add_subplot(gs[0], ylabel='x')
        ax['y'] = fig_h[1].add_subplot(gs[1], ylabel='y', sharex=ax['x'], sharey=ax['x'])
        ax['z'] = fig_h[2].add_subplot(gs[2], ylabel='z', sharex=ax['x'], sharey=ax['x'])
        plt.setp(ax['x'].get_xticklabels(), visible=False)
        plt.setp(ax['y'].get_yticklabels(), visible=False)
        gs.update(wspace=0.025, hspace=0.05)
    else:
        raise IOError("Unexpected value given for layout")

    # Plot data and add legend if required
    h_vcg = dict((key, list()) for key in ax.keys())
    for (vcg, sim_legend, colour, linestyle) in zip(vcgs, legend, colours, linestyles):
        for xyz in vcg:
            h_line, = ax[xyz].plot(vcg[xyz], label=sim_legend, color=colour, linestyle=linestyle)
            h_vcg[xyz].append(h_line)

    if layout == 'figures':
        for xyz in ax:
            h_legend = ax[xyz].legend(handles=h_vcg[xyz], loc=legend_location, handlelength=4.0)
            ax[xyz].add_artist(h_legend)
    else:
        h_legend = ax['x'].legend(handles=h_vcg['x'], loc=legend_location, handlelength=4.0)
        ax['x'].add_artist(h_legend)

    if limits is not None:
        h_limits = dict((key, list()) for key in ax.keys())
        for limit, limit_colours, limit_linestyles, limit_legend in zip(limits, limits_colours, limits_linestyles,
                                                                        limits_legend):
            add_to_legend = True
            for sim_limit, colour, linestyle in zip(limit, limit_colours, limit_linestyles):
                for xyz in ax:
                    h_limit = ax[xyz].axvline(sim_limit, color=colour, alpha=0.5, linestyle=linestyle,
                                              label=limit_legend)
                    if add_to_legend:
                        h_limits[xyz].append(h_limit)
                        add_to_legend = False

        if layout == 'figures':
            for xyz in ax:
                ax[xyz].legend(handles=h_limits[xyz], loc=limits_legend_location, handlelength=4.0)
        else:
            ax['x'].legend(handles=h_limits['x'], loc=limits_legend_location, handlelength=4.0)

    return fig, ax


def plot_2d(vcg: pd.DataFrame,
            x_plot: str = 'x',
            y_plot: str = 'y',
            linestyle: str = '-',
            colourmap: str = 'viridis',
            linewidth: float = 3,
            axis_limits: Union[List[float], float, None] = None,
            fig: Optional[plt.figure] = None) -> plt.figure:
    """
    Plot x vs y (or y vs z, or other combination) for VCG trace, with line colour shifting to show time progression.

    Plot a colour-varying course of a VCG in 2D space

    Parameters
    ----------
    vcg : pd.DataFrame
        VCG data to be plotted
    x_plot, y_plot : str, optional
        Which components of VCG to plot, default='x', 'y'
    linestyle : str, optional
        Linestyle to apply to the plot, default='-'
    colourmap : str, optional
        Colourmap to use for the line, default='viridis'
    linewidth : float, optional
        Linewidth to use, default=3
    axis_limits : list of float or float, optional
        Limits to apply to the axes, default=None
    fig : plt.figure, optional
        Handle to pre-existing figure (if present) on which to plot data, default=None

    Returns
    -------
    fig : plt.figure
        Handle to output figure window
    """

    assert x_plot in vcg.columns, "x_plot value not valid for VCG data"
    assert y_plot in vcg.columns, "y_plot value not valid for VCG data"

    if fig is None:
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
    else:
        ax = fig.gca()

    # Prepare line segments for plotting
    points = np.array([vcg[x_plot], vcg[y_plot]]).transpose().reshape(-1, 1, 2)
    segs = np.concatenate([points[:-1], points[1:]], axis=1)
    lc = LineCollection(segs, cmap=plt.get_cmap(colourmap), linestyle=linestyle, linewidths=linewidth)
    lc.set_array(vcg.index.values)

    # Add the collection to the plot
    ax.add_collection(lc)
    # Line collections don't auto-scale the plot - set it up for a square plot
    tools.plotting.set_axis_limits(ax, data=vcg.loc[:, [x_plot, y_plot]], unit_min=False, axis_limits=axis_limits)

    # Change the positioning of the axes
    # Move left y-axis and bottom x-axis to centre, passing through (0,0)
    ax.spines['left'].set_position('zero')
    ax.spines['bottom'].set_position('zero')

    # Eliminate upper and right axes
    ax.spines['right'].set_color('none')
    ax.spines['top'].set_color('none')

    # Show ticks in the left and lower axes only
    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')

    # Move the labels to the edges of the plot
    ax.set_xlabel('VCG ('+x_plot+')')
    ax.set_ylabel('VCG ('+y_plot+')', rotation='horizontal')
    ax.xaxis.set_label_coords(1.05, 0.5)
    ax.yaxis.set_label_coords(0.5, 1.02)

    t_start, t_end = vcg.index[0], vcg.index[-1]
    tools.plotting.add_colourbar(limits=[t_start, t_end], fig=fig, colourmap=colourmap, n_elements=vcg.shape[0])

    return fig


def plot_3d(vcg: pd.DataFrame,
            linestyle: str = '-',
            colourmap: str = 'viridis',
            linewidth: float = 3.0,
            axis_limits: Optional[Union[List[float], float]] = None,
            unit_min: bool = True,
            sig_fig: int = None,
            fig: Optional[plt.figure] = None) -> plt.figure:
    """
    Plot the evolution of VCG in 3D space

    Parameters
    ----------
    vcg : pd.DataFrame
        VCG data
    linestyle : str, optional
        Linestyle to plot data, default='-'
    colourmap : str, optional
        Colourmap to use when plotting data, default='viridis'
    linewidth : float, optional
        Linewidth to use, default=3
    axis_limits : list of float or float, optional
        Limits to apply to the axes, default=None
    unit_min : bool, optional
        Whether to have the axes set to, as a minimum, unit length, default=True
    sig_fig : int, optional
        Maximum number of decimal places to be used on the axis plots (e.g., if set to 2, 0.12345 will be displayed
        as 0.12). Used to avoid floating point errors, default=None (no adaption made)
    fig : plt.figure, optional
        Handle to existing figure (if exists)

    Returns
    -------
    fig : plt.figure
        Figure handle
    """
    # Prepare line segments for plotting
    t = np.linspace(0, 1, vcg.shape[0])  # "time" variable
    points = vcg.values.reshape(-1, 1, 3)
    segs = np.concatenate([points[:-1], points[1:]], axis=1)
    lc = Line3DCollection(segs, cmap=plt.get_cmap(colourmap), linestyle=linestyle, linewidths=linewidth)
    lc.set_array(t)

    if fig is None:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        # fig, ax = plt.subplots(1, 1, projection='3d')
    else:
        ax = fig.gca()

    # add the collection to the plot
    ax.add_collection3d(lc)

    t_start, t_end = vcg.index[0], vcg.index[-1]
    tools.plotting.add_colourbar(limits=[t_start, t_end], fig=fig, colourmap=colourmap, n_elements=vcg.shape[0])

    ax.set_xlabel('VCG (x)')
    ax.set_ylabel('VCG (y)')
    ax.set_zlabel('VCG (z)')

    # Set axis limits (not automatic for line collections)
    tools.plotting.set_axis_limits(ax, data=vcg, axis_limits=axis_limits, unit_min=unit_min)
    tools.plotting.add_xyz_axes(fig, ax, sig_fig=sig_fig)

    return fig


def animate_3d(vcg: np.ndarray,
               limits: Optional[Union[float, List[float], List[List[float]]]] = None,
               linestyle: Optional[str] = '-',
               colourmap: Optional[str] = 'viridis',
               linewidth: Optional[float] = 3,
               output_file: Optional[str] = 'vcg_xyz.mp4') -> None:
    """
    Animate the evolution of the VCG in 3D space, saving that animation to a file.

    Parameters
    ----------
    vcg : np.ndarray
        VCG data
    limits : float or list of float or list of list of floats, optional
        Limits for the axes. If none, will set to the min/max values of the provided data. Can provide either as:
         1) a single value (+/- of that value applied to all axes)
         2) [min, max] to be applied to all axes
         3) [[xmin, xmax], [ymin, ymax], [zmin, zmax]]
    linestyle : str, optional
        Linestyle for the data, default='-'
    colourmap : str, optional
        Colourmap to use when plotting, default='viridis'
    linewidth : float, optional
        Linewidth when used to plot VCG, default=3
    output_file : str, optional
        Name of the file to save the animation to, default='vcg_xyz.mp4'
    """

    from matplotlib import animation

    # Extract limits
    if limits is None:
        max_lim = vcg.max().max()
        min_lim = vcg.min().min()
        limits = [min_lim, max_lim]

    # Process inputs to ensure the correct formats are used.
    if linestyle is None:
        linestyle = '-'

    # Set up figure and axes
    fig = plt.figure()
    ax = Axes3D(fig)
    tools.plotting.add_xyz_axes(ax, axis_limits=limits, symmetrical_axes=False, equal_limits=False, unit_axes=False)
    line, = ax.plot([], [], lw=3)

    # initialization function: plot the background of each frame
    def init():
        line.set_data([], [])
        return line,

    # animation function.  This is called sequentially
    def animate(i):
        # Prepare line segments for plotting
        t = np.linspace(0, 1, vcg['x'][:i].shape[0])  # "time" variable
        points = np.array([vcg['x'][:i], vcg['y'][:i], vcg['z'][:i]]).transpose().reshape(-1, 1, 3)
        segs = np.concatenate([points[:-1], points[1:]], axis=1)
        lc = Line3DCollection(segs, cmap=plt.get_cmap(colourmap), linestyle=linestyle, linewidths=linewidth)
        lc.set_array(t)
        ax.add_collection3d(lc)
        return line,

    # call the animator.  blit=True means only re-draw the parts that have changed.
    anim = animation.FuncAnimation(fig, animate, init_func=init, frames=vcg.shape[0], interval=30, blit=True)

    # save the animation as an mp4.  This requires ffmpeg or mencoder to be installed.  The extra_args ensure that the
    # x264 codec is used, so that the video can be embedded in html5.  You may need to adjust this for your system: for
    # more information, see http://matplotlib.sourceforge.net/api/animation_api.html
    anim.save(output_file, fps=30, extra_args=['-vcodec', 'libx264'])

    plt.show()
    return None


def plot_xyz_vector(vector: Optional[List[float]] = None,
                    x: float = None,
                    y: float = None,
                    z: float = None,
                    fig: plt.figure = None,
                    linecolour: str = 'C0',
                    linestyle: str = '-',
                    linewidth: float = 2):
    """ Plots a specific vector in 3D space (e.g. to reflect maximum dipole)

    Parameters
    ----------
    vector : list of float
        [x, y, z] values of vector to plot, alternatively given as separate x, y, z variables
    x, y, z : float
        [x, y, z] values of vector to plot, alternatively given as vector variable
    fig : plt.figure, optional
        Existing figure handle, if desired to plot the vector onto an extant plot
    linecolour : str, optional
        Colour to plot the vector as
    linestyle : str, optional
        Linestyle to use to plot the body of the arrow
    linewidth : float, optional
        Width to plot the body of the arrow

    Returns
    -------
    fig : plt.figure
        Figure handle

    Raises
    ------
    ValueError
        Exactly one of vertices and x,y,z must be given

    Notes
    -----
    Must provide either vector or [x,y,z]
    """
    # draw a vector
    from matplotlib.patches import FancyArrowPatch
    from mpl_toolkits.mplot3d import proj3d

    if (vector is None) == (x is None):
        raise ValueError("Exactly one of vertices and x,y,z must be given")
    if vector is not None:
        x = vector[0]
        y = vector[1]
        z = vector[2]

    class Arrow3D(FancyArrowPatch):

        def __init__(self, xs, ys, zs, *args, **kwargs):
            FancyArrowPatch.__init__(self, (0, 0), (0, 0), *args, **kwargs)
            self._verts3d = xs, ys, zs

        def draw(self, renderer):
            xs3d, ys3d, zs3d = self._verts3d
            xs, ys, zs = proj3d.proj_transform(xs3d, ys3d, zs3d, renderer.M)
            self.set_positions((xs[0], ys[0]), (xs[1], ys[1]))
            FancyArrowPatch.draw(self, renderer)

    if fig is None:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
    else:
        ax = fig.gca()

    """ Adapt linestyle variable if required
    (see http://matplotlib.1069221.n5.nabble.com/linestyle-option-for-FancyArrowPatch-and-similar-commands-td39913.html)
    """
    if linestyle == '--':
        linestyle = 'dashed'
    elif linestyle == ':':
        linestyle = 'dotted'
    elif linestyle == '-.':
        linestyle = 'dashdot'
    elif linestyle == '-':
        linestyle = 'solid'
    else:
        warnings.warn('Unrecognised value for linestyle variable...')

    a = Arrow3D([0, x], [0, y], [0, z], mutation_scale=20, lw=linewidth, arrowstyle="-|>", color=linecolour,
                linestyle=linestyle)
    ax.add_artist(a)

    ax.set_aspect('equal', adjustable='box')
    ax.set_xlabel('VCG (x)')
    ax.set_ylabel('VCG (y)')
    ax.set_zlabel('VCG (z)')

    return fig


def add_unit_sphere(ax) -> None:
    """ Add a unit sphere to a 3D plot

    Parameters
    ----------
    ax
        Handles to axes
    """
    u, v = np.mgrid[0:2 * np.pi:20j, 0:np.pi:10j]
    x = np.cos(u) * np.sin(v)
    y = np.sin(u) * np.sin(v)
    z = np.cos(v)
    ax.plot_wireframe(x, y, z, color="k", linewidth=0.5, alpha=0.25)
    return None


def plot_arc3d(vector1: List[float],
               vector2: List[float],
               radius: float = 0.2,
               fig: plt.figure = None,
               colour: str = 'C0') -> plt.figure:
    """ Plot arc between two given vectors in 3D space.

    Parameters
    ----------
    vector1 : list of float
        First vector
    vector2 : list of float
        Second vector
    radius : float, optional
        Radius of arc to plot on figure
    fig : plt.figure, optional
        Handle of figure on which to plot the arc. If not given, will produce new figure
    colour : str, optional
        Colour in which to display the arc

    Returns
    -------
    fig : plt.figure
        Handle for figure on which arc has been plotted
    """

    """ Confirm correct input arguments """
    assert len(vector1) == 3
    assert len(vector2) == 3

    """ Calculate vector between two vector end points, and the resulting spherical angles for various points along 
        this vector. From this, derive points that lie along the arc between vector1 and vector2 """
    v = [i-j for i, j in zip(vector1, vector2)]
    v_points_direct = [(vector2[0]+v[0]*v_distance, vector2[1]+v[1]*v_distance, vector2[2]+v[2]*v_distance)
                       for v_distance in np.linspace(0, 1)]
    v_phis = [atan2(v_point[1], v_point[0]) for v_point in v_points_direct]
    v_thetas = [acos(v_point[2]/np.linalg.norm(v_point)) for v_point in v_points_direct]

    v_points_arc = [(radius*sin(theta)*cos(phi), radius*sin(theta)*sin(phi), radius*cos(theta))
                    for theta, phi in zip(v_thetas, v_phis)]
    v_points_arc.append((0, 0, 0))

    if fig is None:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
    else:
        ax = fig.gca()

    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    """ Plot polygon (face colour must be set afterwards, otherwise it over-rides the transparency)
        https://stackoverflow.com/questions/18897786/transparency-for-poly3dcollection-plot-in-matplotlib """
    points_collection = Poly3DCollection([v_points_arc], alpha=0.4)
    points_collection.set_facecolor(colour)
    ax.add_collection3d(points_collection)

    return fig


def plot_spatial_velocity(vcg: Union[pd.DataFrame, List[pd.DataFrame]],
                          sv: Optional[List[List[float]]] = None,
                          limits: Optional[List[List[float]]] = None,
                          fig: plt.figure = None,
                          legend_vcg: Union[List[str], str, None] = None,
                          legend_limits: Union[List[str], str, None] = None,
                          limits_linestyles: Optional[List[str]] = None,
                          limits_colours: Optional[List[str]] = None,
                          filter_sv: bool = True) -> Tuple:
    """ Plot the spatial velocity for given VCG data

    Plot the spatial velocity and VCG elements, with limits (e.g. QRS limits) if provided. Note that if spatial
    velocity is not provided, default values will be used to calculate it - if anything else is desired, then spatial
    velocity must be calculated first and provided to the function.

    Parameters
    ----------
    vcg : pd.DataFrame or list of pd.DataFrame
        VCG data
    sv : list of list of float, optional
        Spatial velocity data. Only required to be given here if special parameters wish to be given, otherwise it
        will be calculated using default parameters (default)
    limits : list of list of float, optional
        A series of 'limits' to be plotted on the figure with the VCG and spatial plot. Presented as a list of the
        same length of the VCG data, with the required limits within:
            e.g. [[QRS_start1, QRS_start2, ...], [QRS_end1, QRS_end2, ...], ...]
        Default=None
    fig : plt.figure, optional
        Handle to existing figure, if data is wished to be plotted on existing plot, default=None
    legend_vcg : str or list of str, optional
        Labels to apply to the VCG/SV data, default=None
    legend_limits : str or list of str, optional
        Labels to apply to the limits, default=None
    limits_linestyles : list of str, optional
        Linestyles to apply to the different limits being supplied, default=None (will use varying linestyles based
        on tools.plotting.get_plot_lines)
    limits_colours : list of str, optional
        Colours to apply to the different limits being supplied, default=None (will use varying colours based on
        tools.plotting.get_plot_colours)
    filter_sv : bool, optional
        Whether or not to apply filtering to spatial velocity prior to finding the start/end points for the
        threshold, default=True

    Returns
    -------
    fig, ax
        Handles to the figure and axes generated
    """

    # Check input arguments are correctly formatted
    if isinstance(vcg, pd.DataFrame):
        vcg = [vcg]
    n_vcg = len(vcg)
    vcg = tools.python.convert_input_to_list(vcg, n_list=n_vcg)
    limits = tools.python.convert_input_to_list(limits, n_list=n_vcg, list_depth=2)
    legend_vcg = tools.python.convert_input_to_list(legend_vcg, n_list=n_vcg)
    legend_limits = tools.python.convert_input_to_list(legend_limits, n_list=len(limits))
    limits_linestyles = tools.python.convert_input_to_list(limits_linestyles, n_list=len(limits),
                                                           default_entry='line')
    limits_colours = tools.python.convert_input_to_list(limits_colours, n_list=len(limits), default_entry='colour')

    # Prepare figures and aces
    if fig is None:
        fig = plt.figure()
        ax = dict()
        gs = gridspec.GridSpec(3, 3)
        ax['sv'] = fig.add_subplot(gs[:, :-1])
        ax['x'] = fig.add_subplot(gs[0, -1])
        ax['y'] = fig.add_subplot(gs[1, -1])
        ax['z'] = fig.add_subplot(gs[2, -1])
        plt.setp(ax['x'].get_xticklabels(), visible=False)
        plt.setp(ax['y'].get_xticklabels(), visible=False)
        gs.update(hspace=0.05)
        # colours = tools.plotting.get_plot_colours(n_vcg)
    else:
        ax = dict()
        ax_sv, ax_vcg_x, ax_vcg_y, ax_vcg_z = fig.get_axes()
        ax['sv'] = ax_sv
        ax['x'] = ax_vcg_x
        ax['y'] = ax_vcg_y
        ax['z'] = ax_vcg_z
        colours = tools.plotting.get_plot_colours(len(ax_sv.lines) + n_vcg)
        """ If too many lines already exist on the plot, need to recolour them all to prevent cross-talk """
        if len(ax_sv.lines) + n_vcg > 10:
            for key in ax:
                lines = ax[key].get_lines()
                i_vcg = 0
                for line in lines:
                    line.set_color(colours[i_vcg])
                    i_vcg += 1

    """ Add labels to axes """
    ax['sv'].set_xlabel('Time (ms)')
    ax['sv'].set_ylabel('Spatial velocity')
    ax['x'].set_ylabel('VCG (x)')
    ax['y'].set_ylabel('VCG (y)')
    ax['z'].set_ylabel('VCG (z)')
    ax['z'].set_xlabel('Time (ms)')

    if sv is None:
        sv = signalanalysis.vcg.get_spatial_velocity(vcgs=vcg, filter_sv=filter_sv)

    """ Plot spatial velocity and VCG components"""
    i_colour_init = tools.python.get_i_colour(ax['sv'])
    i_colour = i_colour_init
    h_lines = list()
    for (sim_sv, sim_vcg, sim_label) in zip(sv, vcg, legend_vcg):
        for lead in ['x', 'y', 'z']:
            ax[lead].plot(sim_vcg.index, sim_vcg[lead])
        h_lines.append(ax['sv'].plot(sim_sv.index, sim_sv))
        i_colour += 1

    """ Plot QRS limits, if provided """
    h_limits = list()
    if limits[0] is not None:
        print(limits)
        # Cycle through each limit provided, e.g. QRS start, QRS end...
        for (qrs_limit, limits_linestyle, limits_colour) in zip(limits, limits_linestyles, limits_colours):
            # i_colour = i_colour_init
            add_limit_handle = True

            # Plot limits for each given VCG
            for sim_qrs_limit in qrs_limit:
                for key in ax:
                    if key == 'sv' and add_limit_handle:
                        if not isinstance(sim_qrs_limit, float):
                            sim_qrs_limit = sim_qrs_limit.values
                        h_limits.append(ax[key].axvline(sim_qrs_limit, color=limits_colour, alpha=0.8,
                                                        linestyle=limits_linestyle, label=None))
                        add_limit_handle = False
                    else:
                        ax[key].axvline(sim_qrs_limit, color=limits_colour, alpha=0.8,
                                        linestyle=limits_linestyle, label=None)
                # i_colour += 1

    """ Add legend_vcg and legend_limits """
    if legend_vcg[0] is not None:
        labels = [line.get_label() for line in ax['sv'].get_lines()]
        labels = [labelstr for labelstr in labels if not labelstr.startswith('_')]  # Remove implicit labels from list
        plt.rc('font', family='sans-serif')
        plt.rc('text', usetex=True)
        leg_vcg = ax['sv'].legend(labels, loc='upper right')
    else:
        leg_vcg = None

    if legend_limits[0] is not None:
        plt.rc('font', family='sans-serif')
        plt.rc('text', usetex=True)
        ax['sv'].legend(h_limits, legend_limits, loc='upper center', handlelength=5)
        if legend_vcg[0] is not None:
            ax['sv'].add_artist(leg_vcg)

    # if legend_limits is not None:
    #     print()

    return fig, ax


def plot_metric_change(metrics: List[List[List[float]]],
                       metrics_phi: List[List[List[float]]],
                       metrics_rho: List[List[List[float]]],
                       metrics_z: List[List[List[float]]],
                       metric_name: str,
                       metrics_lv: List[bool] = None,
                       labels: List[str] = None,
                       scattermarkers: List[str] = None,
                       linemarkers: List[str] = None,
                       colours: List[str] = None,
                       linestyles: List[str] = None,
                       layout: str = None,
                       axis_match: bool = True,
                       no_labels: bool = False) -> Tuple:
    """ Function to plot all the various figures for trend analysis in one go.

    TODO: labels parameter seems redundant - potentially remove

    Parameters
    ----------
    metrics : list of list of list of float
        Complete list of all metric data recorded
        [phi+rho+z+size+other]
    metrics_phi : list of list of list of float
        Metric data recorded for scar size variations in phi UVC
    metrics_rho : list of list of list of float
        Metric data recorded for scar size variations in rho UVC
    metrics_z : list of list of list of float
        Metric data recorded for scar size variations in z UVC
    metric_name : str
        Name of metric being plotted (for labelling purposes). Can incorporate LaTeX typesetting.
    metrics_lv : list of bool, optional
        Boolean to distinguish whether metrics being plotted are for LV or septal data, default=[True, False]
    labels : list of str, optional
        Labels for the data sets being plotted, default=['LV', 'Septum']
    scattermarkers : list of str, optional
        Markers to use to plot the data on the scatterplots, default=['+', 'o', 'D', 'v', '^', 's', '*', 'x']
    linemarkers : list of str, optional
        Markers to use on the line plots to indicate discrete data points, required to be at least as long as
        the longest line plot to be drawn (rho), default=['.' for _ in range(len(metrics_rho))]
    colours : list of str, optional
        Sequence of colours to plot data (if plotting LV and septal data, will require two different colours to allow
        them to be distinguished), default=common_analysis.get_plot_colours(len(metrics_rho))
    linestyles : list of str, optional
        Linestyles to be used for plotting the data on lineplots, default=['-' for _ in range(len(metrics_rho))]
    layout : {'combined', 'figures'}, optional
        String specifying the output, whether all plots should be combined into one figure window (default), or whether
        individual figure windows should be plotted for each plot
    axis_match : bool, optional
        Whether to make sure all plotted figures share the same axis ranges, default=True
    no_labels : bool, optional
        Whether to have labels on the figures, or not - having no labels can make it far easier to 'prettify' the
        figures manually later in Inkscape, default=False

    Returns
    -------
    fig : plt.figure or dict of plt.figure
        Handle to figure(s)
    ax : dict
        Handles to axes
    """
    plt.rc('font', family='sans-serif')
    plt.rc('text', usetex=True)

    """ Underlying constants (volumes, sizes, labels, etc.) """
    # Create volume variables (nb: percent of whole mesh)
    vol_lv_phi = [0.0, 2.657, 8.667, 14.808]
    vol_lv_rho = [0.0, 3.602, 7.243, 10.964, 14.808]
    vol_lv_z = [0.0, 6.183, 10.897, 14.808]
    vol_lv_size = [0.0, 0.294, 4.062, 14.808]
    vol_lv_other = [5.333]

    vol_septum_phi = [0.0, 6.926, 11.771, 17.019, 21.139]
    vol_septum_rho = [0.0, 5.105, 10.275, 15.586, 21.139]
    vol_septum_z = [0.0, 8.840, 15.818, 21.139]
    vol_septum_size = [0.0, 0.672, 6.531, 21.139]
    vol_septum_other = [8.944]

    volume_lv = vol_lv_phi + vol_lv_rho + vol_lv_z + vol_lv_size + vol_lv_other
    volume_septum = vol_septum_phi + vol_septum_rho + vol_septum_z + vol_septum_size + vol_septum_other

    # Create area variables (in cm^2)
    area_lv_phi = [0.0, 37.365, 85.575, 129.895]
    area_lv_rho = [0.0, 109.697, 115.906, 122.457, 129.895]
    area_lv_z = [0.0, 57.847, 97.439, 129.895]
    area_lv_size = [0.0, 10.140, 57.898, 129.895]
    area_lv_other = [76.501]

    area_septum_phi = [0.0, 56.066, 88.603, 122.337, 149.588]
    area_septum_rho = [0.0, 126.344, 133.363, 141.091, 149.588]
    area_septum_z = [0.0, 72.398, 114.937, 149.588]
    area_septum_size = [0.0, 17.053, 72.104, 149.588]
    area_septum_other = [97.174]

    area_lv = area_lv_phi + area_lv_rho + area_lv_z + area_lv_size + area_lv_other
    area_septum = area_septum_phi + area_septum_rho + area_septum_z + area_septum_size + area_septum_other
    area_lv_norm = [i/area_septum_phi[-1] for i in area_lv]
    area_septum_norm = [i/area_septum_phi[-1] for i in area_septum]

    legend_phi_lv = ['None', r'$1.4\pi/2$ \textrightarrow $1.6\pi/2$', r'$1.2\pi/2$ \textrightarrow $1.8\pi/2$',
                     r'$\pi/2$ \textrightarrow $\pi$']
    legend_phi_septum = ['None', r'-0.25 \textrightarrow 0.25', r'-0.50 \textrightarrow 0.25',
                         r'-0.75 \textrightarrow 0.75', r'-1.00 \textrightarrow 1.00']
    legend_rho = ['None', r'0.4 \textrightarrow 0.6', r'0.3 \textrightarrow 0.7', r'0.2 \textrightarrow 0.8',
                  r'0.1 \textrightarrow 0.9']
    legend_z = ['None', r'0.5 \textrightarrow 0.7', r'0.4 \textrightarrow 0.8', r'0.3 \textrightarrow 0.9']

    """ Assert correct data has been passed (insofar that it is of the right length!) """
    metrics = tools.python.convert_input_to_list(metrics, n_list=-1, list_depth=2)
    metrics_phi = tools.python.convert_input_to_list(metrics_phi, n_list=-1, list_depth=2)
    metrics_rho = tools.python.convert_input_to_list(metrics_rho, n_list=-1, list_depth=2)
    metrics_z = tools.python.convert_input_to_list(metrics_z, n_list=-1, list_depth=2)

    if metrics_lv is None:
        metrics_lv = [True, False]
    if labels is None:
        labels = ['LV', 'Septum']
    volumes = [volume_lv if metric_lv else volume_septum for metric_lv in metrics_lv]
    for metric, volume in zip(metrics, volumes):
        assert len(metric) == len(volume)
    areas = [area_lv_norm if metric_lv else area_septum_norm for metric_lv in metrics_lv]
    for metric, area in zip(metrics, areas):
        assert len(metric) == len(area)

    if scattermarkers is None:
        scattermarkers = ['+', 'o', 'D', 'v', '^', 's', '*', 'x']
    assert len(scattermarkers) >= len(metrics)
    if linemarkers is None:
        linemarkers = ['.' for _ in range(len(metrics_rho))]
    else:
        assert len(linemarkers) >= len(metrics_rho)
    if linestyles is None:
        linestyles = ['-' for _ in range(len(metrics_rho))]
    else:
        assert len(linestyles) >= len(metrics_rho)
    if colours is None:
        colours = tools.plotting.get_plot_colours(len(metrics_rho))
    else:
        assert len(colours) >= len(metrics_rho)

    """ Set up figures and axes """
    keys = ['volume', 'area', 'phi_lv', 'phi_septum', 'rho', 'z']
    if (layout is None) or (layout == 'combined'):
        fig = plt.figure()
        fig.suptitle(metric_name)
        gs = gridspec.GridSpec(4, 3)
        ax = dict()
        ax['volume'] = fig.add_subplot(gs[:2, :2])
        ax['area'] = fig.add_subplot(gs[2:, :2])
        ax['phi_lv'] = fig.add_subplot(gs[0, 2])
        ax['phi_septum'] = fig.add_subplot(gs[1, 2])
        ax['rho'] = fig.add_subplot(gs[2, 2])
        ax['z'] = fig.add_subplot(gs[3, 2])
        # plt.setp(ax['y'].get_yticklabels(), visible=False)
        gs.update(left=0.05, right=0.95, bottom=0.05, top=0.95, wspace=0.09, hspace=0.25)
    elif layout == 'figures':
        fig = {key: plt.figure() for key in keys}
        ax = {key: fig[key].add_subplot(1, 1, 1) for key in keys}
        if not no_labels:
            for key in keys:
                ax[key].set_ylabel(metric_name)
    else:
        print("Unrecognised layout command.")
        return None, None

    """ Plot data on axes """
    # Volume
    for (metric, volume, label, colour, scattermarker) in zip(metrics, volumes, labels, colours, scattermarkers):
        ax['volume'].plot(volume, metric)
    # ax['volume'].plot(volume_lv, metric_lv, '+', label='LV', markersize=10, markeredgewidth=3, color='C0')
    # ax['volume'].plot(volume_septum, metric_septum, 'o', markersize=10, markeredgewidth=3, label='Septum',
    #                       markerfacecolor='none', color='C1')
    if no_labels:
        plt.setp(ax['volume'].get_xticklabels(), visible=False)
        plt.setp(ax['volume'].get_yticklabels(), visible=False)
        ax['volume'].set_title(r'Volume of scar (\% of mesh)')
    else:
        ax['volume'].legend()
        ax['volume'].set_xlabel(r'Volume of scar (\% of mesh)')

    # Area
    for (metric, area, label, colour, scattermarker) in zip(metrics, areas, labels, colours, scattermarkers):
        ax['area'].plot(area, metric)
    # ax['area'].plot(area_lv_norm, metric_lv, '+', label='LV', markersize=10, markeredgewidth=3, color='C0')
    # ax['area'].plot(area_septum_norm, metric_septum, 'o', markersize=10, markeredgewidth=3, label='Septum',
    #                 markerfacecolor='none', color='C1')
    if no_labels:
        plt.setp(ax['area'].get_xticklabels(), visible=False)
        plt.setp(ax['area'].get_yticklabels(), visible=False)
        ax['area'].set_title(r'Surface Area of scar (normalised)')
    else:
        ax['area'].set_xlabel(r'Surface Area of scar (normalised)')

    # Phi (LV)
    for (metric, label, colour, marker, linestyle, metric_lv) in zip(metrics_phi, labels, colours, linemarkers,
                                                                     linestyles, metrics_lv):
        if metric_lv:
            ax['phi_lv'].plot(metric, linewidths_ecg=3)
        else:
            ax['phi_septum'].plot(metric, linewidths_ecg=3)
    # ax['phi_lv'].plot(metric_phi_lv, 'o-', label='LV', linewidth=3, color='C0')
    ax['phi_lv'].set_xticks(list(range(len(legend_phi_lv))))
    if no_labels:
        plt.setp(ax['phi_lv'].get_xticklabels(), visible=False)
        plt.setp(ax['phi_lv'].get_yticklabels(), visible=False)
        ax['phi_lv'].set_title(r'$\phi$')
    else:
        ax['phi_lv'].set_xticklabels(legend_phi_lv)
        ax['phi_lv'].set_xlabel(r'$\phi$')

    # Phi (septum)
    # ax['phi_septum'].plot(metric_phi_septum, 'o-', label='Septum', linewidth=3, color='C1')
    ax['phi_septum'].set_xticks(list(range(len(legend_phi_septum))))
    if no_labels:
        plt.setp(ax['phi_septum'].get_xticklabels(), visible=False)
        plt.setp(ax['phi_septum'].get_yticklabels(), visible=False)
        ax['phi_septum'].set_title(r'$\phi$')
    else:
        ax['phi_septum'].set_xticklabels(legend_phi_septum)
        ax['phi_septum'].set_xlabel(r'$\phi$')

    # Rho
    for (metric, label, colour, marker, linestyle, metric_lv) in zip(metrics_rho, labels, colours, linemarkers,
                                                                     linestyles, metrics_lv):
        ax['rho'].plot(metric, linewidths_ecg=3)
    # ax['rho'].plot(metric_rho_lv, 'o-', label='LV', linewidth=3, color='C0')
    # ax['rho'].plot(metric_rho_septum, 'o-', label='Septum', linewidth=3, color='C1')
    ax['rho'].set_xticks(list(range(len(legend_rho))))
    if no_labels:
        plt.setp(ax['rho'].get_xticklabels(), visible=False)
        plt.setp(ax['rho'].get_yticklabels(), visible=False)
        ax['rho'].set_title(r'$\rho$')
    else:
        ax['rho'].set_xticklabels(legend_rho)
        ax['rho'].set_xlabel(r'$\rho$')

    # z
    for (metric, label, colour, marker, linestyle, metric_lv) in zip(metrics_z, labels, colours, linemarkers,
                                                                     linestyles, metrics_lv):
        ax['z'].plot(metric, linewidths_ecg=3)
    # ax['z'].plot(metric_z_lv, 'o-', label='LV', linewidth=3, color='C0')
    # ax['z'].plot(metric_z_septum, 'o-', label='Septum', linewidth=3, color='C1')
    ax['z'].set_xticks(list(range(len(legend_z))))
    if no_labels:
        plt.setp(ax['z'].get_xticklabels(), visible=False)
        plt.setp(ax['z'].get_yticklabels(), visible=False)
        ax['z'].set_title(r'$z$')
    else:
        ax['z'].set_xticklabels(legend_z)
        ax['z'].set_xlabel(r'$z$')

    if axis_match:
        ax_limits = ax['volume'].get_ylim()
        for key in keys:
            ax[key].set_ylim(ax_limits)

    return fig, ax


def plot_metric_change_barplot(metrics_cont: List[List[float]],
                               metrics_lv: List[List[float]],
                               metrics_sept: List[List[float]],
                               metric_labels: List[str],
                               layout: str = None) -> Tuple:
    """ Plots a bar chart for the observed metrics.

    Parameters
    ----------
    metrics_cont : list of list of float
        Values of series of metrics for no scar
    metrics_lv : list of list of float
        Values of series of metrics for LV scar
    metrics_sept : list of list of float
        Values of series of metrics for septal scar
    metric_labels : list of str
        Names of metrics being plotted
    layout : {'combined', 'fig'}, optional
        Whether to plot bar charts on combined plot window, or in individual figure windows

    Returns
    -------
    fig : plt.figure or list of plt.figure
        Handle(s) to figures
    ax: list
        Handles to axes
    """

    """ Conduct initial checks, and set up values appropriate to plotting """
    assert len(metrics_cont) == len(metrics_lv)
    assert len(metrics_cont) == len(metrics_sept)
    assert len(metrics_cont) == len(metric_labels)

    if layout is None:
        layout = 'combined'

    if layout == 'combined':
        fig = plt.figure()
        gs = gridspec.GridSpec(1, len(metrics_cont))
        axes = list()
        for i, metric_label in enumerate(metric_labels):
            axes.append(fig.add_subplot(gs[i]))
    elif layout == 'fig':
        fig = [plt.figure() for _ in range(len(metrics_cont))]
        axes = [fig_i.add_subplot(1, 1, 1) for fig_i in fig]
    else:
        print("Invalid argument given for layout")
        return None, None
    # fig, ax = plt.subplots()

    # index = np.arange(len(metrics_cont))
    index = [0, 1, 2]
    bar_width = 1.2
    opacity = 0.8

    for ax, metric_cont, metric_lv, metric_sept, label in zip(axes, metrics_cont, metrics_lv, metrics_sept,
                                                              metric_labels):
        ax.bar(index[0], metric_cont, label='Control', alpha=opacity, color='C2', width=bar_width)
        ax.bar(index[1], metric_lv, label='LV Scar', alpha=opacity, color='C0', width=bar_width)
        ax.bar(index[2], metric_sept, label='Septum Scar', alpha=opacity, color='C1', width=bar_width)
        ax.set_title(label)
        ax.set_xticks([])
    axes[-1].legend()

    # """ Plot bar charts """
    # plt.bar(index-bar_width, metrics_cont, bar_width, label='Control')
    # plt.bar(index, metrics_lv, bar_width, label='LV Scar')
    # plt.bar(index+bar_width, metrics_sept, label='Septum Scar')
    #
    # """ Add labels """
    # ax.set_ylabel('Fractional Change')
    # ax.legend()
    # ax.set_xticklabels(index, metric_labels)

    return fig, axes


def plot_density_effect(metrics: List[List[float]],
                        metric_name: str,
                        metric_labels: List[str] = None,
                        density_labels: List[str] = None,
                        linestyles: List[str] = None,
                        colours: List[str] = None,
                        markers: List[str] = None):
    """ Plot the effect of density on metrics.

    TODO: look into decorator for the LaTeX preamble?

    Parameters
    ----------
    metrics : list of list of float
        Effects of scar density on given metrics, presented as e.g. [metric_LV, metric_septum]
    metric_name : str
        Name of metric being assessed
    metric_labels : list of str, optional
        Labels for the metrics being plotted, default=['LV', 'Septum']
    density_labels : list of str, optional
        Labels for the different scar densities being plotted
    linestyles : list of str, optional
        Linestyles for the density effect plots, default=['-' for _ in range(len(metrics))]
    colours : list of str, optional
        Colours to use for the plot, default=common_analysis.get_plot_colours(len(metrics))
    markers : list of str, optional
        Markers to use for the discrete data points in the plot, default=['o' for _ in range(len(metrics))]
    """
    preamble = {
        'font.family': 'sans-serif',
        'text.usetex': True,
        'text.latex.preamble': [r'\usepackage{amsmath}']
    }
    plt.rcParams.update(preamble)
    # plt.rc('text', usetex=True)

    """ Process input arguments. """
    if not isinstance(metrics[0], list):
        metrics = [metrics]
    if metric_labels is None:
        if len(metrics) == 2:
            warnings.warn("Assuming metrics passed in order [LV, Septum].")
            metric_labels = ['LV', 'Septum']
        else:
            metric_labels = [None for _ in range(len(metrics))]
    else:
        assert len(metrics) == len(metric_labels)
    if linestyles is None:
        linestyles = ['-' for _ in range(len(metrics))]
    else:
        assert len(metrics) == len(linestyles)
    if colours is None:
        colours = tools.plotting.get_plot_colours(len(metrics))
    else:
        assert len(metrics) == len(colours)
    if markers is None:
        markers = ['o' for _ in range(len(metrics))]
    else:
        assert len(metrics) == len(markers)

    fig = plt.figure()
    ax = fig.add_subplot(111)
    for (metric, label, linestyle, colour, marker) in zip(metrics, metric_labels, linestyles, colours, markers):
        ax.plot(metric, linewidths_ecg=3)

    if density_labels is None:
        density_labels = ['None',
                          r'\begingroup\addtolength{\jot}{-2mm}\begin{align*}'
                          r'p_\mathrm{low}&=0.2\\'
                          r'p_\mathrm{BZ}&=0.25\\'
                          r'p_\mathrm{dense}&=0.3'
                          r'\end{align*}\endgroup',
                          r'\begingroup\addtolength{\jot}{-2mm}\begin{align*}'
                          r'p_\mathrm{low}&=0.4\\'
                          r'p_\mathrm{BZ}&=0.5\\'
                          r'p_\mathrm{dense}&=0.6'
                          r'\end{align*}\endgroup',
                          r'\begingroup\addtolength{\jot}{-2mm}\begin{align*}'
                          r'p_\mathrm{low}&=0.6\\'
                          r'p_\mathrm{BZ}&=0.75\\'
                          r'p_\mathrm{dense}&=0.9'
                          r'\end{align*}\endgroup']

    """ Set axis labels and ticks """
    ax.set_ylabel(metric_name)
    ax.set_xticks(list(range(len(density_labels))))
    ax.set_xticklabels(density_labels)
    ax.legend()

    return fig

