import math
import numpy as np
import pandas as pd
import scipy.signal
import matplotlib.pyplot as plt
from typing import Optional

import signalanalysis.general
import tools.plotting


class Egm(signalanalysis.general.Signal):
    """Base class for EGM data, inheriting from :class:`signalanalysis.general.Signal`

    See Also
    --------
    :py:class:`signalanalysis.general.Signal`

    Methods
    -------
    read(folder)
        Extract data from unipolar and bipolar DxL files
    get_n_beats()
        Supersedes generalised method to calculate n_beats
    get_at
        Calculates the activation time of the EGM
    """

    def __init__(self,
                 data_location_uni: str,
                 data_location_bi: str = None,
                 **kwargs):
        """Sub-method for __init___

        Will initialise a EGM signal class

        TODO: Fix the self.data reference problem (see
        https://stackoverflow.com/questions/6057130/python-deleting-a-class-attribute-in-a-subclass)

        See Also
        --------
        :py:meth:`signalanalysis.general.Signal.__init__ : Base __init__ method
        :py:meth:`signalanalysis.general.Signal.apply_filter` : Filtering method
        :py:meth:`signalanalysis.general.Signal.get_n_beats` : Beat calculation method

        Notes
        -----
        This used to break the `Liskov substitution principle
        <https://en.wikipedia.org/wiki/Liskov_substitution_principle>`_, removing the single `data` attribute to be
        replaced by `data_uni` and `data_bi`, but now instead (aims to) just point the `.data` attribute to the
        `.data_uni` attibute
        """
        super(Egm, self).__init__(**kwargs)

        self.t_peaks = pd.DataFrame(dtype=float)
        self.n_beats = pd.Series(dtype=int)

        # delattr(self, 'data')
        self.data_uni = pd.DataFrame(dtype=float)
        self.data = self.data_uni
        self.data_bi = pd.DataFrame(dtype=float)

        self.beats_uni = dict()
        self.beats = self.beats_uni
        self.beats_bi = dict()

        self.at = pd.DataFrame(dtype=float)

        self.read(data_location_uni, data_location_bi, **kwargs)
        if self.filter is not None:
            self.apply_filter(**kwargs)
        # self.get_beats(**kwargs)

    def read(self,
             data_location_uni: str,
             data_location_bi: str,
             drop_empty_rows: bool = True,
             **kwargs):
        """ Read the DxL data for unipolar and bipolar data for EGMs

        TODO: Add functionality to read directly from folders, rather than .csv from Matlab

        Parameters
        ----------
        data_location_uni : str
            Location of unipolar data. Currently only coded to deal with a saved .csv file
        data_location_bi : str
            Location of bipolar data. Currently only coded to deal with a saved .csv file
        drop_empty_rows : bool
            Whether to drop empty data rows from the data, default=True

        See Also
        --------
        :py:meth:`signalanalysis.egm.Egm.read_from_csv` : Method to read data from Matlab csv
        """

        if data_location_uni.endswith('.csv'):
            assert data_location_bi.endswith('.csv')
            self.read_from_csv(data_location_uni, data_location_bi, **kwargs)
        else:
            raise IOError("Not coded for this type of input")

        if drop_empty_rows:
            self.data_uni = self.data_uni.loc[:, ~(self.data_uni == 0).all()]
            self.data_bi = self.data_bi.loc[:, ~(self.data_bi == 0).all()]
            assert self.data_uni.shape == self.data_bi.shape, "Error in dropping rows"

    def read_from_csv(self,
                      data_location_uni: str,
                      data_location_bi: str,
                      frequency: float):
        """ Read EGM data that has been saved from Matlab

        Parameters
        ----------
        data_location_uni : str
            Name of the .csv file containing the unipolar data
        data_location_bi : str
            Name of the .csv file containing the bipolar data
        frequency : float
            The frequency of the data recording in Hz

        Notes
        -----
        The .csv file should be saved with column representing an individual EGM trace, and each row representing a
        single instance in time, i.e.

        .. code-block::
            egm1(t1), egm2(t1), egm3(t1), ...
            egm1(t2), egm2(t2), egm3(t2), ...
            ...
            egm1(tn), egm2(tn), egm3(tn)

        Historically, `frequency` has been set to 2034.5 Hz for the importprecision data, an example of which is
        saved in ``tests/egm_unipolar.csv`` and ``tests/egm_bipolar.csv``.
        """

        self.data_uni = pd.read_csv(data_location_uni, header=None)

        interval = (1 / frequency)*1000
        end_val = self.data_uni.shape[0] * interval
        t = np.arange(0, end_val, interval)
        self.data_uni.set_index(t, inplace=True)
        if data_location_bi is not None:
            self.data_bi = pd.read_csv(data_location_bi, header=None)
            self.data_bi.set_index(t, inplace=True)
            self.data_source = [data_location_uni, data_location_bi]
        else:
            self.data_bi = None
            self.data_source = data_location_uni

    def get_peaks(self,
                  threshold: float = 0.33,
                  min_separation: float = 200,
                  plot: bool = False,
                  **kwargs):
        """ Supermethod for get_peaks for EGM data, using the squared bipolar signal rather than RMS data

        See also
        --------
        :py:meth:`signalanalysis.egm.Egm.plot_signal` : Method to plot the calculated AT
        """
        if self.data_bi.empty:
            super(Egm, self).get_peaks()
            return

        egm_bi_square = np.square(self.data_bi)

        i_separation = np.where(self.data_uni.index > min_separation)[0][0]
        self.n_beats = pd.Series(dtype=int, index=self.data_uni.columns)
        self.t_peaks = pd.DataFrame(dtype=float, columns=self.data_uni.columns)
        self.n_beats_threshold = threshold
        for i_signal in egm_bi_square:
            i_peaks, _ = scipy.signal.find_peaks(egm_bi_square.loc[:, i_signal],
                                                 height=threshold*egm_bi_square.loc[:, i_signal].max(),
                                                 distance=i_separation)
            self.n_beats[i_signal] = len(i_peaks)
            if len(i_peaks) == self.t_peaks.shape[0]:
                self.t_peaks[i_signal] = self.data_bi.index[i_peaks]
            elif len(i_peaks) < self.t_peaks.shape[0]:
                self.t_peaks[i_signal] = np.pad(self.data_bi.index[i_peaks],
                                                (0, self.t_peaks.shape[0]-len(i_peaks)),
                                                constant_values=np.nan)
            elif len(i_peaks) > self.t_peaks.shape[0]:
                self.t_peaks = self.t_peaks.reindex(range(len(i_peaks)), fill_value=np.nan)
                self.t_peaks[i_signal] = self.data_bi.index[i_peaks]

        if plot:
            _ = self.plot_signal(plot_peaks=True, plot_bipolar_square=True, **kwargs)

    def plot_signal(self,
                    i_plot: Optional[int] = None,
                    plot_bipolar_square: bool = False,
                    plot_peaks: bool = False,
                    plot_at: bool = False):
        """General use function to plot the unipolar and bipolar signals, and maybe EGM

        Will plot the unipolar and bipolar EGM signals on the same figure, and if requested, also the RMS trace at
        the same time.

        TODO: Move this functionality to a general plotting routine, rather than an internal module for a single object

        Parameters
        ----------
        i_plot : int, optional
            Which signal from the data to plot, default=random
        plot_bipolar_square : bool, optional
            Whether to plot the squared bipolar data on the figure as well, default=False
        plot_peaks, plot_at : bool, optional
            Whether to plot the points of the bipolar peak/AT on the figure, default=False

        Returns
        -------
        fig, ax
            Handles to the figure and axis data, respectively
        """

        # Pick a random signal to plot as an example trace (making sure to not pick a 'dead' trace)
        if i_plot is None:
            i_plot = self.n_beats.sample().index[0]
            while self.n_beats[i_plot] == 0:
                i_plot = self.n_beats.sample().index[0]
        else:
            if self.n_beats[i_plot] == 0:
                raise IOError("No beats detected in specified trace")

        fig = plt.figure()
        fig.suptitle('Trace {}'.format(i_plot))
        ax = dict()
        ax_labels = ['Unipolar', 'Bipolar']
        plot_data = [self.data_uni, self.data_bi]
        if plot_bipolar_square:
            ax_labels.append('Bipolar^2')
            plot_data.append(np.square(self.data_bi))

        for i_ax, data in enumerate(plot_data):
            ax[ax_labels[i_ax]] = fig.add_subplot(len(plot_data), 1, i_ax + 1)
            ax[ax_labels[i_ax]].plot(data.loc[:, i_plot], color='C0')
            ax[ax_labels[i_ax]].set_ylabel(ax_labels[i_ax])

            if plot_peaks:
                ax[ax_labels[i_ax]].scatter(self.t_peaks[i_plot].dropna(),
                                            data.loc[:, i_plot][self.t_peaks[i_plot].dropna()],
                                            label='Peaks',
                                            marker='o', edgecolor='tab:orange', facecolor='none', linewidths=2)
                if ax_labels[i_ax] == 'Bipolar^2':
                    ax[ax_labels[i_ax]].axhline(self.n_beats_threshold*data.loc[:, i_plot].max(),
                                                color='tab:orange', linestyle='--')

            if plot_at:
                ax[ax_labels[i_ax]].scatter(self.at[i_plot].dropna(),
                                            data.loc[:, i_plot][self.at[i_plot].dropna()],
                                            label='AT',
                                            marker='d', edgecolor='tab:green', facecolor='none', linewidths=2)

        # Add legend to top axis
        ax[ax_labels[0]].legend()

        return fig, ax

    def plot_peaks(self,
                   i_plot: Optional[int] = None,
                   **kwargs):
        if self.t_peaks.empty:
            self.get_peaks(**kwargs, plot=False)

        fig, ax = self.plot_signal(i_plot=i_plot, plot_bipolar_square=True)

        for key in ax:
            ydata = pd.Series(ax[key].get_lines()[0].get_ydata(), index=self.data_uni.index)
            ax[key].scatter(
                self.t_peaks[i_plot].dropna(),
                ydata[self.t_peaks[i_plot].dropna()],
                marker='o', edgecolor='tab:orange', facecolor='none', linewidths=2)
        return fig, ax

    def get_beats(self,
                  reset_index: bool = True,
                  offset_start: Optional[float] = None,
                  offset_end: Optional[float] = None,
                  plot: bool = False,
                  **kwargs):
        """ Detects beats in individual EGM signals

        TODO: Replace this with method based on finding AT and RT, then adding buffer round those values

        Supermethod for EGM beat detection, due to the fact that EGM beats are detected on a per signal basis
        rather than  a universal basis (RMS method)

        See also
        --------
        :py:meth:`signalanalysis.general.Signal.get_beats` : Base method
        """
        if self.t_peaks.empty:
            self.get_peaks(**kwargs)

        self.beat_start = pd.Series(dtype=pd.DataFrame, index=self.data_uni.columns)
        self.beats_uni = dict.fromkeys(self.data_uni.columns)
        self.beats_bi = dict.fromkeys(self.data_uni.columns)
        for key in self.data_uni:
            # If only one beat is detected, can end here
            if self.n_beats[key] == 1:
                self.beats_uni[key] = [self.data_uni.loc[:, key]]
                self.beats_bi[key] = [self.data_bi.loc[:, key]]
                continue

            # Calculate series of cycle length values, before then using this to estimate the start and end times of
            # each beat. The offset from the previous peak will be assumed at 0.4*BCL, while the offset from the
            # following peak will be 0.1*BCL (both with a minimum value of 30ms)
            if offset_start is None:
                bcls = np.diff(self.t_peaks[key])
                offset_start_list = [max(0.6 * bcl, 30) for bcl in bcls]
            else:
                offset_start_list = [offset_start] * self.n_beats[key]
            if offset_end is None:
                bcls = np.diff(self.t_peaks[key])
                offset_end_list = [max(0.1 * bcl, 30) for bcl in bcls]
            else:
                offset_end_list = [offset_end] * self.n_beats[key]
            self.beat_start.append([self.data_uni.index[0]] + list(self.t_peaks[key][:-1] + offset_start_list))
            beat_end = [t_p - offset for t_p, offset in zip(self.t_peaks[key][1:], offset_end_list)] + \
                       [self.data_uni.index[-1]]

            signal_beats_uni = list()
            signal_beats_bi = list()
            for t_s, t_p, t_e in zip(self.beat_start[-1], self.t_peaks[key], beat_end):
                assert t_s < t_p < t_e, "Error in windowing process"
                signal_beats_uni.append(self.data_uni.loc[t_s:t_e, :])
                signal_beats_bi.append(self.data_bi.loc[t_s:t_e, :])

            self.beat_index_reset = reset_index
            if reset_index:
                for i_beat in range(self.n_beats[key]):
                    zeroed_index = signal_beats_uni[i_beat].index - signal_beats_uni[i_beat].index[0]
                    signal_beats_uni[i_beat].set_index(zeroed_index, inplace=True)
                    signal_beats_bi[i_beat].set_index(zeroed_index, inplace=True)
            self.beats_uni[key] = signal_beats_uni
            self.beats_bi[key] = signal_beats_bi

        if plot:
            _ = self.plot_beats(offset_end=offset_end, **kwargs)

    def plot_beats(self,
                   offset_end: Optional[float] = None,
                   i_plot: Optional[int] = None,
                   **kwargs):
        # Calculate beats (if not done already)
        if self.beats_uni is None:
            self.get_beats(offset_end=offset_end, plot=False, **kwargs)

        # Pick a random signal to plot as an example trace (making sure to not pick a 'dead' trace)
        if i_plot is None:
            import random
            i_plot = random.randint(0, len(self.n_beats))
            while self.n_beats[i_plot] == 0:
                i_plot = random.randint(0, len(self.n_beats))
        else:
            if self.n_beats[i_plot] == 0:
                raise IOError("No beats detected in specified trace")

        # Recalculate offsets for the end of the beats for the signal to be plotted
        if offset_end is None:
            bcls = np.diff(self.t_peaks[i_plot])
            offset_end_list = [max(0.1 * bcl, 30) for bcl in bcls]
        else:
            offset_end_list = [offset_end] * self.n_beats[i_plot]
        beat_end = [t_p - offset for t_p, offset in zip(self.t_peaks[i_plot][1:], offset_end_list)] + \
                   [self.data_uni.index[-1]]

        fig = plt.figure()
        ax = dict()
        ax_labels = ['Unipolar', 'Bipolar', 'Bipolar^2']
        colours = tools.plotting.get_plot_colours(self.n_beats[i_plot])
        for i_ax, data in enumerate([self.data_uni, self.data_bi]):
            ax[ax_labels[i_ax]] = fig.add_subplot(2, 1, i_ax + 1)
            ax[ax_labels[i_ax]].plot(data.loc[:, i_plot], color='C0')
            ax[ax_labels[i_ax]].scatter(self.t_peaks[i_plot], data.loc[:, i_plot][self.t_peaks[i_plot]],
                                        marker='o', edgecolor='tab:orange', facecolor='none', linewidths=2)
            ax[ax_labels[i_ax]].set_ylabel(ax_labels[i_ax])

            i_beat = 1
            max_height = np.max(data.loc[:, i_plot])
            height_shift = (np.max(data.loc[:, i_plot]) - np.min(data.loc[:, i_plot])) * 0.1
            height_val = [max_height, max_height - height_shift] * math.ceil(self.n_beats[i_plot] / 2)
            for t_s, t_e, col, h in zip(self.beat_start[i_plot], beat_end, colours, height_val):
                ax[ax_labels[i_ax]].axvline(t_s, color=col)
                ax[ax_labels[i_ax]].axvline(t_e, color=col)
                ax[ax_labels[i_ax]].annotate(text='{}'.format(i_beat), xy=(t_s, h), xytext=(t_e, h),
                                             arrowprops=dict(arrowstyle='<->', linewidth=3))
                i_beat = i_beat + 1
        fig.suptitle('Trace {}'.format(i_plot))
        return fig, ax

    def get_at(self,
               at_window: float = 30,
               plot: bool = False,
               **kwargs):
        """ Calculates the activation time for a given beat of EGM data

        Will calculate the activation times for an EGM signal, based on finding the peaks in the squared bipolar
        trace, then finding the maximum downslope in the unipolar signal within a specified window of time around
        those peaks.

        Parameters
        ----------
        at_window : float
            Time in milliseconds, around which the activation time will be searched for round the detected peaks
        plot : bool
            Whether to plot a random signal example of the ATs found

        See also
        --------
        :py:meth:`signalanalysis.egm.Egm.get_peaks` : Method to calculate peaks in bipolar signal
        :py:meth:`signalanalysis.egm.Egm.plot_at` : Method to plot the calculated AT
        """

        if self.t_peaks.empty:
            self.get_peaks()

        egm_uni_grad_full = pd.DataFrame(np.gradient(self.data_uni, axis=0),
                                         index=self.data_uni.index,
                                         columns=self.data_uni.columns)

        # Calculate and adjust the start and end point for window searches
        window_start = self.t_peaks-at_window
        window_end = self.t_peaks+at_window
        window_start[window_start < 0] = 0
        window_end[window_end > self.data_uni.index[-1]] = self.data_uni.index[-1]
        window_start = window_start.applymap(lambda y: min(self.data_uni.index, key=lambda x: abs(x-y)))
        window_end = window_end.applymap(lambda y: min(self.data_uni.index, key=lambda x: abs(x-y)))
        window_start[pd.isna(self.t_peaks)] = float("nan")
        window_end[pd.isna(self.t_peaks)] = float("nan")

        self.at = self.t_peaks.copy()
        # Current brute force method
        for key in window_start:
            for i_row, _ in window_start[key].iteritems():
                t_s = window_start.loc[i_row, key]
                if pd.isna(t_s):
                    continue
                t_e = window_end.loc[i_row, key]
                self.at.loc[i_row, key] = egm_uni_grad_full.loc[t_s:t_e, key].idxmin()

        if plot:
            _ = self.plot_signal(plot_at=True, **kwargs)

    def get_rt(self):
        """ Calculate the repolarisation time

        Calculates the repolarisation time of an action potential from the EGM, based on the Wyatt method of the
        maximum upslope of the T-wave

        Parameters
        ----------

        Returns
        -------
        self.rt : pd.DataFrame
            Repolarisation times for each signal in the trace

        References
        ----------
        .. [1] Porter B, Bishop MJ, Claridge S, Behar J, Sieniewicz BJ, Webb J, Gould J, O’Neill M, Rinaldi CA,
               Razavi R, Gill JS, Taggart P, "Autonomic modulation in patients with heart failure increases
               beat-to-beat variability of ventricular action potential duration. Frontiers in Physiology, 8(MAY 2017).
               https://doi.org/10.3389/fphys.2017.00328
        """

        # Estimate BCL from AT, with the BCL for the last AT being assumed to be identical to the prior BCL
        pass
