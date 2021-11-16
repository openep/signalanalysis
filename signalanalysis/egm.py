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

        self.t_peaks = pd.Series(dtype=float)
        self.n_beats = pd.Series(dtype=int)

        # delattr(self, 'data')
        self.data_uni = pd.DataFrame(dtype=float)
        self.data = self.data_uni
        self.data_bi = pd.DataFrame(dtype=float)

        self.beats_uni = dict()
        self.beats = self.beats_uni
        self.beats_bi = dict()

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

        """
        if self.data_bi.empty:
            super(Egm, self).get_peaks()
            return

        egm_bi_square = np.square(self.data_bi)

        i_separation = np.where(self.data_uni.index > min_separation)[0][0]
        self.n_beats = pd.Series(dtype=int, index=self.data_uni.columns)
        self.t_peaks = pd.DataFrame(dtype=float, columns=self.data_uni.columns)
        # self.t_peaks = dict()
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
            _ = self.plot_peaks()

    def plot_peaks(self,
                   i_plot: Optional[int] = None,
                   **kwargs):
        if self.t_peaks.empty:
            self.get_peaks(**kwargs, plot=False)

        # Pick a random signal to plot as an example trace (making sure to not pick a 'dead' trace)
        if i_plot is None:
            # import random
            # i_plot = random.randint(0, len(self.n_beats))
            i_plot = self.n_beats.sample().index[0]
            while self.n_beats[i_plot] == 0:
                # i_plot = random.randint(0, len(self.n_beats))
                i_plot = self.n_beats.sample().index[0]
        else:
            if self.n_beats[i_plot] == 0:
                raise IOError("No beats detected in specified trace")

        egm_bi_square = np.square(self.data_bi)
        fig = plt.figure()
        fig.suptitle('Trace {}'.format(i_plot))
        ax = dict()
        ax_labels = ['Unipolar', 'Bipolar', 'Bipolar^2']
        for i_ax, data in enumerate([self.data_uni, self.data_bi, egm_bi_square]):
            ax[ax_labels[i_ax]] = fig.add_subplot(3, 1, i_ax+1)
            ax[ax_labels[i_ax]].plot(data.loc[:, i_plot], color='C0')
            ax[ax_labels[i_ax]].scatter(
                self.t_peaks[i_plot].dropna(),
                data.loc[:, i_plot][self.t_peaks[i_plot].dropna()],
                marker='o', edgecolor='tab:orange', facecolor='none', linewidths=2)
            ax[ax_labels[i_ax]].set_ylabel(ax_labels[i_ax])
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

        n_signals = len(self.data_uni.columns)
        self.beat_start = pd.Series(dtype=pd.DataFrame, index=self.data_uni.columns)
        self.beats_uni = dict.fromkeys(self.data_uni.columns)
        self.beats_bi = dict.fromkeys(self.data_uni.columns)
        for i_signal in range(n_signals):
            # If only one beat is detected, can end here
            if self.n_beats[i_signal] == 1:
                self.beats_uni.append([self.data_uni.loc[:, i_signal]])
                self.beats_bi.append([self.data_bi.loc[:, i_signal]])
                continue

            # Calculate series of cycle length values, before then using this to estimate the start and end times of
            # each beat. The offset from the previous peak will be assumed at 0.4*BCL, while the offset from the
            # following peak will be 0.1*BCL (both with a minimum value of 30ms)
            if offset_start is None:
                bcls = np.diff(self.t_peaks[i_signal])
                offset_start_list = [max(0.6 * bcl, 30) for bcl in bcls]
            else:
                offset_start_list = [offset_start] * self.n_beats[i_signal]
            if offset_end is None:
                bcls = np.diff(self.t_peaks[i_signal])
                offset_end_list = [max(0.1 * bcl, 30) for bcl in bcls]
            else:
                offset_end_list = [offset_end] * self.n_beats[i_signal]
            self.beat_start.append([self.data_uni.index[0]] + list(self.t_peaks[i_signal][:-1] + offset_start_list))
            beat_end = [t_p - offset for t_p, offset in zip(self.t_peaks[i_signal][1:], offset_end_list)] + \
                       [self.data_uni.index[-1]]

            signal_beats_uni = list()
            signal_beats_bi = list()
            for t_s, t_p, t_e in zip(self.beat_start[-1], self.t_peaks[i_signal], beat_end):
                if i_signal == 474:
                    pass
                assert t_s < t_p < t_e, "Error in windowing process"
                signal_beats_uni.append(self.data_uni.loc[t_s:t_e, :])
                signal_beats_bi.append(self.data_bi.loc[t_s:t_e, :])

            self.beat_index_reset = reset_index
            if reset_index:
                for i_beat in range(self.n_beats[i_signal]):
                    zeroed_index = signal_beats_uni[i_beat].index - signal_beats_uni[i_beat].index[0]
                    signal_beats_uni[i_beat].set_index(zeroed_index, inplace=True)
                    signal_beats_bi[i_beat].set_index(zeroed_index, inplace=True)
            self.beats_uni.append(signal_beats_uni)
            self.beats_bi.append(signal_beats_bi)

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
               at_window: float = 30):
        """ Calculates the activation time for a given beat of EGM data

        Will calculate the activation times for an EGM signal, based on finding the peaks in the squared bipolar
        trace, then finding the maximum downslope in the unipolar signal within a specified window of time around
        those peaks.

        Parameters
        ----------
        at_window : float
            Time in milliseconds, around which the activation time will be searched for round the detected peaks

        See also
        --------
        :py:meth:`signalanalysis.egm.Egm.get_peaks` : Method to calculate peaks in bipolar signal
        """

        if self.t_peaks == 0:
            self.get_peaks()

        egm_uni_grad_full = pd.DataFrame(np.gradient(self.data_uni, axis=0),
                                         index=self.data_uni.index,
                                         columns=self.data_uni.columns)
        window_start = self.t_peaks-at_window
        window_end = self.t_peaks+at_window
        window_start = window_start.applymap(lambda y: min(self.data_uni.index, key=lambda x: abs(x-y)))
        window_end = window_end.applymap(lambda y: min(self.data_uni.index, key=lambda x: abs(x-y)))
        window_start = pd.DataFrame(columns=self.data_uni.columns)
        window_end = pd.DataFrame(columns=self.data_uni.columns)
        self.qrs_start = self.t_peaks.copy()
        for key in egm_uni_grad_full:
            window_start[key] = [min(self.data_uni.index,
                                 key=lambda x: abs(x-(t_p-at_window))) for t_p in self.t_peaks[key]]
            window_end[key] = [min(self.data_uni.index,
                               key=lambda x: abs(x-(t_p+at_window))) for t_p in self.t_peaks[key]]
            # t_at = [egm_uni_grad_full.loc[w_start:w_end, egm_uni_grad].min() for w_start, w_end in zip(window_start,
            #                                                                                            window_end)]
            pass
