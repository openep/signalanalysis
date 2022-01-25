import math
import warnings

import numpy as np
import pandas as pd
import scipy.signal
import matplotlib.pyplot as plt
from typing import Optional, Union, List
from tqdm import tqdm

from signalanalysis.signalanalysis import general
from signalanalysis import signalplot
from signalanalysis import tools


class Egm(general.Signal):
    """Base class for EGM data, inheriting from :class:`signalanalysis.signalanalysis.general.Signal`

    See Also
    --------
    :py:class:`signalanalysis.signalanalysis.general.Signal`

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
        :py:meth:`signalanalysis.signalanalysis.general.Signal.__init__ : Base __init__ method
        :py:meth:`signalanalysis.signalanalysis.general.Signal.apply_filter` : Filtering method
        :py:meth:`signalanalysis.signalanalysis.general.Signal.get_n_beats` : Beat calculation method

        Notes
        -----
        This used to break the `Liskov substitution principle
        <https://en.wikipedia.org/wiki/Liskov_substitution_principle>`_, removing the single `data` attribute to be
        replaced by `data_uni` and `data_bi`, but now instead (aims to) just point the `.data` attribute to the
        `.data_uni` attribute
        """
        super(Egm, self).__init__(**kwargs)

        self.t_peaks = pd.DataFrame(dtype=float)
        self.n_beats = pd.Series(dtype=int)

        # delattr(self, 'data')
        self.data_uni = pd.DataFrame(dtype=float)
        self.data_bi = pd.DataFrame(dtype=float)

        self.beats_uni = dict()
        self.beats = self.beats_uni
        self.beats_bi = dict()

        self.at = pd.DataFrame(dtype=float)
        self.rt = pd.DataFrame(dtype=float)
        self.ari = pd.DataFrame(dtype=float)
        self.dvdt = pd.DataFrame(dtype=float)

        self.qrs_start = pd.DataFrame(dtype=float)
        self.qrs_end = pd.DataFrame(dtype=float)
        self.qrs_duration = pd.DataFrame(dtype=float)

        self.read(data_location_uni, data_location_bi, **kwargs)
        if self.filter is not None:
            self.apply_filter(**kwargs)
        self.data = self.data_uni
        # self.get_beats(**kwargs)

    def read(self,
             data_location_uni: str,
             data_location_bi: Optional[str] = None,
             drop_empty_rows: bool = True,
             **kwargs):
        """ Read the DxL data for unipolar and bipolar data for EGMs

        TODO: Add functionality to read directly from folders, rather than .csv from Matlab

        Parameters
        ----------
        data_location_uni : str
            Location of unipolar data. Currently only coded to deal with a saved .csv file
        data_location_bi : str, optional
            Location of bipolar data. Currently only coded to deal with a saved .csv file. Doesn't need to be passed,
            default=None
        drop_empty_rows : bool, optional
            Whether to drop empty data rows from the data, default=True

        See Also
        --------
        :py:meth:`signalanalysis.signalanalysis.egm.Egm.read_from_csv` : Method to read data from Matlab csv
        """

        if data_location_uni.endswith('.csv'):
            if data_location_bi is not None:
                assert data_location_bi.endswith('.csv')
            self.read_from_csv(data_location_uni, data_location_bi, **kwargs)
        else:
            raise IOError("Not coded for this type of input")

        if drop_empty_rows:
            # PyCharm highlights an error below (bool doesn't have a .all() method), but I'll be damned if I can
            # figure out how to fix it - the below option replaces *all* 0.00 values, so will put NaN in an otherwise
            # normal trace where it happens to reach 0.00, which is not what we want.
            # self.data_uni = (self.data_uni.where(self.data_uni != 0, axis=0)).dropna(axis=1, how='all')
            self.data_uni = self.data_uni.loc[:, ~(self.data_uni == 0).all(axis=0)]
            if not self.data_bi.empty:
                self.data_bi = self.data_bi.loc[:, ~(self.data_bi == 0).all(axis=0)]
                assert self.data_uni.shape == self.data_bi.shape, "Error in dropping rows"

        return None

    def read_from_csv(self,
                      data_location_uni: str,
                      data_location_bi: Optional[str],
                      frequency: float):
        """ Read EGM data that has been saved from Matlab

        Parameters
        ----------
        data_location_uni : str
            Name of the .csv file containing the unipolar data
        data_location_bi : str, optional
            Name of the .csv file containing the bipolar data
        frequency : float
            The frequency of the data recording in Hz

        Notes
        -----
        It is not technically required to pass the bipolar data, but it is presented here as a required keyword to
        preserve the usage of calling as `read_from_csv(unipolar, bipolar, frequency)`, rather than breaking the data
        files arguments up or requiring keywords.

        The .csv file should be saved with column representing an individual EGM trace, and each row representing a
        single instance in time, i.e.

        .. code-block::
            egm1(t1), egm2(t1), egm3(t1), ...
            egm1(t2), egm2(t2), egm3(t2), ...
            ...
            egm1(tn), egm2(tn), egm3(tn)

        Historically, `frequency` has been set to 2034.5 Hz for the importprecision data, an example of which is
        can be accessed via ``signalanalysis.data.datafiles.EGM_UNIPOLAR`` and ``signalanalysis.data.datafiles.EGM_BIPOLAR``.
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
            self.data_bi = pd.DataFrame()
            self.data_source = data_location_uni

        return None

    def get_peaks(self,
                  threshold: float = 0.33,
                  min_separation: float = 200,
                  plot: bool = False,
                  **kwargs):
        """ Supermethod for get_peaks for EGM data, using the squared bipolar signal rather than RMS data

        See also
        --------
        :py:meth:`signalanalysis.signalanalysis.egm.Egm.plot_signal` : Method to plot the calculated AT
        """
        if self.data_bi.empty:
            # super(Egm, self).get_peaks()
            egm_bi_square = np.abs(self.data_uni)
        else:
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

            # Pad the peaks data or t_peaks dataframe with NaN as appropriate
            if len(i_peaks) == self.t_peaks.shape[0]:
                self.t_peaks[i_signal] = self.data_uni.index[i_peaks]
            elif len(i_peaks) < self.t_peaks.shape[0]:
                self.t_peaks[i_signal] = np.pad(self.data_uni.index[i_peaks],
                                                (0, self.t_peaks.shape[0]-len(i_peaks)),
                                                constant_values=float("nan"))
            elif len(i_peaks) > self.t_peaks.shape[0]:
                self.t_peaks = self.t_peaks.reindex(range(len(i_peaks)), fill_value=float("nan"))
                self.t_peaks[i_signal] = self.data_uni.index[i_peaks]

        if plot:
            _ = signalplot.egm.plot_signal(self, plot_peaks=True, plot_bipolar_square=True, **kwargs)

        return None

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
        :py:meth:`signalanalysis.signalanalysis.general.Signal.get_beats` : Base method
        """
        if self.t_peaks.empty:
            self.get_peaks(**kwargs)

        # we'll store these values in data frames later on
        beat_start_values = np.full_like(self.t_peaks, fill_value=np.NaN)
        beat_end_values = np.full_like(self.t_peaks, fill_value=np.NaN)

        self.beats_uni = dict.fromkeys(self.data_uni.columns)
        self.beats_bi = dict.fromkeys(self.data_uni.columns)

        all_bcls = np.diff(self.t_peaks, axis=0).T
        for key, bcls in zip(self.data_uni, all_bcls):

            # If only one beat is detected, can end here
            n_beats = self.n_beats[key]
            if n_beats == 1:
                self.beats_uni[key] = [self.data_uni.loc[:, key]]
                self.beats_bi[key] = [self.data_bi.loc[:, key]]
                continue

            # Calculate series of cycle length values, before then using this to estimate the start and end times of
            # each beat. The offset from the previous peak will be assumed at 0.4*BCL, while the offset from the
            # following peak will be 0.1*BCL (both with a minimum value of 30ms)

            if offset_start is None:
                offset_start_list = [max(0.6 * bcl, 30) for bcl in bcls[:n_beats-1]]
            else:
                offset_start_list = [offset_start] * (self.n_beats[key] - 1)

            if offset_end is None:
                offset_end_list = [max(0.1 * bcl, 30) for bcl in bcls[:n_beats-1]]
            else:
                offset_end_list = [offset_end] * (self.n_beats[key] - 1)

            beat_start = [self.data_uni.index[0]]
            beat_start.extend(self.t_peaks[key][:n_beats-1].values + offset_start_list)

            beat_end = []
            beat_end.extend(self.t_peaks[key][1:n_beats].values - offset_end_list)
            beat_end.append(self.data_uni.index[-1])

            # we'll store these values in data frames later on
            column_index = self.t_peaks.columns.get_loc(key)
            beat_start_values[:n_beats, column_index] = beat_start
            beat_end_values[:n_beats, column_index] = beat_end

            signal_beats_uni = np.empty(n_beats, dtype=object)
            signal_beats_bi = np.empty(n_beats, dtype=object)
            for beat_index, (t_s, t_p, t_e) in enumerate(zip(beat_start, self.t_peaks[key], beat_end)):
                
                if not (t_s < t_p < t_e):
                    raise ValueError("Error in windowing process - a peak is outside of the window for EGM ", key)
                signal_beats_uni[beat_index] = self.data_uni.loc[t_s:t_e, :]
                signal_beats_bi[beat_index] = self.data_bi.loc[t_s:t_e, :]
                
                if not reset_index:
                    continue
                
                zeroed_index = signal_beats_uni[beat_index].index - signal_beats_uni[beat_index].index[0]
                signal_beats_uni[beat_index].set_index(zeroed_index, inplace=True)
                signal_beats_bi[beat_index].set_index(zeroed_index, inplace=True)

            self.beat_index_reset = reset_index
            self.beats_uni[key] = signal_beats_uni
            self.beats_bi[key] = signal_beats_bi

        self.beat_start = pd.DataFrame(
            data=beat_start_values,
            index=self.t_peaks.index,
            columns=self.t_peaks.columns,
            dtype=float,
        )
        self.beat_end = pd.DataFrame(
            data=beat_end_values,
            index=self.t_peaks.index,
            columns=self.t_peaks.columns,
            dtype=float,
        )

        if plot:
            _ = self.plot_beats(offset_end=offset_end, **kwargs)

    def plot_beats(self,
                   i_plot: Optional[int] = None,
                   **kwargs):
        """
        ..deprecated::
            Need to move this to signalanalysis.signalplot.egm (if this even works!)
        """

        # Calculate beats (if not done already)
        if self.beats_uni is None:
            self.get_beats(offset_end=None, plot=False, **kwargs)

        # Pick a random signal to plot as an example trace (making sure to not pick a 'dead' trace)
        if i_plot is None:
            weights = (self.n_beats.values > 0).astype(int)
            i_plot = self.n_beats.sample(weights=weights).index[0]
        elif self.n_beats[i_plot] == 0:
                raise IOError("No beats detected in specified trace")

        n_beats = n_beats = self.n_beats[i_plot]
        t_peaks = self.t_peaks[i_plot]
        beat_start = self.beat_start[i_plot]
        beat_end = self.beat_end[i_plot]

        ax_labels = ['Unipolar', 'Bipolar']
        egm_data = [self.data_uni, self.data_bi]
        colours = tools.plotting.get_plot_colours(n_beats)

        fig, axes = plt.subplots(2, 1)
        fig.suptitle('Trace {}'.format(i_plot))

        for index, (ax, data) in enumerate(zip(axes, egm_data)):
            
            plt.sca(ax)
            plt.plot(data.loc[:, i_plot], color='C0')
            plt.scatter(
                t_peaks[:n_beats],
                data.loc[:, i_plot][t_peaks[:n_beats]],
                marker='o',
                edgecolor='tab:orange',
                facecolor='none',
                linewidths=2,
            )
            plt.ylabel(ax_labels[index])

            max_height = np.max(data.loc[:, i_plot])
            height_shift = (np.max(data.loc[:, i_plot]) - np.min(data.loc[:, i_plot])) * 0.1
            height_val = [max_height, max_height - height_shift] * math.ceil(n_beats / 2)
            for beat_index, (t_s, t_e) in enumerate(zip(beat_start[:n_beats], beat_end[:n_beats])):

                plt.axvline(t_s, color=colours[beat_index])
                plt.axvline(t_e, color=colours[beat_index])
                plt.annotate(text='{}'.format(beat_index+1), xy=(t_s, height_val[beat_index]), xytext=(t_e, height_val[beat_index]),
                                             arrowprops=dict(arrowstyle='<->', linewidth=3))

        return fig, ax

    def get_at(self,
               at_window: float = 30,
               unipolar_delay: float = 50,
               plot: bool = False,
               **kwargs):
        """ Calculates the activation time for a given beat of EGM data

        Will calculate the activation times for an EGM signal, based on finding the peaks in the squared bipolar
        trace, then finding the maximum downslope in the unipolar signal within a specified window of time around
        those peaks. Note that, if bipolar data are not present, then the squared unipolar signal will be used,
        which will invariably find the pacing artefact. As such, when unipolar peaks are used, a 'delay' will be
        applied to the window to avoid the pacing artefact.

        Parameters
        ----------
        at_window : float, optional
            Time in milliseconds, around which the activation time will be searched for round the detected peaks,
            i.e. the EGM trace will be searched in the window t_peak +/- at_window. Default=30ms
        unipolar_delay : float, optional
            Time in milliseconds to delay the search window after the peak time, if only unipolar data are being
            used, to avoid getting confused with the far-field pacing artefact. Will thus have the search window
            adapted to (t_peak+unipolar_delay) +/- at_window. Default=50ms
        plot : bool, optional
            Whether to plot a random signal example of the ATs found, default=False

        See also
        --------
        :py:meth:`signalanalysis.signalanalysis.egm.Egm.get_peaks` : Method to calculate peaks
        :py:meth:`signalanalysis.signalanalysis.egm.Egm.plot_signal` : Method to plot the signal
        """

        if self.t_peaks.empty:
            self.get_peaks()

        egm_uni_grad_full = pd.DataFrame(np.gradient(self.data_uni, axis=0),
                                         index=self.data_uni.index,
                                         columns=self.data_uni.columns)

        # Calculate and adjust the start and end point for window searches
        if not self.data_bi.empty:
            unipolar_delay = 0
        window_start = self.return_to_index(self.t_peaks.sub(at_window).add(unipolar_delay))
        window_end = self.return_to_index(self.t_peaks.add(at_window).add(unipolar_delay))

        self.at = self.t_peaks.copy()
        # Current brute force method
        from tqdm import tqdm
        for key in tqdm(window_start, desc='Finding AT...', total=len(window_start.columns)):
            for i_row, _ in window_start[key].iteritems():
                t_s = window_start.loc[i_row, key]
                if pd.isna(t_s):
                    continue
                t_e = window_end.loc[i_row, key]
                self.at.loc[i_row, key] = egm_uni_grad_full.loc[t_s:t_e, key].idxmin()
                self.dvdt.loc[i_row, key] = egm_uni_grad_full.loc[t_s:t_e, key].min()

        if plot:
            _ = signalplot.egm.plot_signal(self, plot_at=True, **kwargs)

        return None

    def get_rt(self,
               lower_window_limit: float = 140,
               plot: bool = False,
               **kwargs):
        """ Calculate the repolarisation time

        Calculates the repolarisation time of an action potential from the EGM, based on the Wyatt method of the
        maximum upslope of the T-wave

        TODO: try to improve on the current brute force method used to find the point of RT

        Parameters
        ----------
        lower_window_limit : float, optional
            Minimum time after the AT to have passed before repolarisation can potentially happen, default=150ms
        plot : bool, optional
            Whether to plot a random signal example of the ATs found, default=False

        Returns
        -------
        self.rt : pd.DataFrame
            Repolarisation times for each signal in the trace
        self.ari : pd.DataFrame
            Activation repolarisation intervals for each AT/RT pair

        References
        ----------
        .. [1] Porter B, Bishop MJ, Claridge S, Behar J, Sieniewicz BJ, Webb J, Gould J, O’Neill M, Rinaldi CA,
               Razavi R, Gill JS, Taggart P, "Autonomic modulation in patients with heart failure increases
               beat-to-beat variability of ventricular action potential duration. Frontiers in Physiology, 8(MAY 2017).
               https://doi.org/10.3389/fphys.2017.00328
        """

        # Estimate BCL, then calculate the upper and lower bounds within which to search for the repolarisation time
        if self.at.empty:
            self.get_at(**kwargs)
        bcl = general.get_bcl(self.at)

        # INITIALISE WINDOWS WITHIN WHICH TO SEARCH FOR RT

        window_start = (bcl.mul(0.75)).sub(125)   # Equivalent to 0.75*bcl-125
        window_start[window_start < lower_window_limit] = lower_window_limit
        window_start = self.at+window_start

        window_end = (bcl.mul(0.9)).sub(50)     # Equivalent to 0.9*bcl-50
        window_end = self.at+window_end
        window_end[window_end-window_start < 0.1] = window_start+0.1

        # If the end of the search window is within 20ms of the next AT/end of the recording, shorten the end of the
        # window accordingly
        # Don't bother looking for RT if the start of the search window is within 40ms of the following AT/the end of
        # the recording.

        def window_max_generator(buffer):
            window_max = self.at - buffer
            window_max.set_index(window_max.index - 1, inplace=True)
            window_max.drop(-1, inplace=True)
            window_max = window_max.append(pd.DataFrame(self.data_uni.index[-1] - buffer,
                                                        columns=window_max.columns,
                                                        index=[window_max.index[-1] + 1]))
            window_max[window_max > self.data_uni.index[-1] - buffer] = \
                self.data_uni.index[-1] - buffer
            window_max.fillna(axis=0, method='bfill', inplace=True)
            return window_max

        window_start_max = window_max_generator(40)
        window_start[window_start > window_start_max] = float("nan")
        window_start = self.return_to_index(window_start)

        window_end_max = window_max_generator(20)
        window_end[window_end > window_end_max] = window_end_max
        window_end = self.return_to_index(window_end)

        # Brute force method!
        egm_uni_grad = pd.DataFrame(np.gradient(self.data_uni, axis=0),
                                    index=self.data_uni.index,
                                    columns=self.data_uni.columns)
        self.rt = pd.DataFrame(index=self.at.index, columns=self.at.columns)
        self.ari = pd.DataFrame(index=self.at.index, columns=self.at.columns)
        for key in tqdm(window_start, desc='Finding RT...', total=len(window_start.columns)):
            for i_row, _ in enumerate(window_start[key]):

                # FIND T-WAVE PEAK

                # Look for the peak of the unipolar EGM within the search window. If the maximum (+/- 0.03mV) is at the
                # start/end of the window, shorten the window and check again to try and ensure that the peak
                # represents the T-wave peak rather than the repolarisation/depolarisation preceding/following the
                # T-wave.
                window_error = False
                negative_t_wave = False

                t_start = window_start.loc[i_row, key]
                t_end = window_end.loc[i_row, key]
                if pd.isna(t_start) or pd.isna(t_end):
                    continue

                i_ts = np.where(self.data_uni.index.values == t_start)[0]
                i_te = np.where(self.data_uni.index.values == t_end)[0]
                uni_start = self.data_uni.loc[t_start, key]
                uni_end = self.data_uni.loc[t_end, key]

                uni_peak = self.data_uni.loc[t_start:t_end, key].max()
                uni_start_diff = abs(uni_start-uni_peak)
                uni_end_diff = abs(uni_end-uni_peak)
                while uni_start_diff <= 0.03 or uni_end_diff <= 0.03:
                    while uni_start_diff <= 0.03:
                        i_ts = i_ts+1
                        try:
                            t_start = self.data_uni.index[i_ts][0]
                        except IndexError:
                            pass
                        if t_start >= t_end:
                            window_error = True
                            break
                        uni_start = self.data_uni.loc[t_start, key]
                        uni_peak = self.data_uni.loc[t_start:t_end, key].max()
                        uni_start_diff = abs(uni_start-uni_peak)
                    if window_error:
                        break
                    while uni_end_diff <= 0.03:
                        i_te = i_te-1
                        try:
                            t_end = self.data_uni.index[i_te][0]
                        except IndexError:
                            pass
                        if t_start >= t_end:
                            window_error = True
                            break
                        uni_end = self.data_uni.loc[t_end, key]
                        uni_peak = self.data_uni.loc[t_start:t_end, key].max()
                        uni_end_diff = abs(uni_end-uni_peak)

                # If it is impossible to narrow the search window as above and find a positive peak for the T-wave,
                # set the window to the original values and assume that the T-wave is negative
                if window_error or not (egm_uni_grad.loc[t_start:t_end, key] > 0).any():
                    t_start = window_start.loc[i_row, key]
                    t_end = window_start.loc[i_row, key]
                    t_peak = self.data_uni.loc[t_start:t_end, key].idxmin()
                    negative_t_wave = True
                else:
                    t_peak = self.data_uni.loc[t_start:t_end, key].idxmax()

                assert t_start <= t_peak <= t_end, "Problem setting window values"

                # FIND REPOLARISATION TIME

                max_grad = -100
                t_max_grad = -1
                window_data = egm_uni_grad.loc[t_start:t_end, key]
                t_index_in_uni_data = np.searchsorted(self.data_uni.index.values, window_data.index.values)
                for (t_window, uni_val), i_tm in zip(window_data.iteritems(), t_index_in_uni_data):
                    # Look for maximum gradient in the search window thus far
                    if uni_val > max_grad:
                        max_grad = uni_val
                        t_max_grad = t_window

                    # Perform check to see if we've exited the current T-wave (if we're after the total max peak
                    # (apex) and have negative gradient)
                    if negative_t_wave:
                        self.rt.loc[i_row, key] = t_max_grad
                        self.ari.loc[i_row, key] = t_max_grad - self.at.loc[i_row, key]
                    else:
                        t1 = self.data_uni.index[i_tm-1]
                        t2 = self.data_uni.index[i_tm+2]    # Adding 2 to ensure that the limit is taken at +1, not i_tm
                        if (t_window > t_peak) and (window_data.loc[t1:t2] < 0).all():
                            self.rt.loc[i_row, key] = t_max_grad
                            self.ari.loc[i_row, key] = t_max_grad - self.at.loc[i_row, key]
                            break

        if plot:
            _ = signalplot.egm.plot_signal(self, plot_rt=True, **kwargs)

        return None

    def get_ari(self,
                plot: bool = False,
                **kwargs):
        """Dummy function to calculate ARI

        TODO: check that `plot` keyword is correctly over-ridden (if that is possible)

        ARI is calculated as part of self.get_rt, so this module serves just as useful syntax.

        See also
        --------
        :py:meth:`signalanalysis.signalanalysis.egm.Egm.get_rt` : Actual method called
        """
        if self.ari.empty:
            self.get_rt(plot=False, **kwargs)

        if plot:
            signalplot.egm.plot_signal(plot_at=True, plot_rt=True, **kwargs)
        return None

    def get_qrsd(self,
                 lower_window: float = 30,
                 upper_window: float = 60,
                 threshold: float = 0.1,
                 plot: bool = True,
                 **kwargs):
        """Calculates the QRS duration for EGM data

        TODO: See if this can be improved beyond the current brute force method

        The start and end of the QRS complex is calculated as the duration for which the energy of the bipolar signal
        (defined as the bipolar signal squared) exceeds a threshold value. The 'window' over which to search for this
        complex is defined from the detected activation times, plus/minus specified values (`lower_window` and
        `upper_window`). Note that, due to the calculation method, this cannot be calculated for instances where no
        bipolar data are available.

        Parameters
        ----------
        lower_window, upper_window : float, optional
            Window before/after AT to search for QRS start/end, respectively, given in milliseconds, default=30/60ms
        threshold : float, optional
            Fractional threshold of maximum energy used to define the start and end of the QRS complex, default=0.1
        plot : bool, optional
            Whether or not to plot an example trace

        Returns
        -------
        self.qrs_duration : pd.DataFrame
            QRS durations for each signal

        See also
        --------
        :py:meth:`signalanalysis.signalanalysis.egm.Egm.get_at` : Method used to calculate AT, that uses this method implicitly
        :py:meth:`signalanalysis.signalplot.egm.plot_signal` : Plotting function, with options that can be passed in **kwargs
        """

        if self.data_bi.empty:
            raise IOError('Cannot calculate QRSd for unipolar only data')
        
        # Sanitise inputs and make sure they make sense
        if lower_window < 0.5:
            warnings.warn('Assuming that lWindow has been entered in seconds rather than milliseconds: correcting...')
            lower_window = lower_window * 1000
        if upper_window < 0.5:
            warnings.warn('Assuming that uWindow has been entered in seconds rather than milliseconds: correcting...')
            upper_window = upper_window * 1000
        assert 0.0 < threshold < 1.0, "threshold must be set between 0 and 1"

        if self.at.empty:
            self.get_at(**kwargs)

        window_start = self.return_to_index(self.at.sub(lower_window))
        window_end = self.return_to_index(self.at.add(upper_window))
        for key in tqdm(self.at, desc='Finding QRSd...', total=len(self.at.columns)):
            for i_row, _ in enumerate(self.at[key]):
                # Only continue if the window values aren't NaN
                if pd.isna(window_start.loc[i_row, key]) or pd.isna(window_end.loc[i_row, key]):
                    continue

                # Calculate EGM energy within the window of concern
                energy = np.square(self.data_bi.loc[window_start.loc[i_row, key]:window_end.loc[i_row, key], key])

                # Find threshold values within this window
                energy_threshold = energy.max()*threshold

                i_qrs = np.where(energy > energy_threshold)
                self.qrs_start.loc[i_row, key] = energy.index[i_qrs[0][0]]
                self.qrs_end.loc[i_row, key] = energy.index[i_qrs[0][-1]]
                self.qrs_duration.loc[i_row, key] = self.qrs_end.loc[i_row, key] - self.qrs_start.loc[i_row, key]

        if plot:
            signalplot.egm.plot_signal(self, plot_qrsd=True, **kwargs)
        return None

    def calculate_dvdt(self,
                       time_points: Union[float, List[float], pd.DataFrame] = None,
                       dvdt_normalise: bool = False,
                       dvdt_rescale: bool = False):
        """Return dV/dt values at specified time points

        TODO: Write this function, if useful - currently simpler to just calculate dV/dt for AT directly in method

        Will return the value of dV/dt at specified time points.

        Parameters
        ----------
        time_points : list of float
            List of time points at which to calculate dV/dt for the signal
        dvdt_normalise : bool, optional
            Whether to normalise the ECG trace to a [-1, 1] range prior to calculating the dVdt value - will only adjust
            the values relative to a maximum, and not necessarily rescale within the entire range, i.e. an EGM from [0,
            10] will rescale to [0, 1]. This will not affect any other part of the calculations, default=False
        dvdt_rescale : bool, optional
            Whether to normalise the ECG trace to a [-1, 1] range prior to calculating the dVdt value - will adjust
            the values relative cover the entire new range, i.e. an EGM from [0, 10] will rescale to [-1,
            1]. This will not affect any other part of the calculations, default=false

        Returns
        -------
        self.dvdt : pd.DataFrame
            dV/dt values for each point of the AT

        See also
        --------
        :py:meth:`signalanalysis.signalanalysis.egm.Egm.get_at` : Method used to calculate AT, that uses this method implicitly
        """

        time_points = self.return_to_index(time_points)
        return None
