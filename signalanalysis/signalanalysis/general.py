import math
import numpy as np
import scipy.signal
import scipy.stats
import pandas as pd
import matplotlib.pyplot as plt
from typing import List, Union, Optional

from .. import tools

plt.style.use('seaborn')


class Signal:
    """Base class for general signal, either ECG or VCG

    Attributes
    ----------
    data : pd.DataFrame
        Raw ECG data for the different leads
    filename : str
        Filename for the location of the data
    normalised : bool
        Whether or not the data for the leads have been normalised
    n_beats : int
        Number of beats recorded in the trace. Set to 0 if not calculated
    n_beats_threshold : float (0, 1.0)
        Threshold value used to determine number of beats in overall signal, expressed as a fraction of the maximum
        signal strength detected, default=0.5
    t_peaks : list of float
        Time at which peaks are detected in the overall signal
    beats : list of pd.DataFrame
        Raw data separated into individual beats
    beat_index_reset : bool
        Whether or not the individual data recorded in `beats` has had its time index zeroed (true) or not (false)
    beat_start : list of float
        Start times of the individual beats (allows conversion between individual beat data and overall signal data
        regardless of value of `beat_index_reset`)
    rms : pd.Series
        Record of the RMS data for the signal
    qrs_start : list of float
        Times calculated for the start of the QRS complex
    qrs_end : end
        Times calculated for the end of the QRS complex
    twave_end : end
        Times calculated for the end of the T-wave
    data_source : str
        Source for the data, if known e.g. Staff III database, CARP simulation, etc.
    comments : str
        Any further details known about the data, e.g. sex, age, etc.

    Methods
    -------
    get_rms(unipolar_only=True)
        Returns the RMS of the combined signal
    get_n_beats(threshold=0.5, separation=0.2, plot=False, **kwargs)
        Splits the full signal into individual beats
    plot(separate_beats=False)
        Plot the ECG data
    """
    def __init__(self,
                 **kwargs):
        """Creates parameters that will be common (or are expected to be common) across all signal types

        Parameters
        ----------
        normalise : str
            Whether or not to normalise all ECG leads, default=False
        filter : {'butterworth', 'savitzky-golay'}, optional
            Whether to apply a filter to the signal data, and if so, which filter to apply. Keyword arguments for
            each filter can then be passed (see filters in tools.maths for details)
        """
        # Properties that can be derived subsequently to opening the file
        self.data = pd.DataFrame(dtype=float)
        self.filename = str()

        self.n_beats = 0
        self.n_beats_threshold = 0.5
        self.t_peaks = None
        self.beats = list()
        self.beat_index_reset = False
        self.beat_start = None
        self.rms = None

        self.qrs_start = list()
        self.qrs_end = list()
        self.qrs_duration = list()
        self.twave_end = list()

        self.data_source = None
        self.comments = list()

        # Keyword arguments (optional)
        if 'normalise' in kwargs:
            self.normalised = kwargs.get('normalise')
        else:
            self.normalised = bool()

        # NB: filter must be applied in individual class __init__ functions, as it must be applied after the data
        # have been read into self.data
        if 'filter' in kwargs:
            assert kwargs.get('filter') in ['butterworth', 'savitzky-golay'], "Unknown value for filter_signal passed"
            self.filter = kwargs.get('filter')
        else:
            self.filter = None

    def reset(self):
        """Reset all properties of the class

        Function called when reading in new data into an existing class (for some reason), which would make these
        properties and attributes clash with the other data
        """
        self.data = pd.DataFrame(dtype=float)
        self.filename = str()

        self.n_beats = 0
        self.n_beats_threshold = 0.5
        self.t_peaks = None
        self.beats = list()
        self.beat_index_reset = False
        self.beat_start = None
        self.rms = None

        self.qrs_start = list()
        self.qrs_end = list()
        self.qrs_duration = pd.DataFrame(dtype=float)
        self.twave_end = list()

        self.data_source = None
        self.comments = list()

        self.normalised = bool()
        self.filter = None

    def apply_filter(self, **kwargs):
        """Apply a given filter to the data, using their respective arguments as required

        See Also
        --------
        :py:meth:`tools.maths.filter_butterworth` : Potential filter
        :py:meth:`tools.maths.filter_savitzkygolay` : Potential filter
        """
        if self.filter == 'butterworth':
            self.data = tools.maths.filter_butterworth(self.data, **kwargs)
        elif self.filter == 'savitzky-golay':
            self.data = tools.maths.filter_savitzkygolay(self.data, **kwargs)

    def get_rms(self,
                preprocess_data: pd.DataFrame = None,
                drop_columns: List[str] = None,
                **kwargs):
        """Returns the RMS of the combined signal

        Parameters
        ----------
        preprocess_data : pd.DataFrame, optional
            Only passed if there is some extant data that is to be used for getting the RMS (for example,
            if the unipolar data only from ECG is being used, and the data is thus preprocessed in a manner specific
            for ECG data in the ECG routine)
        drop_columns : list of str, optional
            List of any columns to drop from the raw data before calculating the RMS. Can be used in conjunction with
            preprocess_data

        Returns
        -------
        self.rms : pd.Series
            RMS signal of all combined leads

        Notes
        -----
        The scalar RMS is calculated according to

        .. math::
            \\sqrt{ \\frac{1}{n}\\sum_{i=1}^n (\\textnormal{ECG}_i^2(t)) }

        for all leads available from the signal (12 for ECG, 3 for VCG), unless some leads are excluded via the
        drop_columns parameter.
        """
        if drop_columns is None:
            drop_columns = list()

        if preprocess_data is None:
            signal_rms = self.data.copy()
        else:
            signal_rms = preprocess_data

        if drop_columns is not None:
            assert all(drop_column in signal_rms for drop_column in drop_columns),\
                "Values passed in drop_columns not valid"
            signal_rms.drop(drop_columns, axis=1, inplace=True)

        n_leads = len(signal_rms.columns)
        for key in signal_rms:
            signal_rms.loc[:, key] = signal_rms[key] ** 2
        self.rms = np.sqrt(signal_rms.sum(axis=1) / n_leads)

    def get_peaks(self,
                  threshold: float = 0.4,
                  min_separation: float = 200,
                  plot: bool = False,
                  **kwargs):
        """ Get the peaks of the RMS signal.

        It is often the case that, to determine the number of beats, the peaks in the signal must be determined. This is
        usually done using the RMS of the signal, as demonstrated here. For alternative methods, this method is
        overwritten.

        Parameters
        ----------
        threshold : float {0<1}, optional
            Minimum value to search for for a peak in RMS signal to determine when a beat has occurred, default=0.33
        min_separation : float, optional
            Minimum separation (in ms) required between neighbouring peaks in a given signal (correlates to the minimum
            pacing rate expected), default=200ms
        plot : bool, optional
            Whether to plot results of beat detection, default=False

        Returns
        -------
        self.n_beats : int
            Number of beats in signal
        self.n_beats_threshold : int
            A record of the threshold value used to determine when a peak in the signal could potentially be counted
            as a 'peak'
        self.t_peaks : list of float
            Time at which peaks are detected
        """

        # Calculate locations of RMS peaks to determine number and locations of beats
        if self.rms is None:
            self.get_rms(**kwargs)
        min_separation = self.return_to_index(min_separation)
        i_separation = self.data.index.get_loc(min_separation)
        i_peaks, _ = scipy.signal.find_peaks(self.rms, height=threshold*max(self.rms), distance=i_separation)
        self.n_beats = len(i_peaks)
        self.n_beats_threshold = threshold
        self.t_peaks = self.rms.index[i_peaks]

        if plot:
            _ = self.plot_peaks()

        return

    def plot_peaks(self,
                   **kwargs):
        if self.t_peaks is None:
            self.get_peaks(**kwargs)

        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        ax.plot(self.rms)
        ax.scatter(self.t_peaks, self.rms.loc[self.t_peaks],
                   marker='o', edgecolor='tab:orange', facecolor='none', linewidths=2)
        return fig, ax

    def get_beats(self,
                  reset_index: bool = True,
                  offset_start: Optional[float] = None,
                  offset_end: Optional[float] = None,
                  plot: bool = False,
                  **kwargs):
        """Calculate the number of beats in a given signal, and save the individual beats to the object for later use

        When given the raw data of a given signal (ECG or VCG), will estimate the number of beats recorded in the trace
        based on the RMS of the signal exceeding a threshold value. The estimated individual beats will then be saved in
        a list.

        Parameters
        ----------
        reset_index : bool
            Whether to reset the time index for the separated beats so that they all start from zero (true),
            or whether to leave them with the original time index (false), default=True
        offset_start, offset_end : float, optional
            Time in ms to offset from the prior peak (``offset_start``) and from the following peak (``offset_end``)
            when extracting individual beats, default=None
        plot : bool
            Whether to plot the overall signal, showing where the demarcation between individual plots has been made

        Other Parameters
        ----------------
        unipolar_only : bool, optional
            Only appropriate for ECG data. Whether to use only unipolar ECG leads to calculate RMS, default=True

        Returns
        -------
        self.beats : list of pd.DataFrame
            Individual beats in signal
        self.beat_start : list of float
            Times at which the individual beats start in the overall signal data
        self.beat_index_reset : bool
            Whether or not the time index for each individual beat has been zeroed (true) or not (false)

        Notes
        -----
        If ``offset_start`` and ``offset_end`` are set to 0ms, then the n-th beat will be recorded as starting from the
        point of the (n-1)th peak, and concluding at the point of the (n+1)th peak. This will mean that the beats
        will be saved in a lossless manner, i.e. saved as [ECG1, ECG2, ..., ECG(n)], where ECG1=[0:peak2],
        ECG2=[peak1:peak3], ..., ECGn=[peak(n-1):end]; this will lead to excessive duplication of the data between the
        various recorded peaks, but will ensure that the start of the QRS and T-wave are not accidentally excluded.

        See Also
        --------
        :py:meth:`signalanalysis.general.Signal.get_peaks` : Method to find peaks in the RMS signal
        :py:meth:`signalanalysis.general.Signal.get_rms` : RMS signal calculation required for getting n_beats
        """

        if self.t_peaks is None:
            self.get_peaks(**kwargs)

        # If only one beat is detected, can end here
        self.beat_index_reset = reset_index
        if self.n_beats == 1:
            self.beats = [self.data]
            self.beat_start = [self.data.index[0]]
        else:
            # Calculate series of cycle length values, before then using this to estimate the start and end times of
            # each beat. The offset from the previous peak will be assumed at 0.4*BCL, while the offset from the
            # following peak will be 0.1*BCL (both with a minimum value of 30ms)
            if offset_start is None:
                bcls = np.diff(self.t_peaks)
                offset_start_list = [max(0.6*bcl, 30) for bcl in bcls]
            else:
                offset_start_list = [offset_start]*self.n_beats
            if offset_end is None:
                bcls = np.diff(self.t_peaks)
                offset_end_list = [max(0.1*bcl, 30) for bcl in bcls]
            else:
                offset_end_list = [offset_end]*self.n_beats
            self.beat_start = [self.data.index[0]]+list(self.t_peaks[:-1]+offset_start_list)
            beat_end = [t_p-offset for t_p, offset in zip(self.t_peaks[1:], offset_end_list)]+[self.data.index[-1]]

            for t_s, t_e in zip(self.beat_start, beat_end):
                self.beats.append(self.data.loc[t_s:t_e, :])

            if reset_index:
                for i_beat in range(self.n_beats):
                    zeroed_index = self.beats[i_beat].index-self.beats[i_beat].index[0]
                    self.beats[i_beat].set_index(zeroed_index, inplace=True)

        if plot:
            _ = self.plot_beats(offset_end=offset_end, **kwargs)

    def plot_beats(self,
                   offset_end: Optional[float] = None,
                   **kwargs):
        if self.beats is None:
            self.get_beats(offset_end=offset_end, plot=False, **kwargs)

        if offset_end is None:
            bcls = np.diff(self.t_peaks)
            offset_end_list = [max(0.1*bcl, 30) for bcl in bcls]
        else:
            offset_end_list = [offset_end]*self.n_beats
        beat_end = [t_p-offset for t_p, offset in zip(self.t_peaks[1:], offset_end_list)]+[self.data.index[-1]]

        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        ax.plot(self.rms, color='C0', label='RMS')       # Plot RMS data
        ax.scatter(self.t_peaks, self.rms.loc[self.t_peaks],
                   marker='o', edgecolor='tab:orange', facecolor='none', linewidths=2)
        colours = tools.plotting.get_plot_colours(self.n_beats)
        i_beat = 1
        max_height = np.max(self.rms)
        height_shift = (np.max(self.rms)-np.min(self.rms))*0.1
        height_val = [max_height, max_height-height_shift]*math.ceil(self.n_beats/2)
        for t_s, t_e, col, h in zip(self.beat_start, beat_end, colours, height_val):
            ax.axvline(t_s, color=col)
            ax.axvline(t_e, color=col)
            ax.annotate(text='{}'.format(i_beat), xy=(t_s, h), xytext=(t_e, h),
                        arrowprops=dict(arrowstyle='<->', linewidth=3))
            i_beat = i_beat+1
        return fig, ax

    def get_twave_end(self):
        print("Not coded yet! Don't use!")
        pass

    def return_to_index(self,
                        original_sequence: Union[pd.DataFrame, list, float, int]):
        """Aligns a given sequence of time values to those in the index

        When manipulating time data, it can often fall out of sync with the time values for the data, e.g. the data
        are recorded every 2ms, then searching for a value at 5ms will be difficult. This function realigns a
        sequence of time data to the possible values.

        Parameters
        ----------
        original_sequence : pd.DataFrame or list or float or int
            The sequence of numbers to be returned to the correct time index

        Returns
        -------
        new_sequence : pd.DataFrame or list or float or int
            The sequence of numbers now corrected to match the time index for the data, returned in the same format
            as the original sequence was provided, i.e. pd.DataFrame will return pd.DataFrame, int will return int,
            and so on.
        """

        if isinstance(original_sequence, pd.DataFrame):
            # Reset all values that are less than 0 or greater than the index to be compared against
            original_sequence[original_sequence < 0] = 0
            original_sequence[original_sequence > self.data.index[-1]] = self.data.index[-1]
            new_sequence = original_sequence.applymap(lambda y: min(self.data.index, key=lambda x: abs(x-y)))
            new_sequence[pd.isna(original_sequence)] = float("nan")
        elif isinstance(original_sequence, list):
            # This will technically always find the index greater than the requested value, which may be one step
            # higher than the *actual* closest value, but the difference is judged to be marginal for an approx. five
            # times faster find!
            new_sequence = list()
            for element in original_sequence:
                if element > self.data.index[-1]:
                    element = self.data.index[-1]
                # Below expression is recommended instead of np.where(self.data.index >= original_sequence)[0][0]
                # https://numpy.org/doc/stable/reference/generated/numpy.where.html
                new_sequence.append(self.data.index[np.asarray(self.data.index >= original_sequence).nonzero()[0][0]])
        else:
            if original_sequence > self.data.index[-1]:
                original_sequence = self.data.index[-1]
            try:
                new_sequence = self.data.index[np.asarray(self.data.index >= original_sequence).nonzero()[0][0]]
            except IndexError:
                pass

        return new_sequence


def get_signal_rms(signal: pd.DataFrame,
                   unipolar_only: bool = True) -> List[float]:
    """Calculate the ECG(RMS) of the ECG as a scalar

    .. deprecated::
        The use of this module is deprecated, and the internal class method should be used in preference (
        signalanalysis.general.Signal.get_rms())

    Parameters
    ----------
    signal: pd.DataFrame
        ECG or VCG data to process
    unipolar_only : bool, optional
        Whether to use only unipolar ECG leads to calculate RMS, default=True

    Returns
    -------
    signal_rms : list of float
        Scalar RMS ECG or VCG data

    Notes
    -----
    The scalar RMS is calculated according to

    .. math:: \sqrt{\frac{1}{n}\sum_{i=1}^n (\textnormal{ECG}_i^2(t))}

    for all leads available from the signal (12 for ECG, 3 for VCG). If unipolar_only is set to true, then ECG RMS is
    calculated using only 'unipolar' leads. This uses V1-6, and the non-augmented limb leads (VF, VL and VR)

    ..math:: VF = LL-V_{WCT} = \frac{2}{3}aVF
    ..math:: VL = LA-V_{WCT} = \frac{2}{3}aVL
    ..math:: VR = RA-V_{WCT} = \frac{2}{3}aVR

    References
    ----------
    The development and validation of an easy to use automatic QT-interval algorithm
        Hermans BJM, Vink AS, Bennis FC, Filippini LH, Meijborg VMF, Wilde AAM, Pison L, Postema PG, Delhaas T
        PLoS ONE, 12(9), 1–14 (2017)
        https://doi.org/10.1371/journal.pone.0184352
    """

    assert isinstance(signal, pd.DataFrame)

    signal_rms = signal.copy()
    if unipolar_only and ('V1' in signal_rms.columns):
        signal_rms['VF'] = (2/3)*signal_rms['aVF']
        signal_rms['VL'] = (2/3)*signal_rms['aVL']
        signal_rms['VR'] = (2/3)*signal_rms['aVR']
        signal_rms.drop(['aVF', 'aVL', 'aVR', 'LI', 'LII', 'LIII'], axis=1, inplace=True)
    n_leads = len(signal_rms.columns)
    for key in signal_rms:
        signal_rms.loc[:, key] = signal_rms[key]**2
    signal_rms = np.sqrt(signal_rms.sum(axis=1)/n_leads)

    return signal_rms


def get_bcl(signal_markers: pd.DataFrame):
    """Function to return estimates of the cycle length for a recording

    For a given train of signal markers (either QRS start, AT, or other), will estimate the cycle length as the
    difference between successive markers, with an estimate made that the last cycle length is identical to the
    preceding beat

    Parameters
    ----------
    signal_markers : pd.DataFrame
        Data for the markers to use (usually QRS start for ECG/VCG, or AT for EGM)

    Returns
    -------
    bcl : pd.DataFrame
        Data for estimated BCL for each signal
    """
    bcl = signal_markers.diff(axis=0)
    bcl.drop(0, inplace=True)
    bcl = bcl.append(pd.DataFrame(bcl[-1:].values, columns=bcl.columns))
    bcl.fillna(axis=0, method='ffill', limit=1, inplace=True)
    bcl.reset_index(drop=True, inplace=True)
    return bcl


def get_twave_end(ecgs: Union[List[pd.DataFrame], pd.DataFrame],
                  leads: Union[str, List[str]] = 'LII',
                  i_distance: int = 200,
                  filter_signal: Optional[str] = None,
                  baseline_adjust: Union[float, List[float], None] = None,
                  return_median: bool = True,
                  remove_outliers: bool = True,
                  plot_result: bool = False) -> List[pd.DataFrame]:
    """ Return the time point at which it is estimated that the T-wave has been completed

    Parameters
    ----------
    ecgs : pd.DataFrame or list of pd.DataFrame
        Signal data, either ECG or VCG
    leads : str, optional
        Which lead to check for the T-wave - usually this is either 'LII' or 'V5', but can be set to a list of
        various leads. If set to 'global', then all T-wave values will be calculated. Will return all values unless
        return_median flag is set. Default 'LII'
    i_distance : int, optional
        Distance between peaks in the gradient, i.e. will direct that the function will only find the points of
        maximum gradient (representing T-wave, etc.) with a minimum distance given here (in terms of indices,
        rather than time). Helps prevent being overly sensitive to 'wobbles' in the ecg. Default=200
    filter_signal : {'butterworth', 'savitzky-golay'}, optional
        Whether or not to apply a filter to the data prior to trying to find the actual T-wave gradient. Can pass 
        either a Butterworth filter or a Savitzky-Golay filter, in which case the required kwargs for each can be 
        provided. Default=None (no filter applied)
    baseline_adjust : float or list of float, optional
        Point from which to calculate the adjusted baseline for calculating the T-wave, rather than using the
        zeroline. In line with Hermans et al., this is usually the start of the QRS complex, with the baseline
        calculated as the median amplitude of the 30ms before this point.
    return_median : bool, optional
        Whether or not to return an average of the leads requested, default=True
    remove_outliers : bool, optional
        Whether to remove T-wave end values that are greater than 1 standard deviation from the mean from the data. Only
        has an effect if more than one lead is provided, and return_average is True. Default=True
    plot_result : bool, optional
        Whether to plot the results or not, default=False

    Returns
    -------
    twave_ends : list of pd.DataFrame
        Time value for when T-wave is estimated to have ended.

    Notes
    -----
    Calculates the end of the T-wave as the time at which the T-wave's maximum gradient tangent returns to the
    baseline. The baseline is either set to zero, or set to the median value of 30ms prior to the start of the QRS
    complex (the value of which has to be passed in the `baseline_adjust` variable).

    References
    ----------
    .. [1] Postema PG, Wilde AA, "The measurement of the QT interval," Curr Cardiol Rev. 2014 Aug;10(3):287-94.
           doi:10.2174/1573403x10666140514103612. PMID: 24827793; PMCID: PMC4040880.
    .. [2] Hermans BJM, Vink AS, Bennis FC, Filippini LH, Meijborg VMF, Wilde AAM, Pison L, Postema PG, Delhaas T,
           "The development and validation of an easy to use automatic QT-interval algorithm,"
           PLoS ONE, 12(9), 1–14 (2017), https://doi.org/10.1371/journal.pone.0184352
    """

    # Process the input arguments to ensure they are of the correct form
    if isinstance(ecgs, pd.DataFrame):
        ecgs = [ecgs]

    if leads == 'global':
        leads = ecgs[0].columns
    elif not isinstance(leads, list):
        leads = [leads]

    for ecg in ecgs:
        for lead in leads:
            assert lead in ecg, "Lead not present in ECG"
    
    if filter_signal is not None:
        assert filter_signal in ['butterworth', 'savitzky-golay'], "Unknown value for filter_signal passed"

    # Extract ECG data for the required leads, then calculate the gradient and the normalised gradient
    ecgs_leads = [ecg[leads] for ecg in ecgs]
    if filter_signal == 'butterworth':
        ecgs_leads = [tools.maths.filter_butterworth(ecg) for ecg in ecgs_leads]
    elif filter_signal == 'savitzky-golay':
        ecgs_leads = [tools.maths.filter_savitzkygolay(ecg) for ecg in ecgs_leads]
    ecgs_grad = [pd.DataFrame(index=ecg.index, columns=ecg.columns) for ecg in ecgs_leads]
    ecgs_grad_normalised = [pd.DataFrame(index=ecg.index, columns=ecg.columns) for ecg in ecgs_leads]
    for i_ecg, ecg in enumerate(ecgs_leads):
        for lead in ecg:
            ecg_grad_temp = np.gradient(ecg[lead], ecg.index)
            ecgs_grad[i_ecg].loc[:, lead] = ecg_grad_temp
            ecgs_grad_normalised[i_ecg].loc[:, lead] = tools.maths.normalise_signal(ecg_grad_temp)

    # Calculate the baseline required for T-wave end interpolation
    baseline_adjust = tools.python.convert_input_to_list(baseline_adjust, n_list=len(ecgs), default_entry=0)
    baseline_vals = [pd.DataFrame(columns=ecg.columns, index=[0]) for ecg in ecgs_leads]
    for i_ecg, ecg_leads in enumerate(ecgs_leads):
        if baseline_adjust[i_ecg] == 0:
            for lead in ecg_leads:
                baseline_vals[i_ecg].loc[0, lead] = 0
        else:
            baseline_start = max(0, baseline_adjust[i_ecg]-30)
            for lead in ecg_leads:
                baseline_vals[i_ecg].loc[0, lead] = np.median(ecg_leads[lead][baseline_start:baseline_adjust[i_ecg]])

    # Find last peak in gradient (with the limitations imposed by only looking for a single peak within the range
    # defined by i_distance, to minimise the effect of 'wibbles' in the ecg), then by basic trig find the
    # x-intercept (which is used as the T-wave end point)
    # noinspection PyUnresolvedReferences
    i_peaks = [pd.DataFrame(columns=ecg.columns, index=[0]) for ecg in ecgs_leads]
    twave_ends = [pd.DataFrame(columns=ecg.columns, index=[0]) for ecg in ecgs_leads]
    for i_ecg, ecg in enumerate(ecgs_leads):
        for lead in ecg:
            i_peak_temp = scipy.signal.find_peaks(ecgs_grad_normalised[i_ecg][lead], distance=i_distance)[0]
            t_tpeak_temp = ecg.index[i_peak_temp[-1]]
            baseline_val_temp = baseline_vals[i_ecg].loc[0, lead]
            ecg_grad_temp = ecgs_grad[i_ecg].loc[t_tpeak_temp, lead]
            twave_end_temp = t_tpeak_temp + ((baseline_val_temp-ecg.loc[t_tpeak_temp, lead]) / ecg_grad_temp)
            i_peaks[i_ecg].loc[0, lead] = i_peak_temp
            twave_ends[i_ecg].loc[0, lead] = twave_end_temp

    exclude_columns = [list() for _ in range(len(ecgs))]
    if return_median:
        for i_twave, twave_end in enumerate(twave_ends):
            twave_end_median = np.median(twave_end)
            if remove_outliers:
                twave_end_std = np.std(twave_end.values)
                while True:
                    no_outliers = pd.DataFrame(np.abs((twave_end-twave_end_median)) < 2*twave_end_std)
                    if all(no_outliers.values[0]):
                        break
                    else:
                        twave_end = twave_end[no_outliers]
                        exclude_columns[i_twave].append(twave_end[twave_end.columns[twave_end.isna().any()]].columns[0])
                        twave_end.dropna(axis='columns', inplace=True)
                        twave_end_median = np.median(twave_end)
                        twave_end_std = np.std(twave_end.values)
            twave_ends[i_twave].loc[0, 'median'] = twave_end_median

    if plot_result:
        import signalplot.ecg as ep
        import signalplot.vcg as vp
        from sklearn import preprocessing

        for i_ecg, ecg in enumerate(ecgs):
            # Extract required variables into temporary files (makes code more readable!)
            twave_end = twave_ends[i_ecg]
            i_peak = i_peaks[i_ecg]
            ecg_grad_normalised = ecgs_grad_normalised[i_ecg]
            baseline_val = baseline_vals[i_ecg]
            exclude_column = exclude_columns[i_ecg]

            # Generate plots and axes for leads, as required
            ecg_leads = ['V1', 'V2', 'V3', 'V4', 'V5', 'V6', 'LI', 'LII', 'LIII', 'aVR', 'aVL', 'aVF']
            vcg_leads = ['x', 'y', 'z']
            axes_ecg, axes_vcg = {}, {}
            if any([lead in ecg_leads for lead in leads]):
                _, axes_ecg = ep.plot(ecg)
            if any([lead in vcg_leads for lead in leads]):
                _, axes_vcg = vp.plot_spatial_velocity(ecg)
            axes = {**axes_ecg, **axes_vcg}

            for lead in leads:
                # Rescale the gradient to maximise the plotting range shown, then plot in background
                rescale = preprocessing.MinMaxScaler((min(ecg[lead]), max(ecg[lead])))
                ecg_grad_plot = rescale.fit_transform(ecg_grad_normalised[lead].values[:, None])
                axes[lead].fill_between(ecg.index, min(ecg[lead]), ecg_grad_plot.flatten(), color='C0',
                                        alpha=0.3)

                # Add baseline used to calculate T-wave end, and vertical line to show the calculated T-wave end (and,
                # if calculated, the median value)
                axes[lead].axhline(baseline_val[lead].values, color='k')
                if lead in exclude_column:
                    axes[lead].axvline(twave_end[lead].values, color='r')
                else:
                    axes[lead].axvline(twave_end[lead].values, color='k')
                if 'median' in twave_end:
                    axes[lead].axvline(twave_end['median'].values, color='k', linestyle='--')

                # Add markers for the maximum gradient points, with special labelling for the gradient relevant for
                # the T-wave
                t_peak = ecg.index[i_peak[lead][0][-1]]
                t_peak_full = ecg.index[i_peak[lead][0]]
                axes[lead].plot(t_peak_full, ecg[lead][t_peak_full], marker='s', markerfacecolor='none',
                                markeredgecolor='g', linestyle='none')
                axes[lead].plot(t_peak_full, ecg_grad_plot[i_peak[lead][0]], marker='.', markerfacecolor='none',
                                markeredgecolor='g', linestyle='none')
                axes[lead].plot(t_peak, ecg[lead][t_peak], marker='o', markerfacecolor='none', markeredgecolor='r',
                                linestyle='none')
            if 'sv' in axes:
                axes['sv'].axvline(twave_end['median'].values, color='k', linestyle='--')
            for lead in exclude_column:
                if lead in ecg_leads:
                    axes[lead].set_title(lead, color='r')
                elif lead in vcg_leads:
                    axes[lead].set_ylabel('VCG ('+lead+')', color='r')

    return twave_ends
