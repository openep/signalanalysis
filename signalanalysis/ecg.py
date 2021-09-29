import re
import numpy as np
import pandas as pd
import scipy.signal
import wfdb
from typing import List, Union, Optional
import warnings
import matplotlib.pyplot as plt
import seaborn as sns

from carputils.carpio import igb  # type: ignore

import tools.maths
import signalanalysis.general

sns.set()


class Ecg:
    """Base ECG class to encapsulate data regarding an ECG recording

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
    qrs_start : list of float
        Times calculated for the start of the QRS complex
    qrs_end : end
        Times calculated for the end of the QRS complex
    data_source : str
        Source for the data, if known e.g. Staff III database, CARP simulation, etc.
    comments : str
        Any further details known about the data, e.g. sex, age, etc.

    Methods
    -------
    read(filename)
        Reads in the data from the original file. Called upon initialisation
    read_ecg_from_wfdb(filename, normalise=False)
        Reads data from a WFDB type series of files, e.g. from the Lobachevsky ECG database
        (https://physionet.org/content/ludb/1.0.1/)
    get_n_beats(threshold=0.5, min_separation=0.2)
        Calculates the number of beats given in the recording
    get_qrs_start()
        Calculates the start of the QRS complex
    """
    def __init__(self,
                 filename: str = '',
                 **kwargs):
        """When initialising an ECG, the minimum requirement is to pass the name for the file when opening

        Parameters
        ----------
        filename : str
            Location of data
        normalise : str
            Whether or not to normalise all ECG leads, default=False
        dt : float
            Time interval between recording data
        electrode_file : str
            File containing the indentifiers for the electrode ECG placement. Only useful if `filename` refers to a
            .igb data file for a whole torso simulation, and the ECG needs to be derived from these data
        """
        # Properties that can be derived subsequently to opening the file
        self.n_beats = 0
        self.n_beats_threshold = 0.5
        self.beats = list()

        self.qrs_start = list()
        self.qrs_end = list()
        self.twave_end = list()

        self.data_source = None
        self.comments = None

        # Minimum requirements for an ECG - the raw data, and the file from which it is derived
        self.data = pd.DataFrame()
        self.filename = filename
        self.normalised = False
        if self.filename:
            self.read(filename, **kwargs)
            self.get_n_beats()

    def read(self,
             filename: str,
             normalise: bool = False,
             **kwargs):
        if filename.endswith("igb"):
            self.data = read_ecg_from_igb(filename, normalise=normalise, **kwargs)
        elif filename.endswith("csv"):
            self.data = read_ecg_from_csv(filename, normalise=normalise)
        elif filename.endswith("dat"):
            self.data = read_ecg_from_dat(filename, normalise=normalise)
        else:
            self.read_ecg_from_wfdb(filename, normalise=normalise)
        self.normalised = normalise

        # Reset all other values to zero to prevent confusion if reading new data to an existing structure,
        # for some unknown reason
        self.reset()

    def reset(self):
        """Reset all properties of the class"""
        self.n_beats = 0
        self.qrs_start = list()
        self.qrs_end = list()
        self.data_source = None
        self.comments = None

    def read_ecg_from_wfdb(self,
                           filename: str,
                           normalise: bool = False):
        data_full = wfdb.rdrecord(filename)

        # Reformat the column names according to existing data patterns
        columns_full = [w.replace('iii', 'LIII') for w in data_full.sig_name]
        columns_full = [w.replace('ii', 'LII') for w in columns_full]
        columns_full = [w.replace('i', 'LI') for w in columns_full]
        columns_full = [w.replace('avr', 'aVR') for w in columns_full]
        columns_full = [w.replace('avl', 'aVL') for w in columns_full]
        columns_full = [w.replace('avf', 'aVF') for w in columns_full]
        columns_full = [w.replace('v1', 'V1') for w in columns_full]
        columns_full = [w.replace('v2', 'V2') for w in columns_full]
        columns_full = [w.replace('v3', 'V3') for w in columns_full]
        columns_full = [w.replace('v4', 'V4') for w in columns_full]
        columns_full = [w.replace('v5', 'V5') for w in columns_full]
        columns_full = [w.replace('v6', 'V6') for w in columns_full]

        if 'lobachevsky' in filename:
            sample_rate = 500
        else:
            warnings.warn('Unknown sample rate for data - assuming 500 Hz')
            sample_rate = 500
        interval = 1/sample_rate
        end_val = data_full.p_signal.shape[0]*interval
        t = np.arange(0, end_val, interval)

        data_temp = pd.DataFrame(data=data_full.p_signal, columns=columns_full, index=t)
        if normalise:
            self.data = data_temp/data_temp.abs().max()
        else:
            self.data = pd.DataFrame(data=data_full.p_signal, columns=columns_full, index=t)

    def get_n_beats(self,
                    threshold: float = 0.5,
                    min_separation: float = 0.2,
                    unipolar_only: bool = True,
                    plot: bool = False):
        """Calculate the number of beats in an ECG trace, and save the individual beats to file for later use

        When given the raw data of an ECG trace, will estimate the number of beats recorded in the trace based on the
        RMS of the ECG signal exceeding a threshold value. The estimated individual beats will then be saved in a
        list in a lossless manner, i.e. saved as [ECG1, ECG2, ..., ECG(n)], where ECG1=[0:peak2], ECG2=[peak1:peak3],
        ..., ECGn=[peak(n-1):end]

        Parameters
        ----------
        threshold : float {0<1}
            Minimum value to search for for a peak in RMS signal to determine when a beat has occurred, default=0.5
        min_separation : float
            Minimum time (in s) that should be used to separate separate beats, default=0.2s
        unipolar_only : bool, optional
            Whether to use only unipolar ECG leads to calculate RMS, default=True
        plot : bool
            Whether to plot results of beat detection, default=False

        Returns
        -------
        self.n_beats : int
            Number of beats detected in signal

        Notes
        -----
        The scalar RMS is calculated according to

        .. math:: \sqrt{\frac{1}{n}\sum_{i=1}^n (\textnormal{ECG}_i^2(t))}

        for all leads available from the signal (12 for ECG, 3 for VCG). If unipolar_only is set to true, then ECG RMS
        is calculated using only 'unipolar' leads. This uses V1-6, and the non-augmented limb leads (VF, VL and VR)

        ..math:: VF = LL-V_{WCT} = \frac{2}{3}aVF
        ..math:: VL = LA-V_{WCT} = \frac{2}{3}aVL
        ..math:: VR = RA-V_{WCT} = \frac{2}{3}aVR
        """

        # Calculate locations of RMS peaks to determine number and locations of beats
        rms = signalanalysis.general.get_signal_rms(self.data, unipolar_only=unipolar_only)
        i_separation = self.data.index.get_loc(min_separation)
        i_peaks, _ = scipy.signal.find_peaks(rms, height=threshold*max(rms), distance=i_separation)
        self.n_beats = len(i_peaks)
        self.n_beats_threshold = threshold

        # Split the trace into individual beats
        t_peaks = rms.index[i_peaks]
        t_split = [self.data.index[0]]+list(t_peaks)+[self.data.index[-1]]
        for i_split in range(self.n_beats):
            self.beats.append(self.data.loc[t_split[i_split]:t_split[i_split+2], :])

        if plot:
            fig = plt.figure()
            ax = fig.add_subplot(1, 1, 1)
            ax.plot(rms)
            ax.plot(t_peaks, rms[t_peaks], 'o', markerfacecolor='none')

    def get_qrs_start(self,
                      unipolar_only: bool = True,
                      min_separation: float = 0.05,
                      plot_result: bool = False):
        """Calculates start of QRS complex using method of Hermans et al. (2017)

        Calculates the start of the QRS complex by a simplified version of the work presented in [1]_, wherein the
        point of maximum second derivative of the ECG RMS signal is used as the start of the QRS complex

        Parameters
        ----------
        self : Ecg
            ECG data to analyse
        unipolar_only : bool, optional
            Whether to use only unipolar leads to calculate RMS, default=True
        min_separation : float, optional
            Minimum separation from the peak used to detect various beats, default=0.05s
        plot_result : bool, optional
            Whether to plot the results for error-checking, default=False

        Returns
        -------
        self.qrs_start : list of float
            QRS start times

        Notes
        -----
        For further details of the action of unipolar_only, see general_analysis.get_signal_rms

        It is faster to use scipy.ndimage.laplace() rather than np.gradient(np.gradient)), but preliminary checks
        indicated some edge problems that might throw off the results.

        References
        ----------
        .. [1] Hermans BJM, Vink AS, Bennis FC, Filippini LH, Meijborg VMF, Wilde AAM, Pison L, Postema PG, Delhaas T,
               "The development and validation of an easy to use automatic QT-interval algorithm,"
               PLoS ONE, 12(9), 1–14 (2017), https://doi.org/10.1371/journal.pone.0184352
        """

        # self.qrs_start = get_qrs_start(self.data, unipolar_only=unipolar_only, plot_result=plot_result)[0]

        # If individual beats not yet separated, do so now
        if not self.beats:
            self.get_n_beats()

        # Remove the requested sections from the beginning/end of the individual beat traces (which are originally
        # lossless, so extend to the peak of the prior and following beat RMS data)
        beats_short = self.beats.copy()
        for i_beat in range(1, len(beats_short)-1):
            beat_short = beats_short[i_beat]
            beat_start = beat_short.index[0]+min_separation
            beat_end = beat_short.index[-1]-min_separation
            i_start = np.argmin(abs(beat_short.index-beat_start).values)
            i_end = np.argmin(abs(beat_short.index-beat_end).values)
            beats_short[i_beat] = beats_short[i_beat].iloc[i_start:i_end, :]

        # Perform QRS detection on the individual beats
        ecgs_rms = [signalanalysis.general.get_signal_rms(beat_short, unipolar_only=unipolar_only) for beat_short in
                    beats_short]
        ecgs_grad = [pd.Series(np.gradient(np.gradient(ecg_rms)), index=ecg_rms.index) for ecg_rms in ecgs_rms]
        self.qrs_start = [ecg_grad[ecg_grad == ecg_grad.max()].index[0] for ecg_grad in ecgs_grad]

        if plot_result:
            for (ecg_rms, ecg_grad, qrs_start) in zip(ecgs_rms, ecgs_grad, self.qrs_start):
                fig, ax = plt.subplots(2, 1, sharex='all')
                ax[0].plot(ecg_rms)
                ax[0].set_ylabel('ECG_{RMS}')
                ax[1].plot(ecg_grad)
                ax[1].set_ylabel('ECG Sec Der')
                ax[0].axvline(qrs_start, color='k', linestyle='--')
                ax[1].axvline(qrs_start, color='k', linestyle='--')


def read_ecg_from_igb(filename: str,
                      electrode_file: Optional[str] = None,
                      normalise: bool = False,
                      dt: float = 2) -> pd.DataFrame:
    """Translate the phie.igb file(s) to 10-lead, 12-trace ECG data

    Extracts the complete mesh data from the phie.igb file using CARPutils, which contains the data for the body
    surface potential for an entire human torso, before then extracting only those nodes that are relevant to the
    12-lead ECG, before converting to the ECG itself
    https://carpentry.medunigraz.at/carputils/generated/carputils.carpio.igb.IGBFile.html#carputils.carpio.igb.IGBFile

    Parameters
    ----------
    filename : str
        Filename for the phie.igb data to extract
    electrode_file : str, optional
        File which contains the node indices in the mesh that correspond to the placement of the leads for the
        10-lead ECG. Default given in get_electrode_phie function.
    normalise : bool, optional
        Whether or not to normalise the ECG signals on a per-lead basis, default=False
    dt : float, optional
        Time interval from which to construct the time data to associate with the ECG, default=2

    Returns
    -------
    ecgs : pd.DataFrame
        DataFrame with Vm data for each of the labelled leads (the dictionary keys are the names of the leads)
    """

    data, _, _ = igb.read(filename)
    electrode_data = get_electrode_phie(data, electrode_file)
    ecg_data = get_ecg_from_electrodes(electrode_data)

    # Add time data
    ecg_data['t'] = [i*dt for i in range(len(ecg_data))]
    ecg_data.set_index('t', inplace=True)

    if normalise:
        return tools.maths.normalise_signal(ecg_data)
    else:
        return ecg_data


def read_ecg_from_dat(filename: str,
                      normalise: bool = False) -> pd.DataFrame:
    """Read ECG data from .dat file

    Parameters
    ----------
    filename : str
        Name/location of the .dat file to read
    normalise : bool, optional
        Whether or not to normalise the ECG signals on a per-lead basis, default=False

    Returns
    -------
    ecg : pd.DataFrame
        Extracted data for the 12-lead ECG
    """
    ecgdata = np.loadtxt(filename, dtype=float)

    ecg = pd.DataFrame()
    # Limb Leads
    ecg['LI'] = ecgdata[:, 1]
    ecg['LII'] = ecgdata[:, 2]
    ecg['LIII'] = ecgdata[:, 3]
    # Augmented leads
    ecg['aVR'] = ecgdata[:, 4]
    ecg['aVL'] = ecgdata[:, 5]
    ecg['aVF'] = ecgdata[:, 6]
    # Precordeal leads
    ecg['V1'] = ecgdata[:, 7]
    ecg['V2'] = ecgdata[:, 8]
    ecg['V3'] = ecgdata[:, 9]
    ecg['V4'] = ecgdata[:, 10]
    ecg['V5'] = ecgdata[:, 11]
    ecg['V6'] = ecgdata[:, 12]

    ecg['t'] = ecgdata[:, 0]
    ecg.set_index('t', inplace=True)

    if normalise:
        ecg = tools.maths.normalise_signal(ecg)

    return ecg


def read_ecg_from_csv(filename: str,
                      normalise: bool = False) -> pd.DataFrame:
    """Extract ECG data from CSV file exported from St Jude Medical ECG recording

    Parameters
    ----------
    filename : str
        Name/location of the .dat file to read
    normalise : bool, optional
        Whether or not to normalise the ECG signals on a per-lead basis, default=True

    Returns
    -------
    ecg : list of pd.DataFrame
        Extracted data for the 12-lead ECG
    """
    line_count = 0
    with open(filename, 'r') as pFile:
        while True:
            line_count += 1
            line = pFile.readline()
            if 'number of samples' in line.lower():
                n_rows = int(re.search(r'\d+', line).group())
                break
            if not line:
                raise EOFError('Number of Samples entry not found - check file input')
    ecgdata = pd.read_csv(filename, skiprows=line_count, index_col=False)
    ecgdata.drop(ecgdata.tail(1).index, inplace=True)
    n_rows_read, _ = ecgdata.shape
    assert n_rows_read == n_rows, "Mismatch between expected data and read data"

    ecg = pd.DataFrame()
    # Limb Leads
    ecg['LI'] = ecgdata['I'].values
    ecg['LII'] = ecgdata['II'].values
    ecg['LIII'] = ecgdata['III'].values
    # Augmented leads
    ecg['aVR'] = ecgdata['aVR'].values
    ecg['aVL'] = ecgdata['aVL'].values
    ecg['aVF'] = ecgdata['aVF'].values
    # Precordeal leads
    ecg['V1'] = ecgdata['V1'].values
    ecg['V2'] = ecgdata['V2'].values
    ecg['V3'] = ecgdata['V3'].values
    ecg['V4'] = ecgdata['V4'].values
    ecg['V5'] = ecgdata['V5'].values
    ecg['V6'] = ecgdata['V6'].values

    ecg['t'] = ecgdata['t_ref'].values
    ecg.set_index('t', inplace=True)

    if normalise:
        ecg = tools.maths.normalise_signal(ecg)

    return ecg


def get_electrode_phie(phie_data: np.ndarray, electrode_file: Optional[str] = None) -> pd.DataFrame:
    """Extract phi_e data corresponding to ECG electrode locations

    Parameters
    ----------
    phie_data : np.ndarray
        Numpy array that holds all phie data for all nodes in a given mesh
    electrode_file : str, optional
        File containing entries corresponding to the nodes of the mesh which determine the location of the 10 leads
        for the ECG. Will default to very project specific location. The input text file has each node on a separate
        line (zero-indexed), with the node locations given in order: V1, V2, V3, V4, V5, V6, RA, LA, RL,
        LL. Will default to '12LeadElectrodes.dat', but this is almost certainly not going to right for an individual
        project

    Returns
    -------
    electrode_data : pd.DataFrame
        Dataframe of phie data for each node, with the dictionary key labelling which node it is.
    """

    # Import default arguments
    if electrode_file is None:
        electrode_file = '12LeadElectrodes.dat'

    # Extract node locations for ECG data, then pull data corresponding to those nodes
    pts_electrodes = np.loadtxt(electrode_file, usecols=(1,), dtype=int)

    electrode_data = pd.DataFrame({'V1': phie_data[pts_electrodes[0], :],
                                   'V2': phie_data[pts_electrodes[1], :],
                                   'V3': phie_data[pts_electrodes[2], :],
                                   'V4': phie_data[pts_electrodes[3], :],
                                   'V5': phie_data[pts_electrodes[4], :],
                                   'V6': phie_data[pts_electrodes[5], :],
                                   'RA': phie_data[pts_electrodes[6], :],
                                   'LA': phie_data[pts_electrodes[7], :],
                                   'RL': phie_data[pts_electrodes[8], :],
                                   'LL': phie_data[pts_electrodes[9], :]})

    return electrode_data


def get_ecg_from_electrodes(electrode_data: pd.DataFrame) -> pd.DataFrame:
    """Converts electrode phi_e data to ECG lead data

    Takes dictionary of phi_e data for 10-lead ECG, and converts these data to standard ECG trace data

    Parameters
    ----------
    electrode_data : pd.DataFrame
        Dictionary with keys corresponding to lead locations

    Returns
    -------
    ecg : pd.DataFrame
        Dictionary with keys corresponding to the ECG traces
    """

    # Wilson Central Terminal
    wct = (electrode_data['LA'] + electrode_data['RA'] + electrode_data['LL']) / 3

    # V leads
    ecg = pd.DataFrame()
    ecg['V1'] = electrode_data['V1'] - wct
    ecg['V2'] = electrode_data['V2'] - wct
    ecg['V3'] = electrode_data['V3'] - wct
    ecg['V4'] = electrode_data['V4'] - wct
    ecg['V5'] = electrode_data['V5'] - wct
    ecg['V6'] = electrode_data['V6'] - wct

    # Eindhoven limb leads
    ecg['LI'] = electrode_data['LA'] - electrode_data['RA']
    ecg['LII'] = electrode_data['LL'] - electrode_data['RA']
    ecg['LIII'] = electrode_data['LL'] - electrode_data['LA']

    # Augmented leads
    ecg['aVR'] = electrode_data['RA'] - 0.5 * (electrode_data['LA'] + electrode_data['LL'])
    ecg['aVL'] = electrode_data['LA'] - 0.5 * (electrode_data['RA'] + electrode_data['LL'])
    ecg['aVF'] = electrode_data['LL'] - 0.5 * (electrode_data['LA'] + electrode_data['RA'])

    return ecg


def get_qrs_start(ecgs: Union[pd.DataFrame, List[pd.DataFrame]],
                  unipolar_only: bool = True,
                  plot_result: bool = False) -> List[float]:
    """Calculates start of QRS complex using method of Hermans et al. (2017)

    Calculates the start of the QRS complex by a simplified version of the work presented in [1]_, wherein the point of
    maximum second derivative of the ECG RMS signal is used as the start of the QRS complex

    Parameters
    ----------
    ecgs : pd.DataFrame or list of pd.DataFrame
        ECG data to analyse
    unipolar_only : bool, optional
        Whether to use only unipolar leads to calculate RMS, default=True
    plot_result : bool, optional
        Whether to plot the results for error-checking, default=False

    Returns
    -------
    qrs_starts : list of float
        QRS start times

    Notes
    -----
    For further details of the action of unipolar_only, see general_analysis.get_signal_rms

    It is faster to use scipy.ndimage.laplace() rather than np.gradient(np.gradient)), but preliminary checks
    indicated some edge problems that might throw off the results.

    References
    ----------
    .. [1] Hermans BJM, Vink AS, Bennis FC, Filippini LH, Meijborg VMF, Wilde AAM, Pison L, Postema PG, Delhaas T,
           "The development and validation of an easy to use automatic QT-interval algorithm,"
           PLoS ONE, 12(9), 1–14 (2017), https://doi.org/10.1371/journal.pone.0184352"""

    if isinstance(ecgs, pd.DataFrame):
        ecgs = [ecgs]

    ecgs_rms = [signalanalysis.general.get_signal_rms(ecg, unipolar_only=unipolar_only) for ecg in ecgs]
    ecgs_grad = [pd.Series(np.gradient(np.gradient(ecg_rms)), index=ecg_rms.index) for ecg_rms in ecgs_rms]
    qrs_starts = [ecg_grad[ecg_grad == ecg_grad.max()].index[0] for ecg_grad in ecgs_grad]

    if plot_result:
        import matplotlib.pyplot as plt
        for (ecg_rms, ecg_grad, qrs_start) in zip(ecgs_rms, ecgs_grad, qrs_starts):
            fig, ax = plt.subplots(2, 1, sharex='all')
            ax[0].plot(ecg_rms)
            ax[0].set_ylabel('ECG_{RMS}')
            ax[1].plot(ecg_grad)
            ax[1].set_ylabel('ECG Sec Der')
            ax[0].axvline(qrs_start, color='k', linestyle='--')
            ax[1].axvline(qrs_start, color='k', linestyle='--')

    return qrs_starts
