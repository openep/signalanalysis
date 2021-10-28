import ast
import re
import numpy as np
import pandas as pd
import wfdb
from typing import List, Union
import matplotlib.pyplot as plt

from carputils.carpio import igb  # type: ignore

import tools.maths
import signalanalysis.general
import signalplot.ecg

plt.style.use('seaborn')


class Ecg(signalanalysis.general.Signal):
    """Base class to encapsulate data regarding an ECG recording, inheriting from :class:`signalanalysis.general.Signal`

    See Also
    --------
    :py:class:`signalanalysis.general.Signal`

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
                 filename: str,
                 **kwargs):
        """When initialising an ECG, the minimum requirement is to pass the name for the file when opening

        Parameters
        ----------
        filename : str
            Location of data

        Other Parameters
        ----------------
        electrode_file : str
            File containing the identifiers for the electrode ECG placement. Required if and only if `filename` refers
            to a .igb data file for a whole torso simulation, and the ECG needs to be derived from these data
        dt : float
            Time interval between recording data

        See Also
        --------
        :py:meth:`signalanalysis.general.Signal.__init__` : Base initialisation method
        :py:meth:`signalanalysis.ecg.Ecg.read` : Reads the data from the file into the object
        """
        super(Ecg, self).__init__(**kwargs)

        # Minimum requirements for an ECG - the raw data, and the file from which it is derived
        self.filename = filename
        self.read(filename, **kwargs)
        if self.filter is not None:
            self.apply_filter(**kwargs)
        self.get_n_beats()

    def read(self,
             filename: str,
             normalise: bool = False,
             **kwargs):
        """Reads in the data from the original file. Called upon initialisation

        Parameters
        ----------
        filename : str
            Location for data (either filename or directory)
        normalise : bool, optional
            Whether or not to normalise the individual leads (no real biophysical rationale for doing this,
            to be honest), default=False

        See Also
        --------
        signalanalysis.ecg.Ecg.read_ecg_from_wfdb : underlying read method for wfdb files
        """

        # Reset all other values to zero to prevent confusion if reading new data to an existing structure,
        # for some unknown reason
        super().reset()

        if filename.endswith("igb"):
            self.data = read_ecg_from_igb(filename, normalise=normalise, **kwargs)
        elif filename.endswith("csv"):
            self.data = read_ecg_from_csv(filename, normalise=normalise)
        elif filename.endswith("dat"):
            self.data = read_ecg_from_dat(filename, normalise=normalise)
        else:
            self.read_ecg_from_wfdb(filename, normalise=normalise, **kwargs)
        self.normalised = normalise

    def read_ecg_from_wfdb(self,
                           filename: str,
                           sample_rate: float,
                           comments_file: str = None,
                           normalise: bool = False):
        """Read data from a waveform database file format

        Parameters
        ----------
        filename : str
            Base filename for data (e.g. /path/to/data/1 will read /path/to/data/1.{avf, avl, avr, dat, hea}
        sample_rate : float
            Rate at which data is recorded, e.g. 500 Hz
        comments_file : str, optional
            .csv file containing additional data for the ECG, if available
        normalise : bool, optional
            Whether to normalise all individual leads, default=False

        Notes
        -----
        The waveform database format is used by several different ECG repositories on www.physionet.org,
        and the finer points of each import can be difficult to seamlessly integrate. Where values are subject to
        change between datasets, they are not available as defaults. The required values are thus, but it should be
        remembered that there remains a certain level of hard-coding (for example, in the use of `comments_file` to
        extract data).

        * Lobachevsky

            * sample_rate = 500

        * PTB-XL

            * sample_rate = 100 if from records100/*/*_lr
            * sample_rate = 500 if from records500/*/*_hr
            * comments_file = 'ptbxl_database.csv'
        """
        data_full = wfdb.rdrecord(filename)

        # Reformat the column names according to existing data patterns
        # Potential values for each lead
        lead_poss = dict()
        lead_poss['LI'] = ['LI', 'li', 'I', 'i']
        lead_poss['LII'] = ['LII', 'lii', 'II', 'ii']
        lead_poss['LIII'] = ['LIII', 'liii', 'III', 'iii']
        lead_poss['aVR'] = ['aVR', 'avr', 'AVR']
        lead_poss['aVL'] = ['aVL', 'avl', 'AVL']
        lead_poss['aVF'] = ['aVF', 'avf', 'AVF']
        lead_poss['V1'] = ['V1', 'v1']
        lead_poss['V2'] = ['V2', 'v2']
        lead_poss['V3'] = ['V3', 'v3']
        lead_poss['V4'] = ['V4', 'v4']
        lead_poss['V5'] = ['V5', 'v5']
        lead_poss['V6'] = ['V6', 'v6']
        columns_full = data_full.sig_name
        for key in lead_poss:
            # Find which of the possible entries matches the given data, then find the relevant index before
            # replacing with the accepted value
            lead_match = [temp_lead for temp_lead in lead_poss[key] if temp_lead in columns_full]
            assert len(lead_match) == 1, "Too many lead matches found!"
            lead_index = columns_full.index(lead_match[0])
            columns_full[lead_index] = key

        interval = (1 / sample_rate)*1000
        end_val = data_full.p_signal.shape[0] * interval
        t = np.arange(0, end_val, interval)

        data_temp = pd.DataFrame(data=data_full.p_signal, columns=columns_full, index=t)
        if normalise:
            self.data = data_temp / data_temp.abs().max()
        else:
            self.data = data_temp

        if comments_file is not None:
            comments_data = pd.read_csv(comments_file, index_col='ecg_id')
            comments_data.scp_codes = comments_data.scp_codes.apply(lambda x: ast.literal_eval(x))
            if '_hr' in filename:
                filename_key = 'filename_hr'
            elif '_lr' in filename:
                filename_key = 'filename_lr'
            else:
                raise ValueError("filename isn't working here...")
            temp = [temp_name for temp_name in comments_data[filename_key] if temp_name in filename]
            self.comments.append(comments_data.loc[comments_data[filename_key] == temp[0]])
        self.comments.append(data_full.comments)

    def get_rms(self,
                preprocess_data: pd.DataFrame = None,
                drop_columns: List[str] = None,
                unipolar_only: bool = True):
        """Supplement the :meth:`signalanalysis.general.Signal.get_rms` with `unipolar_only`

        Parameters
        ----------
        preprocess_data : pd.DataFrame, optional
            See :meth:`signalanalysis.general.Signal.get_rms`
        drop_columns : list of str, optional
            See :meth:`signalanalysis.general.Signal.get_rms`
        unipolar_only : optional
            Whether to use only unipolar ECG leads to calculate RMS, default=True

        See Also
        --------
        :py:meth:`signalanalysis.general.Signal.get_rms`

        Notes
        -----
        If `unipolar_only` is set to true, then ECG RMS is calculated using only 'unipolar' leads. This uses V1-6,
        and the non-augmented limb leads (VF, VL and VR)

        .. math::
            VF = LL-V_{WCT} = \\frac{2}{3}aVF
        .. math::
            VL = LA-V_{WCT} = \\frac{2}{3}aVL
        .. math::
            VR = RA-V_{WCT} = \\frac{2}{3}aVR
        """
        signal_rms = self.data.copy()
        if unipolar_only and ('V1' in signal_rms.columns):
            signal_rms['VF'] = (2 / 3) * signal_rms['aVF']
            signal_rms['VL'] = (2 / 3) * signal_rms['aVL']
            signal_rms['VR'] = (2 / 3) * signal_rms['aVR']
            signal_rms.drop(['aVF', 'aVL', 'aVR', 'LI', 'LII', 'LIII'], axis=1, inplace=True)
        super(Ecg, self).get_rms(signal_rms, drop_columns=drop_columns)

    def get_qrs_start(self,
                      unipolar_only: bool = True,
                      min_separation: float = 50,
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
            Minimum separation from the peak used to detect various beats, default=50ms
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
        for i_beat in range(1, len(beats_short) - 1):
            beat_short = beats_short[i_beat]
            beat_start = beat_short.index[0] + min_separation
            beat_end = beat_short.index[-1] - min_separation
            i_start = np.argmin(abs(beat_short.index - beat_start).values)
            i_end = np.argmin(abs(beat_short.index - beat_end).values)
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

    def plot(self,
             separate_beats: bool = False,
             separate_figs: bool = False):
        """Method to plot the data of the ECG

        Passes the function to the ECG plotting script, with the option to plot the individual beats instead of the
        entire signal; also, to plot these on individual figures or overlaid on the same figure

        Parameters
        ----------
        separate_beats : bool, optional
            Whether to plot the entire signal (false), or to plot the individual detected beats one after the other,
            default=False
        separate_figs : bool, optional
            When plotting the separate beats (if requested), whether to plot the individual beats one on top of the
            other on a single figure (true), or on separate figures (false), default=False

        Returns
        -------
        fig, ax
        """
        if separate_beats:
            fig, ax = signalplot.ecg.plot(self.data)
        else:
            if separate_figs:
                fig = list()
                ax = list()
                for beat in self.beats:
                    fig_temp, ax_temp = signalplot.ecg.plot(beat)
                    fig.append(fig_temp)
                    ax.append(ax_temp)
            else:
                fig, ax = signalplot.ecg.plot(self.beats)

        return fig, ax


def read_ecg_from_igb(filename: str,
                      electrode_file,
                      dt: float,
                      normalise: bool = False) -> pd.DataFrame:
    """Translate the phie.igb file(s) to 10-lead, 12-trace ECG data

    Extracts the complete mesh data from the phie.igb file using CARPutils, which contains the data for the body
    surface potential for an entire human torso, before then extracting only those nodes that are relevant to the
    12-lead ECG, before converting to the ECG itself

    Parameters
    ----------
    filename : str
        Filename for the phie.igb data to extract
    electrode_file : str
        File which contains the node indices in the mesh that correspond to the placement of the leads for the
        10-lead ECG. Default given in get_electrode_phie function.
    dt : float
        Time interval from which to construct the time data to associate with the ECG
    normalise : bool, optional
        Whether or not to normalise the ECG signals on a per-lead basis, default=False

    Returns
    -------
    ecgs : pd.DataFrame
        DataFrame with Vm data for each of the labelled leads (the dictionary keys are the names of the leads)

    Notes
    -----
    For the .igb data used thus far, the `electrode_file` can be found at ``tests/12LeadElectrodes.dat``, and `dt` is
    2ms

    References
    ----------
    https://carpentry.medunigraz.at/carputils/generated/carputils.carpio.igb.IGBFile.html#carputils.carpio.igb.IGBFile
    """

    data, _, _ = igb.read(filename)
    electrode_data = get_electrode_phie(data, electrode_file)
    ecg_data = get_ecg_from_electrodes(electrode_data)

    # Add time data
    ecg_data['t'] = [i * dt for i in range(len(ecg_data))]
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

    Notes
    -----
    Relies on the first line of the .csv labelling the 12 leads of the ECG, in the form ['I', 'II', 'III',  'aVR',
    'aVL', 'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6'], and the time data under the label 't_ref'
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


def get_electrode_phie(phie_data: np.ndarray,
                       electrode_file: str) -> pd.DataFrame:
    """Extract phi_e data corresponding to ECG electrode locations

    Parameters
    ----------
    phie_data : np.ndarray
        Numpy array that holds all phie data for all nodes in a given mesh
    electrode_file : str
        File containing entries corresponding to the nodes of the mesh which determine the location of the 10 leads
        for the ECG. Will default to very project specific location. The input text file has each node on a separate
        line (zero-indexed), with the node locations given in order: V1, V2, V3, V4, V5, V6, RA, LA, RL,
        LL.

    Returns
    -------
    electrode_data : pd.DataFrame
        Dataframe of phie data for each node, with the dictionary key labelling which node it is.

    Notes
    -----
    For the .igb data used thus far, the `electrode_file` can be found at ``tests/12LeadElectrodes.dat``
    """

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
        for (ecg_rms, ecg_grad, qrs_start) in zip(ecgs_rms, ecgs_grad, qrs_starts):
            fig, ax = plt.subplots(2, 1, sharex='all')
            ax[0].plot(ecg_rms)
            ax[0].set_ylabel('ECG_{RMS}')
            ax[1].plot(ecg_grad)
            ax[1].set_ylabel('ECG Sec Der')
            ax[0].axvline(qrs_start, color='k', linestyle='--')
            ax[1].axvline(qrs_start, color='k', linestyle='--')

    return qrs_starts
