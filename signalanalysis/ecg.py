import re
import numpy as np
import pandas as pd
from typing import List, Union, Optional

from carputils.carpio import igb  # type: ignore

import tools.maths
import signalanalysis.general


def read_ecg_from_igb(phie_file: Union[List[str], str],
                      electrode_file: Optional[str] = None,
                      normalise: bool = True,
                      dt: float = 2) -> List[pd.DataFrame]:
    """Translate the phie.igb file(s) to 10-lead, 12-trace ECG data

    Extracts the complete mesh data from the phie.igb file using CARPutils, before then extracting only those nodes that
    are relevant to the 12-lead ECG, before converting to the ECG itself
    https://carpentry.medunigraz.at/carputils/generated/carputils.carpio.igb.IGBFile.html#carputils.carpio.igb.IGBFile

    Parameters
    ----------
    phie_file : list or str
        Filename for the phie.igb data to extract
    electrode_file : str, optional
        File which contains the node indices in the mesh that correspond to the placement of the leads for the
        10-lead ECG. Default given in get_electrode_phie function.
    normalise : bool, optional
        Whether or not to normalise the ECG signals on a per-lead basis, default=True
    dt : float, optional
        Time interval from which to construct the time data to associate with the ECG, default=2

    Returns
    -------
    ecgs : list(dict)
        List of dictionaries with Vm data for each of the labelled leads (the dictionary keys are the names of the
        leads)
    """

    if isinstance(phie_file, str):
        phie_file = [phie_file]
    data = [data_tmp for data_tmp, _, _ in (igb.read(filename) for filename in phie_file)]

    electrode_data = [get_electrode_phie(data_tmp, electrode_file) for data_tmp in data]

    ecgs = [get_ecg_from_electrodes(elec_tmp) for elec_tmp in electrode_data]

    # Add time data
    for ecg in ecgs:
        ecg['t'] = [i*dt for i in range(len(ecg))]
        ecg.set_index('t', inplace=True)

    if normalise:
        return [tools.maths.normalise_signal(sim_ecg) for sim_ecg in ecgs]
    else:
        return ecgs


def read_ecg_from_dat(ecg_files: Union[List[str], str],
                      normalise: bool = True) -> List[pd.DataFrame]:
    """Read ECG data from .dat file

    Parameters
    ----------
    ecg_files : str or list of str
        Name/location of the .dat file to read
    normalise : bool, optional
        Whether or not to normalise the ECG signals on a per-lead basis, default=True

    Returns
    -------
    ecg : dict
        Extracted data for the 12-lead ECG
    times : np.ndarray
        Time data associated with the ECG data
    """
    if isinstance(ecg_files, str):
        ecg_files = [ecg_files]

    ecgs = list()
    for ecg_file in ecg_files:
        ecgdata = np.loadtxt(ecg_file, dtype=float)

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

        ecgs.append(ecg)

    if normalise:
        ecgs = [tools.maths.normalise_signal(ecg) for ecg in ecgs]

    return ecgs


def read_ecg_from_csv(ecg_files: Union[List[str], str],
                      normalise: bool = True) -> List[pd.DataFrame]:
    """Extract ECG data from CSV file exported from St Jude Medical ECG recording

    Parameters
    ----------
    ecg_files : str or list of str
        Name/location of the .dat file to read
    normalise : bool, optional
        Whether or not to normalise the ECG signals on a per-lead basis, default=True

    Returns
    -------
    ecg : list of pd.DataFrame
        Extracted data for the 12-lead ECG
    """
    if isinstance(ecg_files, str):
        ecg_files = [ecg_files]

    ecgs = list()
    for ecg_file in ecg_files:
        line_count = 0
        with open(ecg_file, 'r') as pFile:
            while True:
                line_count += 1
                line = pFile.readline()
                if 'number of samples' in line.lower():
                    n_rows = int(re.search(r'\d+', line).group())
                    break
                if not line:
                    raise EOFError('Number of Samples entry not found - check file input')
        ecgdata = pd.read_csv(ecg_file, skiprows=line_count, index_col=False)
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

        ecgs.append(ecg)

    if normalise:
        ecgs = [tools.maths.normalise_signal(ecg) for ecg in ecgs]

    return ecgs


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
        LL. Will default to '/home/pg16/Documents/ecg-scar/ecg-analysis/12LeadElectrodes.dat'

    Returns
    -------
    electrode_data : pd.DataFrame
        Dataframe of phie data for each node, with the dictionary key labelling which node it is.

    """

    # Import default arguments
    if electrode_file is None:
        electrode_file = '/home/pg16/Documents/ecg-scar/ecg-analysis/12LeadElectrodes.dat'

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

    ecgs_rms = signalanalysis.general.get_signal_rms(ecgs, unipolar_only=unipolar_only)
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

