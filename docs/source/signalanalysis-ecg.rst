============
ECG Analysis
============

.. _ecg-reading:

Reading ECG data
----------------

Currently, only ECG reading is supported; VCG data is calculated from ECG data using the Kors method (see method).
ECG files are read upon the instantiation of the ECG class, though it is possible to re-read data if required for
some reason.

.. code-block:: python3

    >>> import signalanalysis as sa
    >>> ecg_example = sa.signalanalysis.ecg.Ecg("filename")
    >>> vcg_example = sa.signalanalysis.vcg.Vcg(ecg_example)

Currently, sample data are included in the GitHub archive - it is hoped that these will be freely available, but it
unfortunately cannot be guaranteed. These can be loaded as follows:

.. code-block:: python3

    >>> import signalanalysis as sa
    >>> from signalanalysis.data import datafiles
    >>> ecg_lob = sa.signalanalysis.ecg.Ecg(
            datafiles.LOBACHEVSKY,
            sample_rate=500,
        )
    >>> ecg_ptb_100 = sa.signalanalysis.ecg.Ecg(
            datafiles.PTB_100,
            sample_rate=100,
            comments_file=datafiles.PTB_DATABASE,
        )
    >>> ecg_ptb_500 = sa.signalanalysis.ecg.Ecg(
            datafiles.PTB_500,
            sample_rate=500,
            comments_file=datafiles.PTB_DATABASE,
        )

These sample files are downloaded from PhysioNet, and represent only a fraction of the available data.
