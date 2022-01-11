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

    >>> from signalanalysis import signalanalysis.ecg
    >>> from signalanalysis import signalanalysis.vcg
    >>> ecg_example = signalanalysis.ecg.Ecg("filename")
    >>> vcg_example = signalanalysis.vcg.Vcg(ecg_example)

Currently, sample data are included in the GitHub archive - it is hoped that these will be freely available, but it
unfortunately cannot be guaranteed. These can be loaded as follows:

.. code-block:: python3

    >>> ecg_example_lob = signalanalysis.ecg.Ecg("tests/lobachevsky/3")
    >>> test_ecg_ptb100 = signalanalysis.ecg.Ecg('tests/ptb-xl/records100/00000/00001_lr', \
                                                 sample_rate=100, \
                                                 comments_file='tests/ptb-xl/ptbxl_database.csv')
    >>> test_ecg_ptb500 = signalanalysis.ecg.Ecg('tests/ptb-xl/records500/00000/00001_hr', \
                                                 sample_rate=500, \
                                                 comments_file='tests/ptb-xl/ptbxl_database.csv')

These sample files are downloaded from PhysioNet, and represent only a fraction of the available data.