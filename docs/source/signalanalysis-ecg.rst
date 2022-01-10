============
ECG Analysis
============

.. _reading:

Reading ECG data
----------------

Currently, only ECG reading is supported; VCG data is calculated from ECG data using the Kors method (see method).
ECG files are read upon the instantiation of the ECG class, though it is possible to re-read data if required for
some reason.

.. code-block:: python3

    >>> import signalanalysis.ecg
    >>> import signalanalysis.vcg
    >>> ecg_example = signalanalysis.ecg.Ecg("filename")
    >>> vcg_example = signalanalysis.vcg.Vcg(ecg_example)