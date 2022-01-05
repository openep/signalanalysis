============
VCG Analysis
============

.. _reading:

Reading VCG data
----------------

Currently, only ECG reading is supported; VCG data is calculated from ECG data using the Kors method (see method),
and thus VCG data can only (at this stage) be derived from ECG data.

.. code-block:: python3

    >>> import signalanalysis.ecg
    >>> import signalanalysis.vcg
    >>> ecg_example = signalanalysis.ecg.Ecg("filename")
    >>> vcg_example = signalanalysis.vcg.Vcg(ecg_example)