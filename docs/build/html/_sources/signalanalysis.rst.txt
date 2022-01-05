=======================
Overall signal analysis
=======================

Where possible, modules and functions are reused between different types of signal analysis. This happens most
frequently with ECG and VCG data, and infrequently with EGM data. Documentation is within the individual functions
and modules, with general use cases for each signal type described in the following:

.. toctree::
   :maxdepth: 2
   :caption: Contents:
   :glob:

   signalanalysis-*

..
  The following is the original list, but I realised that it may be better to put in a toctree instead
  - :doc:`ECG <signalanalysis-ecg>`
  - :doc:`VCG <signalanalysis-vcg>`
  - :doc:`EGM <signalanalysis-egm>`