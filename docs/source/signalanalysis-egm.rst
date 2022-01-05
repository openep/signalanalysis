============
EGM Analysis
============

.. _reading:

Reading EGM data
----------------

Currently, this software is only able to read EGM data saved in a .csv format. In this format, each new line
represents a fresh timepoint, and each comma-separated value on each line represents a different recording, i.e. a
file containing 1000 lines of individual values will be for one EGM trace of 1000 timepoints.

Currently, the unipolar data is **required**, whereas the bipolar data is optional. It is also required to pass the
frequency of the data when reading the data.

.. code-block:: python3

    >>> import signalanalysis.egm
    >>> example_egm = signalanalysis.egm.Egm('unipolar.csv', 'bipolar.csv', frequency=2034.5)

Other stuff...

