============
EGM Analysis
============

.. _egm-example:

Example data
------------

Sample EGM data are stored in ``signalanalysis/data/egm_unipolar.csv`` and ``signalanalysis/data/egm_bipolar.csv``.

.. _egm-reading:

Reading EGM data
----------------

Currently, this software is only able to read EGM data saved in a .csv format. In this format, each new line
represents a fresh timepoint, and each comma-separated value on each line represents a different recording, i.e. a
file containing 1000 lines of individual values will be for one EGM trace of 1000 timepoints.

Currently, the unipolar data is **required**, whereas the bipolar data is optional. It is also required to pass the
frequency of the data when reading the data.

.. code-block:: python3

    >>> import signalanalysis as sa
    >>> example_egm = sa.signalanalysis.egm.Egm('unipolar.csv', 'bipolar.csv', frequency=2034.5)
    >>> example_egm_nobipolar = sa.signalanalysis.egm.Egm('unipolar.csv', frequency=2034.5)

From this point on, the code should automatically adapt what methods it uses - no change in calls are required in
subsequent calculations depending on whether bipolar data are present.

.. _egm-activation:

Getting activation time
-----------------------

:py:meth:`signalanalysis.signalanalysis.egm.Egm.get_at()`

Activation time is calculated based on peaks in the EGM data. Where bipolar data are available, the peaks in the
squared bipolar signal are found (with minimum separation between peaks, and the threshold that must be passed before
it counts as a peak, being able to be set manually if desired). These peaks provide a 'search window', within which
the maximum downstroke of the unipolar signal is found---this is defined as the activation time for that particular
beat. The width of the search window, centred on the bipolar peak, can also be manually specified.

If bipolar data are not available, the squared unipolar signal is used for peak detection instead. However, this will
be 'thrown off' by  far-field pacing artefacts. To correct for this, the search window for the AT is not centred on
the detected unipolar peak, but is instead delayed to a point afterwards so that the peak is not included in the
search window (though this can be manually over-ridden).

The activation time for a given set of EGM data is calculated using the ``.get_at`` method. In its most basic call
this can be as simple as ``example_egm.get_at()``, with options available to plot example traces as required. These
are documented more thoroughly in the ``signalplot.egm.plot_signal`` method.

Once activation time is calculated, a dataframe of the ATs is stored in ``example_egm.at``. It should be noted that,
as a dataframe, it will be a `m` x `n` matrix, where `m` is the number of EGM signals available, and `n` is the
maximum number of ATs detected. For those signals where fewer than `n` ATs are found, ``NaN`` will be recorded to
fill in the blanks.

.. _egm-repolarisation:

Getting repolarisation time
---------------------------

:py:meth:`signalanalysis.signalanalysis.egm.Egm.get_rt()`

The repolarisation time is calculated based on the activation times - if AT has not been calculated prior to
calculating RT, it is automatically included in the process.

The cycle length of the signal is calculated based on the AT, and from that, a search window is estimated within
which the T-wave is searched for. Initially, this search looks for the T-wave peak by searching for the maximum value
of the EGM within the window. If the maximum occurs at the beginning or end of the window, it is assumed that the
maximum actually represents the tail end of the QRS complex, or the depolarisation associated with the following
beat, and the search window is shortened to avoid these artefacts. If the search window is narrowed continuously as
no peak is found, it is assumed that the T-wave is negative: the search window is set back to original size, and the
minimum value of the EGM is used as the negative peak.

Once the search window is determined, the signal in the window is search progressively, with the maximum gradient of
the unipolar signal being searched for. This is not a universal search, but a progressive one - once the peak has
been passed, as determined by the unipolar gradient moving to negative values, the search is stopped.

Repolarisation time is calculated as ``example_egm.get_rt()``, with the result stored in ``example_egm.rt``.

.. _egm-dvdt:

Getting (dV/dt)max
------------------

:py:meth:`signalanalysis.signalanalysis.egm.Egm.get_at()`

The value of (dV/dt) at the point of AT is automatically stored in the ``example_egm.dvdt`` attribute when AT is being
calculated.

.. _egm-qrsd:

Getting QRS duration
--------------------

:py:meth:`signalanalysis.signalanalysis.egm.Egm.get_qrsd()`

QRS duration can only be calculated when bipolar data are given. A search window based on the AT is isolated, and the
squared bipolar signal is calculated within that window. The time at which this signal exceeds a threshold value
(itself based on the maximum value within the search window) is used as the start of the QRS complex, and the last
point within the window at which the threshold is exceeded is the end of the QRS.

It is calculated using ``example_egm.get_qrsd()``, with the data stored as ``example_egm.qrs_duration``; QRS start
and end points are recorded in ``example_egm.qrs_start`` and ``example_egm.qrs_end``, respectively.