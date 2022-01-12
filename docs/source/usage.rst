Usage
=====

.. _installation:

Installation & Getting Started
------------------------------

To use signalanalysis, it is highly recommended to install using ``pipenv``, the virtual environment, to ensure that
dependencies are where possible maintained. Clone the repository, then install the requirements as follows:

.. code-block:: console

    user@home:~$ git clone git@github.com:philip-gemmell/signalanalysis.git
    user@home:~$ cd signalanalysis
    user@home:~/signalanalysis$ pipenv install -e .

Once the repository is cloned, it is currently the case that all work must be done within the Python3 environment.
However, it is recommended to use the virtual environment from pipenv rather than the system-wide Python3 (after
entering a pipenv shell, it is quit using the ``exit`` command as shown)

.. code-block:: console

    user@home:~/signalanalysis$ pipenv shell
    (signalanalysis) user@home:~/signalanalysis$ python3
    >>> import signalanalysis
    >>> quit()
    (signalanalysis) user@home:~/signalanalysis$ exit
    user@home:~/signalanalysis$

The project is arranged into various subdivisions. The required analysis/plotting packages for the ECG/VCG are
separated out, and require separate importing. The individual functions must be imported and used
separately---further details for each function can be found within each files documentation.

- :doc:`signalanalysis`

  - signalanalysis.signalanalysis.general
  - :doc:`signalanalysis.signalanalysis.ecg <signalanalysis-ecg>`
  - :doc:`signalanalysis.signalanalysis.vcg <signalanalysis-vcg>`
  - :doc:`signalanalysis.signalanalysis.egm <signalanalysis-egm>`

- signalplot

  - signalanalysis.signalplot.general
  - signalanalysis.signalplot.ecg
  - signalanalysis.signalplot.vcg
  - signalanalysis.signalplot.egm

- tools

  - signalanalysis.tools.maths
  - signalanalysis.tools.plotting
  - signalanalysis.tools.python

.. _classplan:

Shifting to classes from methods
--------------------------------

Previously, all ECG/VCG data was extracted and stored in DataFrames, and most of the modules in this code currently
support this format. However, it is planned to shift the main focus of the project to use classes, which allow
encapsulation of linked data in one data structure. While the focus of this documentation will be future-facing, and
look at using the classes, note that sometimes access to the raw, underlying DataFrames will still be required. To
that end, the raw data can be accessed as the ``.data`` attribute:

.. code-block:: python3

    >>> import signalanalysis as sa
    >>> ecg_class = sa.signalanalysis.ecg.Ecg("filename") # Returns an Ecg class
    >>> vcg_class = sa.signalanalysis.vcg.Vcg(ecg_class)  # Returns a Vcg class
    >>> ecg_data = ecg_class.data                         # Returns a Pandas DataFrame of the underlying data
    >>> vcg_data = vcg_class.data                         # Returns a Pandas DataFrame of the underlying data


