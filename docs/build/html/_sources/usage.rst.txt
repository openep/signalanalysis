Usage
=====

.. _installation:

Installation & Getting Started
------------------------------

To use signalanalysis, it is highly recommended to install using ``pipenv``, the virtual environment, to ensure that dependencies are where possible maintained. Clone the repository, then install the requirements as follows:

.. code-block:: console
	
	user@home:~$ git clone git@github.com:philip-gemmell/signalanalysis.git
	user@home:~$ cd signalanalysis
	user@home:~/signalanalysis$ pipenv install

Once the repository is cloned, it is currently the case that all work must be done within the Python3 environment. However, it is recommended to use the virtual environment from pipenv rather than the system-wide Python3 (after entering a pipenv shell, it is quit using the ``exit`` command as shown)

.. code-block:: console

	user@home:~/signalanalysis$ pipenv shell
	(signalanalysis) user@home:~/signalanalysis$ python3
	>>> import signalanalysis
	>>> quit()
	(signalanalysis) user@home:~/signalanalysis$ exit
	user@home:~/signalanalysis$

The project is arranged into various subdivisions. The required analysis/plotting packages for the ECG/VCG are separated out, and require separate importing. See the next section, and individual help files for further details.

- signalanalysis

  - signalanalysis.general
  - signalanalysis.ecg
  - signalanalysis.vcg

- signalplot
- tools

.. _reading:

Reading ECG/VCG data
--------------------

Currently, only ECG reading is supported; VCG data is calculated from ECG data using the Kors method (see method). ECG files are read upon the instantiation of the ECG class, though it is possible to re-read data if required for some reason.

.. code-block:: python3

	>>> import signalanalysis.ecg
	>>> import signalanalysis.vcg
	>>> ecg_example = signalanalysis.ecg.Ecg("filename")
	>>> vcg_example = signalanalysis.vcg.Vcg(ecg_example)

.. _classplan:

Shifting to classes from methods
--------------------------------

Previously, all ECG/VCG data was extracted and stored in DataFrames, and most of the modules in this code currently support this format. However, it is planned to shift the main focus of the project to use classes, which allow encapsulation of linked data in one data structure. While the focus of this documentation will be future-facing, and look at using the classes, note that sometimes access to the raw, underlying DataFrames will still be required. To that end, the raw data can be accessed as the .data attribute:

.. code-block:: python3

	>>> ecg_class = signalanalysis.ecg.Ecg("filename") # Returns an Ecg class
	>>> vcg_class = signalanalysis.vcg.Vcg(ecg_class)  # Returns a Vcg class
	>>> ecg_data = ecg_class.data                      # Returns a Pandas DataFrame of the underlying data
	>>> vcg_data = vcg_class.data                      # Returns a Pandas DataFrame of the underlying data


