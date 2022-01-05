==========================================
Welcome to signalanalysis's documentation!
==========================================

**signalanalysis** is a library including various tools for the reading, analysis and plotting of ECG and VCG data.
It is designed to be as agnostic as possible for the types of data that it can read. Currently, it can read ECG data
from:

#. CARP simulations of whole torso activity, using existing projects from
   `CARPutil <https://git.opencarp.org/openCARP/carputils>`_
#. .csv and .dat records
#. wfdb file formats, using `wfdb-python <https://github.com/MIT-LCP/wfdb-python>`_

Futher details of how to use these functions are in :doc:`usage`, with the finer points for each individual
function within the files themselves.

.. toctree::
   :maxdepth: 2
   :caption: Contents:
   :glob:
   
   usage
   Signal Analysis <signalanalysis>
   Signal Plot <signalplot>
   docutils
