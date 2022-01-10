docutils Documentation
======================

This is the index of all modules and their associated methods and classes, documenting their use, parameters and
return values. See :doc:`usage` for a more step-by-step introduction to the intended use cases.

Broadly speaking, the modules are split thus:

* ``signalanalysis`` covers the analysis scripts for ECG/VCG/EGM analysis (e.g. calculating QRS duration)
* ``signalplot`` covers plotting methods (e.g. plotting the ECG leads on a single figure with annotation, plotting a
  3D plot of VCG (including animation!))
* ``tools`` covers more general use tools that are not limited to ECG/VCG analysis.

.. autosummary::
	:toctree: _autosummary
	:template: custom-module-template.rst
	:recursive:
	
	signalanalysis
	signalplot
	tools
