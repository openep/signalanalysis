=======================
Overall signal plotting
=======================

Currently, the plotting functions are the ones least likely to have migrated to the new class based system. This
means that, under almost all circumstances, the ``.data`` attribute of the class will need to be passed to the
plotting function rather than the class itself. This should be clear from the type-hinting for the function: where
``class.data`` is required, the type-hint will be for ``pd.DataFrame``.