.. _pca-keypoints:
.. currentmodule:: LaserPy

Keypoints
---------

PCA can be used to generate keypoints at highly flat or linear regions, as in
[quadros2012feature]_. First, compute :ref:`PCA` on your data, then call
:func:`ComputePCAFrames`. The resulting :class:`PCAFrames` class contains all
the keypoint information.

For a class that does all of this for you, see
:class:`~pyception.algorithms.keypoints.PCAFrameFromScan` in
Pyception.

ComputePCAFrames
^^^^^^^^^^^^^^^^

.. autofunction:: ComputePCAFrames

PCAFrames
^^^^^^^^^

.. autoclass:: PCAFrames
   :members:

