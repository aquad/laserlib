.. _VeloRangeImage:
.. currentmodule:: LaserPy

Velodyne Range Image 
====================

The range image here isn't really an image. Due to the Velodyne's irregular
sampling in azimuth, converting a scan to an 'image' (a regular grid of range
values) will lose some points. Instead, this class stores multiple points in
each cell (actually, just the point id's, so all associated velodyne data
fields can be referred to). This class is used in efficient :ref:`region
selection <Selector>`.


.. autoclass:: VeloRangeImage
   :members:


Functions
---------

.. autofunction:: RangeImageMatch

.. autofunction:: NNImageQuery

