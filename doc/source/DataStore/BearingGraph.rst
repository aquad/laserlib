.. _BearingGraph:
.. currentmodule:: LaserPy

Bearing Graph
=============

See [moosmann2009segmentation]_ for the concept of the bearing graph.
The :class:`BearingGraphBuilder` produces this graph, a (n,4) int32
ndarray specifying point connectivity (like a mesh). So for example, the
connectivity of point 6 is shown on row 6 of the graph, with the left, right,
up, down neighbours indicated. -1 indicates no neighbour. This array is 
owned by :class:`BearingGraphBuilder`, and is overwritten on subsequent method
calls, so copy it out if you want to keep it.

The bearing graph can be used to compute surface normals, :doc:`select regions
of points <Selector>`, or determine the :func:`sampling density
<minRadiusSelection>` at a given point in the scan.

The :class:`BearingGraphBuilder` requires certain velodyne data fields.  These
are provided by :class:`~pyception.sensors.velodyne.VeloReader` by adding
'id', 'w' and 'D' to the 'fields' list on construction::

    r = VeloReader( veloFolder, dbFile, fields=['id','D','w'], indexFile=indexFile )

.. [moosmann2009segmentation]  F. Moosmann, O. Pink, and C. Stiller.
  Segmentation of 3D lidar data in non-flat urban environments using a local
  convexity criterion. In Intelligent Vehicles Symposium, 2009 IEEE, pages
  215–220. IEEE, 2009.


.. autoclass:: BearingGraphBuilder
   :members:


Surface Normals
---------------

Surface normals can be calculated using the average cross product of
neighbouring links in the graph. This is very fast but noisy, as opposed to
:ref:`PCA`, which is slow but cleaner. Blurring helps.

.. autofunction:: CalcSurfNorms

.. autofunction:: BlurSurfNorms

