.. _Selector:
.. currentmodule:: LaserPy

Region Selection
================

LaserLib provides several ways to select regions of 3D points:

- KDTrees via Flann
- Using the range image
- Using the bearing graph

.. /comment inheritance-diagram:: Selector, ImageSphereSelector, FlannKDTree,
   PreFourNeighSelector, GraphSphereSelector

The :class:`Selector` is an abstract base class with just one method-
:meth:`~Selector.SelectRegion`, which selects a region about the specified
point id.  This allows the actual 3D points, plus any extra associated data,
and things like the radius/size of selection to be specified in the derived
class.

.. digraph:: selection

   "ImageSphereSelector" -> "Selector";
   "FlannKDTree" -> "Selector";
   "PreFourNeighSelector" -> "Selector";
   "GraphSphereSelector" -> "Selector";
   "ImagePlusSelector" -> "Selector";


.. class:: Selector

   .. method:: SelectRegion(centre)

      Select a region about a centre point.

      Parameters
      ----------
      centre : int
        Refers to centre point id.

      Returns
      -------
      ids : (k,) ndarray, int32
        Point ids within selection.


Range Image
-----------

The :class:`VeloRangeImage` class organises velodyne data in a 2D grid, which can
be used for efficient region selection. This is faster than building a kd-tree.
The :class:`ImageSphereSelector` uses the depth of the centre point and some
basic trig to define an angular region encompassing the desired 3D spherical
one. 3D point distances are then compared to select the points. This method was
found to be ~2x faster than Flann's kd-tree (although there are many
kd-tree parameters to try).

.. autoclass:: ImageSphereSelector
   :members:



KDTree
------

This is a simple wrapper of Flann's kd-tree to work as a :class:`Selector`.
Flann has it's own python bindings if you want more functionality.

This class can also select points about arbitrary (ie. non-sensed) 3D points 
with :meth:`FlannKDTree.Select3D`.

.. autoclass:: FlannKDTree
   :members:



Bearing Graph
-------------

See [moosmann2009segmentation]_ for the concept of the bearing graph.
The :class:`BearingGraphBuilder` produces this graph, a (n,4) int32
ndarray specifying point connectivity (like a mesh).
This is used to select regions of points with a breadth-first-search.
You can incorporate segmentation by disconnecting links between different
segments, allowing you to select points only on a given segment.
This method is actually quite slow, and probably scales badly. The c++ could
use a revamp, or some boost::graph love.

A faster, sparser region selection involves just finding neighbours
left/right/up/down along the graph. These four neighbours are sufficient
for calculating surface normals and curvature. See :func:`Get4Neighs` and
:class:`PreFourNeighSelector`.

GraphSphereSelector
^^^^^^^^^^^^^^^^^^^

.. autoclass:: GraphSphereSelector
   :members:

Get4Neighs
^^^^^^^^^^

.. autofunction:: Get4Neighs

.. autofunction:: Get4NeighsValid

.. autoclass:: PreFourNeighSelector
   :members:


