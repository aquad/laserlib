.. _LineImage:
.. currentmodule:: LaserPy

Line Image 
==========

The Line Image is designed to capture as much information about a region of
space, while being robust to occlusion and missing data in Velodyne scans.
See [quadros2012feature]_ for a full description. At present,
it is for Velodyne data only, and requires the full range image (ie. azimuth,
laser id, range, 3D points).

Pyception is required (some parameter classes are defined there).  See the
classes in :mod:`pyception <pyception.algorithms.line_image>` for more info,
some helpful wrapper classes and :mod:`visualisation
<pyception.algorithms.line_image_draw>` tools.

In order to compare regions, this feature has custom functions to compute
'shape distances' between regions.  A number of different measures of distance
treat empty and occluded space differently; more research may be done on
finding better ones.  The distance specified in the ICRA paper is metric 2, and
seems to work well.

.. [quadros2012feature] A. Quadros, J.P. Underwood, and B. Douillard. An
   occlusion-aware feature for range images. In Robotics and Automation, 2012.
   ICRA’12. IEEE International Conference on. IEEE, May 14-18 2012.


Computation
-----------

.. autoclass:: ComputeLineImage
   :members:


Comparison
----------

.. autofunction:: match_line_images
.. autofunction:: match_line_images_keys
.. autofunction:: match_line_image_sets
.. autofunction:: match_line_images_all


Classification
--------------

LineImageKnn
^^^^^^^^^^^^

.. autoclass:: LineImageKnn
    :members:

LineImageKnnAligned
^^^^^^^^^^^^^^^^^^^

.. autoclass:: LineImageKnnAligned
    :members:

ObjectMatchHistogram
^^^^^^^^^^^^^^^^^^^^

.. autoclass:: ObjectMatchHistogram
    :members:

