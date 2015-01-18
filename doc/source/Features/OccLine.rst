.. _OccLine:
.. currentmodule:: LaserPy

Occupancy Line
==============

This is a component of the :ref:`LineImage` feature for velodyne scans. Given a
3D line segment, it traverses along it, detecting if the line intercepts a
surface, or goes into unknown space (eg. an occlusion), or if it's completely
in empty space.

Pyception is required (the parameter class is defined there).  See 
:mod:`pyception <pyception.algorithms.line_image>` for more info.

.. autoclass:: OccLine
   :members:

