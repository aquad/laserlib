.. _StreamScanBuffer:
.. currentmodule:: LaserPy


StreamScanBuffer
================

Buffer a stream of binary data (eg. from velodyne-to-csv).

Comma/Snark/Ark utilise streaming applications, allowing velodyne
data to be segmented, tracked etc. In order to insert a custom
python application in such a pipeline, python must constantly read
from stdin. However, if processing is too slow, stale data is
received. 

The :class:`StreamScanBuffer` class constantly reads in
data in another thread, allowing a python application to retrieve
the latest scan at it's leisure. It expects binary-csv data, with
an int field which counts the scan number. The data is stored into
a structured numpy array, so any data format can be specified.

.. autoclass:: StreamScanBuffer
   :members:


