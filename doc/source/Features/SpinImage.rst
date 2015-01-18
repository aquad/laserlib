.. _SpinImage:
.. currentmodule:: LaserPy

Spin Image
==========

Implemented from [johnson1997spin]_. Requires surface normals and 3D points.

.. [johnson1997spin] A.E. Johnson. Spin-images: a representation for 3-D
   surface matching. 1997.


Computation
-----------

.. autoclass:: SpinImage
   :members:


Comparison
----------

Two different similarity measures are available.
In the original paper, two line images are compared by computing the linear
correlation coefficient (and applying arctanh^2 for arcane statistical
reasons).  However, to deal with occlusion, only overlapping occupied bins are
used (if a bin is empty in one spin image, it and the bin from the other spin
image are not included in the calculation). The number of bins included is then
added as a separate term with a scaling factor 'lambda'. This removal of empty
bins appears to have a detrimental effect on the distinguishability of line
images, at least on sparse veloydne data. As such, two distance measures are
presented: the original 'similarity' measure removing empty bins etc, and the
initial linear correlation coefficient (with arctanh^2) involving all bins. 

.. autofunction:: MatchSpinSets
.. autofunction:: MatchSpinSetsCorr
.. autofunction:: SpinCorrelation
.. autofunction:: SpinSimilarity
.. autofunction:: match_spin_image_correlation_sets
.. autofunction:: match_spin_image_similarity_sets


Classification
--------------

KNN on objects
^^^^^^^^^^^^^^

These classes classify objects, each with their own set of spin images. An
object dataset is defined by a list of objects.  Each object is an ndarray of
images for :class:`SpinImageKnn`, or a class containing attributes as the spin
images and alignment information for :class:`SpinImageKnnAligned`.
This allows easy creation of testing and training sets by manipulating lists of
objects, rather than aggregating objects into a large contiguous feature array.

.. autoclass:: SpinImageKnn
   :members:

.. autoclass:: SpinImageKnnAligned
   :members:


KNN between 2 sets
^^^^^^^^^^^^^^^^^^

When there are just 2 sets of line images (testing & training), these functions
can be used.

.. autofunction:: SpinKnnCorrelation
.. autofunction:: SpinKnnSimilarity

