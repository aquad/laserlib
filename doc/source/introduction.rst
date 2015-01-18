
Introduction 
============

Current functions include computing graph and :doc:`image
<DataStore/VeloRangeImage>` structures for Velodyne laser data, :doc:`selecting
<DataStore/Selector>` regions of points, computing various :doc:`features
<Features>` on point clouds and classifying them. It is developed for close use
with python and numpy- all functions have a python wrapper.  Typically, python
functions get converted to c++ and end up here.

For examples of use, see the :ref:`pyception <pyception-index-label>`
repository. 


Prerequisites
-------------

LaserPy is a part of `LaserLib
<http://www-personal.acfr.usyd.edu.au/aqua1490/LaserLib>`_.  It should be
cross-platform, but has only been tested on Linux.  CMake options exist that
allow LaserLib to be built without many extra dependencies.

Required dependencies:

- `Eigen 3 <http://eigen.tuxfamily.org/>`_ matrix library
- `Boost <http://www.boost.org/>`_

Highly recommended dependencies:  

- `Python <http://www.python.org/>`_ (2.7 recommended)
- `Numpy <http://numpy.scipy.org/>`_ >= 1.6 (Ubuntu users: get it through `pip <http://en.wikipedia.org/wiki/Pip_(Python)>`_)

Extra dependencies:

- `Flann <http://people.cs.ubc.ca/~mariusm/index.php/FLANN/FLANN>`_ - Nearest
  neighbours, KDTree
- `PCL <http://pointclouds.org/>`_ Point Cloud Library

Usage examples and the 'front end' of typical use:

- `Perception Python
  <http://www-personal.acfr.usyd.edu.au/p.morton/pyception/>`_ aka pyception


Installation 
------------

From a Package
^^^^^^^^^^^^^^

Only Arch Linux is currently packaged, others should build from source.

Arch Linux
""""""""""

The repo is available to ACFR IP addresses only.
To get pacman to see this repo, add this to your ``/etc/pacman.conf``::

    [alas-repo]
    Server = http://www-personal.acfr.usyd.edu.au/a.quadros/repo 

The repo contains some other packages (snark etc) you might find useful.



From Source
^^^^^^^^^^^

LaserLib can be checked out from the svn repository::

    svn co svn+ssh://user@sequoia.acfr.usyd.edu.au/u1/svn/BAE/Perception/cpp/trunk/LaserLib2

It follows a CMake build system:

- Create a new directory for building.
- Run ccmake (or the cmake gui) in that
  directory and set the desired options:

  - USE_Flann: Not needed, but choose ON if you want the ability to select
    arbitrary regions of unstructured 3D points. A spin image routine uses this,
    but otherwise not needed.  
  - USE_OPENMP: ON recommended, for multi-cpu parallel for loops. OpenMP is a
    part of most compilers, there is no need to install anything extra for this.  
  - USE_PCL: Not needed unless you want PCL functions (see the PCL directory).  
  - USE_PYTHON: ON recommended, requires Python and Numpy (unless you only want
    to use the c++ libraries in your own c++ project).

An example (in linux)::

    cd ~/code/source
    svn co svn+ssh://user@sequoia.acfr.usyd.edu.au/u1/svn/BAE/Perception/cpp/trunk/LaserLib2
    mkdir ~/code/build/LaserLib2
    cd ~/code/build/LaserLib2
    ccmake ~/code/source/LaserLib2

Then enable/disable options, press c. If there is no 'Press [g]' option, press
c again or check if any directories couldn't be found. Press g::

    make -j4
    sudo make install



Common pitfalls 
^^^^^^^^^^^^^^^

- CMake complains about missing ``FindSomething.cmake`` files.  You probably ran
  cmake in the same directory as the checked out source, resulting in cmake
  deleting said files in the CMakeFiles directory. Delete and checkout the source
  again, and build in a separate directory.

- When importing LaserPy into python, it gives a linking error.  LaserLib has
  no applications, so it could build happily but be unable to find certain
  functions. A common problem is Flann, which may install it's library files in
  ``/usr/local/lib64`` (where noone can find it). To fix this, edit
  ``/etc/ld.so.conf`` and in a new line put::

    /usr/local/lib64

  Then run::
  
    sudo ldconfig

  Now applications can find Flann. Check if it worked by running::

    python -c "import LaserPy"


Library Overview 
----------------

- Common: Currently headers-only, small commonly used routines.

- :doc:`DataStore`: Anything to do with storing data for certain uses. Eg. storing point
  data in a KDTree or range image to facilitate region selection.

- :doc:`Features`: Simple features such as surface normals and PCA.  Detailed features:
  spin image, line image. Also has some classes for object classification.

- LaserPy: All python-wrapped functions are declared in ``LaserPy.cpp``. Has some
  python-only code, most python wrappers are in ``DataStore/Python`` etc.

- Misc: Anything else, currently has ICP and geometric consistency functions.

- PCL: Point Cloud Library wrappers. Currently includes NARF keypoints/features,
  raytracing routines.


Typical Data Containers 
-----------------------

Point cloud data can come with many extra fields, such as laser
elevation/azimuth, intensity, time etc. Rather than create clunky data
structures that encapsulate extra information that may or may not be needed,
all data fields are kept in separate arrays. Eigen matrix `maps
<http://eigen.tuxfamily.org/dox/QuickRefPage.html#QuickRef_Map>`_ are often used
to encapsulate data for easy access. 

A typical use example: an nx3 contiguous
array of 3D points are received from a sensor, along with an associated array
of azimuth angle. These two arrays are given to a function, which computes a
value for each point, returning a result array. 

LaserLib expects row-ordered 'c-style' contiguous arrays (eg, 3D point exists
contigously in memory, xyz,xyz,... where row i corresponds to point i). Numpy
uses row-ordered, but Eigen (and PCL) defaults to column-ordered arrays. Make
sure the line::

    #define EIGEN_DEFAULT_TO_ROW_MAJOR 1

exists before any eigen headers are included in a file. 
In general, be weary of Eigen types coming from outside (such as from
PCL functions).


Python Bindings 
---------------

Python bindings exist in separate directories, for example:
``DataStore/BearingGraph.h``, ``DataStore/Python/BearingGraph_py.h``.  The main
implementation is independent of python, with a separate derived class or
wrapping function in the Python folder. This file does some conversions and may
instead take in an array of inputs to be called in a for loop (so you don't
have to in python). All the python classes & functions are also defined in
``LaserPy.cpp`` (anything in here is visible from python).

When you call a function in python, boost automatically converts simple int,
double or string arguments, as well as any class defined in the
``BOOST_PYTHON_MODULE`` section (it even downcasts derived types). However, it
does not do numpy arrays. Ideally, one would write boost converters to
automatically convert numpy arrays to a c++ friendly interface (such as Eigen).
However, you get a lot of flexibiliy from using the Numpy C API directly,
without too much difficulty. As such, python wrappers take in numpy arrays as a
raw ``PyObject*``, and use the C API to get the pointer to the underlying array
data, the data type, size etc. A convenience function ``checkNumpyArray()``
comes in handy. While Boost can do a lot of crazy things, in general, it makes
sense to have an extra layer of c++ code to do some python-oriented things like
applying an operation elementwise on an array, whereas in pure c++ there may be
no benefit to which side of the interface the for loop goes.

For more information, see:

- `Python C API <http://docs.python.org/c-api/index.html>`_
- `Numpy C API <http://docs.scipy.org/doc/numpy/reference/c-api.array.html>`_
- `Boost python
  <http://www.boost.org/doc/libs/1_50_0/libs/python/doc/index.html>`_
- `Python wiki on boost <http://wiki.python.org/moin/boost.python>`_


Sensors
-------

Most functions were developed and tested on Velodyne data, and some may not
work on anything else. Many algorithms are based around the concept of a range
image, and should work on any range image data. However, 3D sensors can have
very different properties, requiring their own specialised functions. For
example, velodyne range image pixels are not 'square'. Nodding sick lasers,
structured light (eg kinect) sensors, 3D stereo vision sensors, all have
different properties. It is a future goal that 3D features be designed such
that they can be easily modified/subclassed to use with any sensor.

Kinect functions (maybe) coming soon!


Developer pitfalls 
------------------

- When importing LaserPy in python, it fails with a linking error. Check if
  the specified function has been renamed or the arguments changed in a
  declaration but not the definition. If you use Eigen types, check that ``#define
  EIGEN_DEFAULT_TO_ROW_MAJOR 1`` is before any Eigen includes, or PCL includes.
  Eigen defaults to column major arrays, but python and LaserLib uses row major.
  Your function declaration may have row major, but the definition has column
  major.

- Random segfault, debug ends with numpy_to_eigen or some numpy c-api function.
  Check you have these defined before any numpy includes::

    #define PY_ARRAY_UNIQUE_SYMBOL _LaserPy_ARRAY_API
    #define NO_IMPORT_ARRAY


Documentation 
-------------

`Doxygen <www.doxygen.org>`_ is used in the source code, but the documentation
you are reading is generated by `Sphinx <http://sphinx.pocoo.org>`_. Doxygen
will document the C++ side of things, so all pure C++ code should have
doxygen-style comments. Python interfaces should have separate documentation,
placed in the docstring of the corresponding ``boost::python::def`` or ``class_``
section, using rst format (see the export_* functions in python wrapper files
for examples). Sphinx will read the docstrings to document the Python
interface.

To build the Doxygen documentation, in the ``doc`` folder, run ``doxygen
Doxyfile``. To build the Sphinx documentation, first build/install LaserLib
with Python enabled in CMake. Then in the ``doc`` folder, run ``make html``.

