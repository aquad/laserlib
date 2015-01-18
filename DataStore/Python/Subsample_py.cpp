/*! Subsample_py.cpp
 *
 * Copyright (C) 2011 Alastair Quadros.
 *
 * This file is part of LaserLib.
 *
 * LaserLib is free software: you can redistribute it and/or modify it under
 * the terms of the GNU Lesser General Public License as published by the Free
 * Software Foundation, either version 3 of the License, or (at your option)
 * any later version.
 *
 * LaserLib is distributed in the hope that it will be useful, but WITHOUT ANY
 * WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
 * FOR A PARTICULAR PURPOSE.  See the GNU Lesser General Public License for
 * more details.
 *
 * You should have received a copy of the GNU Lesser General Public License
 * along with LaserLib.  If not, see <http://www.gnu.org/licenses/>.
 *
 * \author     Alastair Quadros
 * \date       31-05-2011
*/

#include <Python.h>
#define PY_ARRAY_UNIQUE_SYMBOL _LaserPy_ARRAY_API
#define NO_IMPORT_ARRAY
#include "LaserPy/numpy_to_eigen.h"

#include <boost/python.hpp>
#include "Subsample_py.h"


void export_Subsample()
{
    using namespace boost::python;

    def("SubSampleEvenly", &SubSampleEvenly_py,
        "SubSampleEvenly(sel, nTotal)\n\n"
        "Evenly subsample by region selection.\n\n"
        "The selector is used to select points about a centre point. These "
        "neighbours are removed, the centre point added to the output sample, "
        "and the next non-removed centre point is selected.\n\n"
        "Parameters\n"
        "----------\n"
        "sel : :class:`Selector`\n"
        "   Defines the points, radius/method of selecting the region\n"
        "nTotal : int\n"
        "   number of points given to :obj:`sel`\n\n"
        "Returns\n"
        "-------\n"
        "subsample : (k,) ndarray, int32\n"
        "   Point ids (refers to points provided to :obj:`sel`)");

    def("SubSampleKeysEvenly", &SubSampleKeysEvenly_py,
        "SubSampleKeysEvenly(sel, nTotal, keys)\n\n"
        "Evenly subsample keypoints by region selection.\n\n"
        "As in :func:`SubSampleEvenly`, but :obj:`keys` are iterated over as centre "
        "points. This is useful if you have special points (eg. points with "
        "valid surface normals) that you wish to subsample, but the region "
        "selection method (kdtree/range image etc) is defined on all the "
        "points\n\n"
        "Parameters\n"
        "----------\n"
        "sel : :class:`Selector`\n"
        "   Defines the points, radius/method of selecting the region\n"
        "nTotal : int\n"
        "   number of points given to :obj:`sel`\n"
        "keys : (k,) ndarray, int32\n"
        "   Keypoints to be subsampled\n\n"
        "Returns\n"
        "-------\n"
        "subsample : (m,) ndarray, int32\n"
        "   Point ids (refers to points provided to :obj:`sel`)");

    def("SubsampleBySurfNorm", &SubsampleBySurfNorm_py,
        "SubsampleBySurfNorm(sel, nTotal, keys, sn, thresh)\n\n"
        "Remove points near each other with similar surface normals.\n\n"
        "Parameters\n"
        "----------\n"
        "sel : :class:`Selector`\n"
        "   Defines the points, radius/method of selecting the region\n"
        "nTotal : int\n"
        "   number of points given to :obj:`sel`\n\n"
        "keys : (k,) ndarray, int32\n"
        "   Keypoints to be subsampled\n"
        "sn : (n,3) ndarray, float64\n"
        "   surface normals, aligned to the points given to :obj:`sel` (not :obj:`keys`)\n"
        "thresh : double\n"
        "   If the dot product of two surface normals is above this, a point is removed "
        "(cos(thresh) is min angle difference).\n\n"
        "Returns\n"
        "-------\n"
        "subsample : (m,) ndarray, int32\n"
        "   Point ids (refers to points provided to :obj:`sel`)");

    def("LocalMax", &LocalMax_py,
        "LocalMax(sel, nTotal, keys, val, maxBy)\n\n"
        "Find local maxima\n\n"
        "Given a set of scalar values at each point, find points that have a "
        "maximum value within a certain radius, giving how much they are "
        "greater on average.\n\n"
        "Parameters\n"
        "----------\n"
        "sel : :class:`Selector`\n"
        "   Defines the points, radius/method of selecting the region\n"
        "nTotal : int\n"
        "   number of points given to :obj:`sel`\n\n"
        "keys : (k,) ndarray, int32\n"
        "   Keypoints to be subsampled\n"
        "val : (n,) ndarray, float64\n"
        "   The value to maximise\n"
        "maxBy : (n,) ndarray, float64\n"
        "   (output) how much the resulting maximum was larger than it's "
        "surrounds. It must be a full sized array (ie, the number of points "
        "in the selector = nTotal = val.shape[0]). It will be 0 where there is "
        "no max (so the actual subsampling is given by non-zero values)");
}

using namespace Eigen;

PyObject* SubSampleEvenly_py( Selector& sel, int nTotal )
{
    std::vector<int> sample;
    SubSampleEven( sel, nTotal, sample );

    npy_intp dims[1] = {static_cast<npy_intp>(sample.size())};
    PyArrayObject* sample_py = (PyArrayObject*) PyArray_SimpleNew(1, dims, NPY_INT );
    memcpy( PyArray_DATA(sample_py), &(sample[0]), sizeof(int)*sample.size() );
    return PyArray_Return(sample_py);
}



PyObject* SubSampleKeysEvenly_py( Selector& sel, int nTotal, PyObject* keys_py )
{
    Vect<int>::type keys = numpy_to_eigen<int, Dynamic, 1>( keys_py, "keys", NPY_INT );
    std::vector<int> sample;
    SubSampleKeysEvenly( sel, nTotal, keys, sample );

    npy_intp dims[1] = {static_cast<npy_intp>(sample.size())};
    PyArrayObject* sample_py = (PyArrayObject*) PyArray_SimpleNew(1, dims, NPY_INT );
    memcpy( PyArray_DATA(sample_py), &(sample[0]), sizeof(int)*sample.size() );
    return PyArray_Return(sample_py);
}



void LocalMax_py( Selector& sel, int nTotal, PyObject* items_py, PyObject* val_py, PyObject* maxBy_py )
{
    Vect<int>::type items = numpy_to_eigen<int, Dynamic, 1>( items_py, "items", NPY_INT );
    Vect<double>::type val = numpy_to_eigen<double, Dynamic, 1>( val_py, "val", NPY_DOUBLE, nTotal );
    Vect<double>::type maxBy = numpy_to_eigen<double, Dynamic, 1>( maxBy_py, "maxBy", NPY_DOUBLE, nTotal );

    LocalMax( sel, nTotal, items, val, maxBy );
}


PyObject* SubsampleBySurfNorm_py( Selector& sel, int nTotal, PyObject* keys_py, PyObject* sn_py, double thresh)
{
    Vect<int>::type keys = numpy_to_eigen<int, Dynamic, 1>( keys_py, "keys", NPY_INT );
    Mat3<double>::type sn = numpy_to_eigen<double, Dynamic, 3>( sn_py, "sn", NPY_DOUBLE, nTotal );
    std::vector<int> sample;
    SubsampleBySurfNorm( sel, nTotal, keys, sn, thresh, sample );

    npy_intp dims[1] = {static_cast<npy_intp>(sample.size())};
    PyArrayObject* sample_py = (PyArrayObject*) PyArray_SimpleNew(1, dims, NPY_INT );
    memcpy( PyArray_DATA(sample_py), &(sample[0]), sizeof(int)*sample.size() );
    Py_INCREF(sample_py);
    return PyArray_Return( sample_py );
}

