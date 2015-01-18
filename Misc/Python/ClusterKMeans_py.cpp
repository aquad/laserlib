/*! ClusterKMeans_py.cpp
 *
 * Copyright (C) 2012 Alastair Quadros.
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
 * \date       18-01-2012
*/

#include <Python.h>
#define PY_ARRAY_UNIQUE_SYMBOL _LaserPy_ARRAY_API
#define NO_IMPORT_ARRAY
#include "LaserPy/numpy_to_eigen.h"
#include <boost/python.hpp>
#include "Misc/ClusterKMeans.h"


PyObject* ClusterKMeans_py( PyObject* data_py, int nClusters, int nIters );


void export_ClusterKMeans()
{
    using namespace boost::python;
    def("ClusterKMeans", &ClusterKMeans_py);
}

using namespace Eigen;


PyObject* ClusterKMeans_py(PyObject* data_py, int nClusters, int nIters )
{
    MapMatXf data = numpy_to_eigen<float, Dynamic, Dynamic>( data_py, "data", NPY_FLOAT );

    //make output array (of ids)
    npy_intp dims[2];
    dims[0] = PyArray_DIM( (PyArrayObject*)data_py, 0 );
    PyArrayObject *ids_py = (PyArrayObject*)PyArray_SimpleNew(1, dims, NPY_INT);
    Vect<int>::type ids( (int*)PyArray_DATA(ids_py), PyArray_DIM(ids_py,0) );

    ClusterKMeans( data, nClusters, ids, nIters );
    return PyArray_Return(ids_py);
}

