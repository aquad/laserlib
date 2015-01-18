/*! ICP_py.cpp
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
 * \date       11-05-2011
*/

#include <Python.h>
#define PY_ARRAY_UNIQUE_SYMBOL _LaserPy_ARRAY_API
#define NO_IMPORT_ARRAY
#include "LaserPy/numpy_to_eigen.h"
#include <boost/python.hpp>

#include "ICP_py.h"


void export_ICP()
{
    using namespace boost::python;
    def("ICP_PointPlane_2D", &ICP_PointPlane_2D_py,
        "ICP_PointPlane_2D(targetP, n, templateP)\n\n"
        "Point to plane ICP, yaw is not optimized (the template moves in 2D).\n\n"
        "Parameters\n"
        "----------\n"
        "targetP : ndarray (n,3) float64\n"
        "   target points (they stay still)\n"
        "n : ndarray (n,3) float64\n"
        "   surface normals of target points\n"
        "templateP : ndarray (n,3) float64\n"
        "   template points, these are moved to align with the target points.\n\n"
        "Returns\n"
        "-------\n"
        "R : ndarray (4,4) float64\n"
        "   rotation matrix\n"
        "errSqr : float\n"
        "   final ICP error\n"
        "isSingular : bool\n"
        "   If true, the rotation matrix was not found\n");

    def("ICP_PointPlane_3D", &ICP_PointPlane_3D_py,
        "ICP_PointPlane_3D(targetP, n, templateP)\n\n"
        "Parameters\n"
        "----------\n"
        "targetP : ndarray (n,3) float64\n"
        "   target points (they stay still)\n"
        "n : ndarray (n,3) float64\n"
        "   surface normals of target points\n"
        "templateP : ndarray (n,3) float64\n"
        "   template points, these are moved to align with the target points.\n\n"
        "Returns\n"
        "-------\n"
        "R : ndarray (4,4) float64\n"
        "   rotation matrix\n"
        "errSqr : float\n"
        "   final ICP error\n"
        "isSingular : bool\n"
        "   If true, the rotation matrix was not found\n");
}


using namespace Eigen;


PyObject* ICP_PointPlane_2D_py( PyObject* targetP_py, PyObject* n_py, PyObject* templateP_py )
{
    Mat3<double>::type targetP = numpy_to_eigen<double, Dynamic, 3>( targetP_py, "targetP", NPY_DOUBLE );
    Mat3<double>::type n = numpy_to_eigen<double, Dynamic, 3>( n_py, "n", NPY_DOUBLE, targetP.rows() );
    Mat3<double>::type templateP = numpy_to_eigen<double, Dynamic, 3>(
                templateP_py, "templateP", NPY_DOUBLE, targetP.rows() );

    TransformMatrix R = Eigen::Matrix4d::Identity();
    double errSqr = 0.0;
    bool isSingular = true;
    ICP_PointPlane_2D( targetP, n, templateP, R, errSqr, isSingular );

    //make output array
    npy_intp dims[2] = {4,4};
    PyArrayObject* R_py = (PyArrayObject*) PyArray_SimpleNew(2, dims, NPY_DOUBLE);
    memcpy( (double*)PyArray_DATA(R_py), R.data(), 16*sizeof(double) );

    PyObject* tup = PyTuple_Pack(3, R_py, PyFloat_FromDouble(errSqr), PyBool_FromLong(isSingular));
    Py_DECREF(R_py);
    return tup;
}


PyObject* ICP_PointPlane_3D_py( PyObject* targetP_py, PyObject* n_py, PyObject* templateP_py )
{
    Mat3<double>::type targetP = numpy_to_eigen<double, Dynamic, 3>( targetP_py, "targetP", NPY_DOUBLE );
    Mat3<double>::type n = numpy_to_eigen<double, Dynamic, 3>( n_py, "n", NPY_DOUBLE, targetP.rows() );
    Mat3<double>::type templateP = numpy_to_eigen<double, Dynamic, 3>(
                templateP_py, "templateP", NPY_DOUBLE, targetP.rows() );

    TransformMatrix R = Eigen::Matrix4d::Identity();
    double errSqr = 0.0;
    bool isSingular = true;
    ICP_PointPlane_3D( targetP, n, templateP, R, errSqr, isSingular );

    //make output array
    npy_intp dims[2] = {4,4};
    PyArrayObject* R_py = (PyArrayObject*) PyArray_SimpleNew(2, dims, NPY_DOUBLE);
    memcpy( (double*)PyArray_DATA(R_py), R.data(), 16*sizeof(double) );

    PyObject* tup = PyTuple_Pack(3, R_py, PyFloat_FromDouble(errSqr), PyBool_FromLong(isSingular));
    Py_DECREF(R_py);
    return tup;
}
