/*! PCAFrame_py.cpp
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
 * \date       25-01-2012
*/

#include <Python.h>
#define PY_ARRAY_UNIQUE_SYMBOL _LaserPy_ARRAY_API
#define NO_IMPORT_ARRAY
#include "LaserPy/numpy_to_eigen.h"
#include <boost/python.hpp>
#include "PCAFrame_py.h"



void export_PCAFrame()
{
    using namespace boost::python;
    //boost::shared_ptr<PCAFrames_py> //boost::noncopyable
    class_<PCAFrames_py, boost::shared_ptr<PCAFrames_py> >(
        "PCAFrames",
        "PCAFrames(n)\n\n"
        "Defines a set of frames (3D points with a 3D orientation) calculated from PCA keypoints",
        init<int>() )

        .add_property("P", &PCAFrames_py::get_P,
                      "(n,3) ndarray, float32.\n"
                      " 3D points")

        .add_property("R", &PCAFrames_py::get_R,
                      "(n,3,3) ndarray, float32.\n"
                      " Rotation matrices")

        .add_property("frameType", &PCAFrames_py::get_frameType,
                      "(n,) ndarray, uint8.\n"
                      " Type of region, 0 = surface, 1 = linear")

        .add_property("alignVect", &PCAFrames_py::get_alignVect,
                      "(n,3) ndarray, float32.\n"
                      " Alignment vector (surf norm for surface region, linear direction for linear region)")

        .add_property("pcaId", &PCAFrames_py::get_pcaId,
                      "(n,) ndarray, int32.\n"
                      " Associated PCA results for each frame can be found with this id")

        .enable_pickling()
        .def("__getstate__", &PCAFrames_py::__getstate__)
        .def("__setstate__", &PCAFrames_py::__setstate__)
        .def("__getinitargs__", &PCAFrames_py::__getinitargs__);


    // has default arguments!
    def("ComputePCAFrames", &ComputePCAFrames_py,
        ComputePCAFrames_overloads(
        "ComputePCAFrames(meanP, evals, evects, frames[, surfThresh=0.3, linThresh=0.5, ssRad=0.1])\n\n"
        "After computing PCA on regions in a point cloud, the eigenvalues/vectors"
        "can be used to compute a frame of orientation to compute features at. "
        "If a region forms a good plane, x,y is parallel (y faces "
        "downwards/towards origin), z faces away from origin. "
        "If it's a good line, x is parallel, z aligns to origin (facing outwards).\n\n"
        "Parameters\n"
        "----------\n"
        "meanP : (n,3) ndarray, float64\n"
        "   Mean point of each region (computed from PCA)\n"
        "evals : (n,3) ndarray, float32\n"
        "   Eigenvalues\n"
        "evects : (n,3,3) ndarray, float32\n"
        "   Eigenvectors\n"
        "frames : :class:`PCAFrames`\n"
        "   (output) Resulting PCA frames\n"
        "surfThresh : float (optional)\n"
        "   If the surface-ness of a region is above this, it is considered a "
        "'surface' keypoint. Default = 0.3.\n"
        "linThresh : float (optional)\n"
        "   If the linear-ness of a region is above this, it is considered a "
        "'linear' keypoint. Default = 0.5.\n"
        "ssRad : float (optional)\n"
        "   Subsampling radius (m). All points are at least this distance from "
        "each other. Default = 0.01.\n"
        ));
}


using namespace Eigen;


void ComputePCAFrames_py(
        PyObject* meanP_py, PyObject* evals_py, PyObject* evects_py, PCAFrames_py& frames,
        float surfThresh, float linThresh, float ssRad)
{
    Mat3<double>::type meanP = numpy_to_eigen<double, Dynamic, 3>( meanP_py, "meanP", NPY_DOUBLE );
    int n = meanP.rows();
    Mat3<float>::type evals = numpy_to_eigen<float, Dynamic, 3>( evals_py, "evals", NPY_FLOAT, n );
    MapMat33Xf evects = numpy_to_eigen<float, Dynamic, 9>( evects_py, "evects", NPY_FLOAT, n );

    Py_BEGIN_ALLOW_THREADS;
    ComputePCAFrames( meanP, evals, evects, frames, surfThresh, linThresh, ssRad );
    Py_END_ALLOW_THREADS;
}





PyObject* PCAFrames_py::__getstate__()
{
    PyObject* tup = PyTuple_Pack(5, get_P(), get_R(), get_frameType(), get_alignVect(), get_pcaId());
    return tup;
}


PyObject* PCAFrames_py::__getinitargs__()
{
    PyObject* tup = PyTuple_Pack(1, PyInt_FromLong(size));
    return tup;
}


void PCAFrames_py::__setstate__(PyObject* state)
{
    PyObject* P_py = PyTuple_GetItem(state,0);
    PyObject* R_py = PyTuple_GetItem(state,1);
    PyObject* frameType_py = PyTuple_GetItem(state,2);
    PyObject* alignVect_py = PyTuple_GetItem(state,3);
    PyObject* pcaId_py = PyTuple_GetItem(state,4);

    //check sizes etc...
    int n = PyArray_DIM( (PyArrayObject*)P_py, 0 );
    float* P_p = numpy_to_ptr<float>( P_py, "P", NPY_FLOAT, n, 3 );
    float* R_p = numpy_to_ptr<float>( R_py, "R", NPY_FLOAT, n, 3, 3 );
    unsigned char* frameType_p = numpy_to_ptr<unsigned char>( frameType_py, "frameType", NPY_UBYTE, n );
    float* alignVect_p = numpy_to_ptr<float>( alignVect_py, "alignVect", NPY_FLOAT, n, 3 );
    int* pcaId_p = numpy_to_ptr<int>( pcaId_py, "pcaId", NPY_INT, n );

    //copy to eigen containers
    resize(n);
    memcpy( P.data(), P_p, size*3*sizeof(float) );
    memcpy( R.data(), R_p, size*9*sizeof(float) );
    memcpy( frameType.data(), frameType_p, size*sizeof(unsigned char) );
    memcpy( alignVect.data(), alignVect_p, size*3*sizeof(float) );
    memcpy( pcaId.data(), pcaId_p, size*sizeof(int) );
}


