/*! LineImage_py.cpp
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
 * \date       08-03-2011
*/

#include <Python.h>
#define PY_ARRAY_UNIQUE_SYMBOL _LaserPy_ARRAY_API
#define NO_IMPORT_ARRAY
#include "LaserPy/numpy_to_eigen.h"
#include "LaserPy/numpy_to_std.h"
#include "LaserPy/numpy_helpers.h"
#include <boost/python.hpp>
#include <omp.h>

#include "LineImage_py.h"
#include "DataStore/Python/VelodyneDb_py.h"
#include "Common/ProgressIndicator.h"
#include "LaserLibConfig.h"

using namespace Eigen;
using namespace boost::python;


void export_LineImage()
{
    class_<ComputeLineImage_py, boost::shared_ptr<ComputeLineImage_py> >("ComputeLineImage",
                                                                         "ComputeLineImage(params, P, id, D, w, pcaResults, db, image[, nThreads=1])\n\n",
                                                                         no_init)

            .def( "__init__", boost::python::make_constructor( &ComputeLineImage_py_constructor,
                                                               default_call_policies(),
                                                               (arg("params"), arg("P"), arg("id"), arg("D"), arg("w"),
                                                                arg("pcaResults"), arg("db"), arg("image"), arg("nThreads")=1) ) )


            .def("setObjectMask", &ComputeLineImage_py::setObjectMask_py,
                 "setObjectMask(mask)\n\n")

            .def("compute", &ComputeLineImage_py::compute_py,
                 "compute(R, P, values, status)\n\n");
}




ComputeLineImage_py::ComputeLineImage_py( LineImageParams& params, Mat3<double>::type& P, Vect<unsigned char>::type& id,
                                         Vect<double>::type& D, Vect<double>::type& w,
                                         PCAResults& pcaResults, VelodyneDb& db, VeloRangeImage& image, int nThreads )
    :   ComputeLineImage( params, P, id, D, w, pcaResults, db, image ),
        nThreads_(nThreads)
{}



boost::shared_ptr<ComputeLineImage_py> ComputeLineImage_py_constructor(
        PyObject* params_py, PyObject* P_py, PyObject* id_py, PyObject* D_py, PyObject* w_py,
        PyObject* pcaResults_py, PyObject* db_pyobj, VeloRangeImage& image, int nThreads )
{
    //these are temporary, but they get copied in the ComputeLineImage constructor.
    VelodyneDb_py db_py(db_pyobj);
    VelodyneDb& db = *dynamic_cast<VelodyneDb*>(&db_py);
    LineImageParams_py params_py_derived(params_py);
    LineImageParams& params = *dynamic_cast<LineImageParams*>(&params_py_derived);

    Mat3<float>::type evals = get_attribute_numpy_array<float, Dynamic, 3>( pcaResults_py, "evals", NPY_FLOAT );
    MapMat33Xf evects = get_attribute_numpy_array<float, Dynamic, 9>( pcaResults_py, "evects", NPY_FLOAT );
    Mat3<double>::type meanP = get_attribute_numpy_array<double, Dynamic, 3>( pcaResults_py, "meanP", NPY_DOUBLE );
    Vect<int>::type pidToResult = get_attribute_numpy_array<int, Dynamic, 1>( pcaResults_py, "pidToResult", NPY_INT );
    PCAResults pcaResults( evals, evects, meanP, pidToResult );

    Vect<unsigned char>::type id = numpy_to_eigen<unsigned char, Dynamic, 1>( id_py, "id", NPY_UBYTE );
    int nPoints = id.rows();
    Vect<double>::type D = numpy_to_eigen<double, Dynamic, 1>( D_py, "D", NPY_DOUBLE, nPoints );
    Vect<double>::type w = numpy_to_eigen<double, Dynamic, 1>( w_py, "w", NPY_DOUBLE, nPoints );
    Mat3<double>::type P = numpy_to_eigen<double, Dynamic, 3>( P_py, "P", NPY_DOUBLE, nPoints );

    return boost::shared_ptr<ComputeLineImage_py>(
                new ComputeLineImage_py( params, P, id, D, w, pcaResults, db, image, nThreads ) );
}



void ComputeLineImage_py::setObjectMask_py( PyObject* mask_py )
{
    checkNumpyArray( mask_py, "mask", NPY_BOOL, nPoints );
    std::vector<bool> mask;
    mask.reserve(nPoints);
    for( int i=0 ; i<nPoints ; i++ )
    {
        mask.push_back( *(char*)PyArray_GETPTR1((PyArrayObject*)mask_py, i) );
    }
    setObjectMask(mask);
}



void ComputeLineImage_py::compute_py( PyObject* R_py, PyObject* P_py,
                                      PyObject* values_py, PyObject* status_py )
{
    int nDims = PyArray_NDIM((PyArrayObject*)P_py);
    if(nDims == 2)
    {
        Mat3<double>::type P = numpy_to_eigen<double, Dynamic, 3>( P_py, "P", NPY_DOUBLE );
        int n = P.rows();
        checkNumpyArray( R_py, "R", NPY_FLOAT, n, 3, 3 );
        checkNumpyArray( values_py, "values", NPY_FLOAT, n, params.nLines );
        checkNumpyArray( status_py, "status", NPY_UBYTE, n, params.nLines );

        ComputeLineImage comp( *dynamic_cast<ComputeLineImage*>(this) );
        Py_BEGIN_ALLOW_THREADS;

        #ifdef LaserLib_USE_OPENMP
        if( nThreads_ > 0 )
            omp_set_num_threads(nThreads_);
        #endif

        int i;
        #pragma omp parallel firstprivate(comp) shared(P)
        for( i=0 ; i<P.rows() ; i++ )
        {
            Eigen::Vector3d p3d = P.row(i);
            float* R_p = (float*)PyArray_GETPTR3((PyArrayObject*)R_py, i, 0, 0);
            Map<Matrix3f> thisR( R_p, 3, 3);
            float* values = (float*)PyArray_GETPTR2((PyArrayObject*)values_py, i, 0);
            unsigned char* status = (unsigned char*)PyArray_GETPTR2((PyArrayObject*)status_py, i, 0);
            comp.compute( thisR, p3d, values, status );
        }
        Py_END_ALLOW_THREADS;
    }
    else if( nDims == 1)
    {
        Map<Matrix3f> R = numpy_to_eigen<float, 3,3>( R_py, "R", NPY_FLOAT );
        Vector3d P = numpy_to_eigen_matrix<double, 3, 1>( P_py, "P", NPY_DOUBLE );

        float* values = numpy_to_ptr<float>( values_py, "values", NPY_FLOAT, params.nLines );
        unsigned char* status = numpy_to_ptr<unsigned char>( status_py, "status", NPY_UBYTE, params.nLines );

        compute( R, P, values, status );
    }
    else
    {
        PyErr_SetString(PyExc_ValueError, "arguments must have 1 or 2 dimensions" );
        throw_error_already_set();
    }
}




LineImageParams_py::LineImageParams_py( PyObject* params )
{
    PyObject* angularSections_py = PyObject_GetAttrString(params, "angularSections");
    if( angularSections_py == NULL )
    {
        PyErr_SetString(PyExc_AttributeError, "LineImageParams: no angularSections attribute" );
        throw_error_already_set();
    }
    if( PyArray_DIM( (PyArrayObject*)angularSections_py, 0 ) == 0 )
    {
        PyErr_SetString(PyExc_ValueError, "angularSections is an empty array!" );
        throw_error_already_set();
    }
    angularSections = numpy_to_std_vector<int>(angularSections_py, "angularSections", NPY_INT);
    Py_DECREF(angularSections_py);

    nLines = 0;
    for( int i=0 ; i<angularSections.size() ; i++ )
    {
        nLines += angularSections[i];
    }
    nRadSections = angularSections.size();

    diskRad = get_attribute_value<double>( params, "diskRad" );
    regionRad = get_attribute_value<double>( params, "regionRad" );
    wNearThresh = get_attribute_value<double>( params, "wNearThresh" );
    angleNoDataThresh = get_attribute_value<double>( params, "angleNoDataThresh" );
    lineLengthToCheck = get_attribute_value<double>( params, "lineLengthToCheck" );
    pcaInterceptThresh = get_attribute_value<double>( params, "pcaInterceptThresh" );
}



