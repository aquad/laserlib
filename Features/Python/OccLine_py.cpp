/*! OccLine_py.cpp
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
 * \date       05-12-2012
*/

#include <Python.h>
#define PY_ARRAY_UNIQUE_SYMBOL _LaserPy_ARRAY_API
#define NO_IMPORT_ARRAY
#include "LaserPy/numpy_to_eigen.h"
#include "LaserPy/numpy_to_std.h"

#include <iostream>
#include <boost/python.hpp>

#include "OccLine_py.h"
#include "DataStore/VeloRangeImage.h"
#include "DataStore/Python/VelodyneDb_py.h"
#include "Common/ProgressIndicator.h"


using namespace boost::python;


void export_OccLine()
{
    class_<OccLine_py, boost::shared_ptr<OccLine_py> >("OccLine",
            "OccLine(params, P, id, D, w, pcaResults, db, image)\n\n"
            "Given a 3D line segment 'probing' a region of data, traverse along "
            "it looking for a surface intercept. Detects occupancy (line is "
            "empty, intercepts a surface, or encounters unknown space). This "
            "forms the key component of the Line Image feature.\n"
            ":meth:`compute` is the main function to use, the rest are for "
            "debugging/visualisation.\n\n"
            "Parameters\n"
            "----------\n"
            "params : :class:`~pyception.algorithms.line_image.OccLineParams`\n"
            "P : ndarray float64 (n,3)\n"
            "   3D points\n"
            "id : ndarray uint8 (n,)\n"
            "   laser ids [0-63]\n"
            "D : ndarray float64 (n,)\n"
            "   range (m)\n"
            "w : ndarray float64 (n,)\n"
            "   azimuth (rad)\n"
            "pcaResults : :class:`~pyception.algorithms.line_image.PCAResults`\n"
            "db : :class:`~pyception.sensors.velodyne.db_xml`\n"
            "image : :class:`VeloRangeImage`\n",
            no_init)

            .def( "__init__", boost::python::make_constructor( &OccLine_py_constructor ) )

            .def("setObjectMask", &OccLine_py::setObjectMask_py,
                 "setObjectMask(mask)\n\n"
                 "If you have a target object, and don't want intercepts "
                 "elsewhere, provide a mask.\n\n"
                 "Parameters\n"
                 "----------\n"
                 "mask : ndarray bool (n,)\n")

            .def("compute", &OccLine_py::compute_py,
                 "compute(start, end)\n\n"
                 "Compute one or multiple occupancy line(s). Lines 'start' "
                 "below a surface, and end above it (defines the 'order of "
                 "traversal')\n\n"
                 "Parameters\n"
                 "----------\n"
                 "start : ndarray float32; (3,) or (k,3)\n"
                 "  start 3D point of line (below surface)\n"
                 "end : ndarray float32; (3,) or (k,3)\n"
                 "  end 3D point of line (above surface)\n\n"
                 "Returns\n"
                 "-------\n"
                 "value : float, or ndarray float32 (k,)\n"
                 "  Depth (m) of intercept / unknown region. 0 = centre (between "
                 "  *start* and *end*), +ve is towards the line end (above the surface).\n"
                 "status : int, or ndarray uint8 (k,)\n"
                 "  0 = unknown, 1 = empty, 2 = surface intercept\n")

            .def("lineTrace", &OccLine_py::lineTrace_py,
                 "lineTrace(bounds)\n\n"
                 "Trace a 2D (range image) line defined by angular *bounds*, "
                 "selecting the pixels.\n\n"
                 "Parameters\n"
                 "----------\n"
                 "bounds : ndarray float32 (4,)\n\n"
                 "  Defines a 2D angular line (azimuth1, elevation1), "
                 "  (azimuth2, elevation2)\n\n"
                 "Returns\n"
                 "-------\n"
                 "pixels : ndarray int32 (k,2)\n\n"
                 "  x,y pixels along the line\n")

            .def("getPointsOnLine", &OccLine_py::getPointsOnLine_py,
                 "getPointsOnLine(bounds)\n\n"
                 "Given a 2D (range image) line defined by angular *bounds*, "
                 "find all point ids near it. This is equivalent to :meth:`lineTrace` "
                 "followed by :meth:`VeloRangeImage.GetAllPoints`\n\n"
                 "Parameters\n"
                 "----------\n"
                 "bounds : ndarray float32 (4,)\n\n"
                 "  Defines a 2D angular line (azimuth1, elevation1), "
                 "  (azimuth2, elevation2)\n\n"
                 "Returns\n"
                 "-------\n"
                 "pids : ndarray int32 (k,)\n\n"
                 "  point ids along the line\n")

            .def("orderPointsOnLine", &OccLine_py::orderPointsOnLine_py,
                 "orderPointsOnLine(start, end, pointsOnLine)\n\n"
                 "Order the points retrieved along the line, such that (in 2D) "
                 "they proceed from the start to the end. They are also filtered "
                 "(removed in further than *wNearThresh*, see "
                 ":class:`~pyception.algorithms.line_image.OccLineParams`).\n\n"
                 "Parameters\n"
                 "----------\n"
                 "start : ndarray float32; (3,)\n"
                 "  start 3D point of line (below surface)\n"
                 "end : ndarray float32; (3,)\n"
                 "  end 3D point of line (above surface)\n"
                 "pointsOnLine : ndarray int32 (k,)\n"
                 "  Point ids along the line (from :meth:`getPointsOnLine`)\n\n"
                 "Returns\n"
                 "-------\n"
                 "lineT : ndarray float32 (l,)\n"
                 "  These indicate where each point (*linePid*) is along the line. "
                 "  The line is defined as p = n*lineT + start, where n is the "
                 "  normalised vector from *start* to *end*.\n"
                 "linePid : ndarray int32 (l,)\n"
                 "  Point ids (ordered, aligned to *lineT*)")

            .def("traverseLineForIntercept", &OccLine_py::traverseLineForIntercept_py,
                 "traverseLineForIntercept(lineT, linePid, start, end)\n\n"
                 "Called after :meth:`orderPointsOnLine`.\n\n"
                 "Parameters\n"
                 "----------\n"
                 "lineT : ndarray float32 (l,)\n"
                 "  Point locations along the line, see :meth:`orderPointsOnLine`.\n"
                 "linePid : ndarray int32 (l,)\n"
                 "  Point ids (ordered, aligned to *lineT*)\n"
                 "start : ndarray float32; (3,)\n"
                 "  start 3D point of line (below surface)\n"
                 "end : ndarray float32; (3,)\n"
                 "  end 3D point of line (above surface)\n\n"
                 "Returns\n"
                 "-------\n"
                 "closestPid : int\n"
                 "  For intercepts, this provides the associated point. The "
                 "  surface 'model' is a gaussian function attached to most points "
                 "  (computed from PCA). See construction argument *pcaResults*.\n"
                 "value : float\n"
                 "  Depth (m) of intercept / unknown region. 0 = centre (between "
                 "  *start* and *end*), +ve is towards the line end (above the surface).\n"
                 "status : int\n"
                 "  0 = unknown, 1 = empty, 2 = surface intercept\n");


    enum_<LineStatus>("LineStatus")
        .value("unknown", UNKNOWN)
        .value("empty", EMPTY)
        .value("value", VALUE);
}


using namespace Eigen;



OccLine_py::OccLine_py( OccLineParams& params, Mat3<double>::type& P,
        Vect<unsigned char>::type& id, Vect<double>::type& D, Vect<double>::type& w,
        PCAResults& pcaResults, VelodyneDb& db, VeloRangeImage& image )
    :   OccLine( params, P, id, D, w, pcaResults, db, image )
{}



boost::shared_ptr<OccLine_py> OccLine_py_constructor(
        PyObject* params_py, PyObject* P_py, PyObject* id_py,
        PyObject* D_py, PyObject* w_py,
        PyObject* pcaResults_py, PyObject* db_pyobj, VeloRangeImage& image )
{
    //these are temporary, but they get copied in the ComputeLineImage constructor.
    VelodyneDb_py db_py(db_pyobj);
    VelodyneDb& db = *dynamic_cast<VelodyneDb*>(&db_py);
    OccLineParams_py params_py_derived(params_py);
    OccLineParams& params = *dynamic_cast<OccLineParams*>(&params_py_derived);

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

    return boost::shared_ptr<OccLine_py>(
            new OccLine_py( params, P, id, D, w, pcaResults, db, image ) );
}



void OccLine_py::setObjectMask_py( PyObject* mask_py )
{
    checkNumpyArray( mask_py, "mask", NPY_BOOL, P.rows() );
    std::vector<bool> mask;
    mask.reserve(P.rows());
    for( int i=0 ; i<P.rows() ; i++ )
    {
        mask.push_back( *(char*)PyArray_GETPTR1((PyArrayObject*)mask_py, i) );
    }
    setObjectMask(mask);
}



PyObject* OccLine_py::compute_py( PyObject* start_py, PyObject* end_py )
{
    int nDims = PyArray_NDIM((PyArrayObject*)start_py);
    if(nDims == 2)
    {
        Mat3<float>::type start = numpy_to_eigen<float, Dynamic, 3>( start_py, "start", NPY_FLOAT );
        int nLines = start.rows();
        Mat3<float>::type end = numpy_to_eigen<float, Dynamic, 3>( end_py, "end", NPY_FLOAT, nLines );

        npy_intp dims[2] = {nLines, nLines};
        PyArrayObject* values_py = (PyArrayObject*) PyArray_SimpleNew(1, dims, NPY_FLOAT);
        PyArrayObject* status_py = (PyArrayObject*) PyArray_SimpleNew(1, dims, NPY_UBYTE);
        float* values = (float*)PyArray_DATA(values_py);
        unsigned char* status = (unsigned char*)PyArray_DATA(status_py);

        //self-copy for OMP (not really worth it)
        //OccLine occLine( *dynamic_cast<OccLine*>(this) );
        for( int i=0 ; i<nLines ; i++ )
        {
            compute( start.row(i), end.row(i), values[i], status[i] );
        }

        PyObject* tup = PyTuple_Pack(2, values_py, status_py);
        return tup;
    }
    else if( nDims == 1)
    {
        Vector3f start = numpy_to_eigen_matrix<float, 3, 1>( start_py, "start", NPY_FLOAT );
        Vector3f end = numpy_to_eigen_matrix<float, 3, 1>( end_py, "end", NPY_FLOAT );
        float value;
        unsigned char status;
        compute( start, end, value, status );

        PyObject* tup = PyTuple_Pack(2, PyFloat_FromDouble(value), PyInt_FromLong(status));
        return tup;
    }
    else
    {
        PyErr_SetString(PyExc_ValueError, "arguments must have 1 or 2 dimensions" );
        throw_error_already_set();
    }
}



PyObject* OccLine_py::lineTrace_py( PyObject* bounds_py )
{
    float* bounds = numpy_to_ptr<float>( bounds_py, "bounds", NPY_FLOAT, 4 );
    std::vector< std::pair<int,int> > pixels;
    pixels.reserve(100);

    lineTraceFat( bounds, pixels );

    npy_intp dims[2] = {static_cast<npy_intp>(pixels.size()), 2};
    PyArrayObject *pixels_py = (PyArrayObject*) PyArray_SimpleNew(2, dims, NPY_INT);
    memcpy( PyArray_DATA(pixels_py), &(pixels[0]), pixels.size()*2*sizeof(int) );
    return PyArray_Return(pixels_py);
}



PyObject* OccLine_py::getPointsOnLine_py( PyObject* bounds_py )
{
    float* bounds_p = numpy_to_ptr<float>( bounds_py, "bounds", NPY_FLOAT, 4 );
    std::vector<int> points;
    getPointsOnLine( bounds_p, points );

    npy_intp dims[1] = {static_cast<npy_intp>(points.size())};
    PyArrayObject* points_py;
    if( points.size()==0 )
    {
        points_py = (PyArrayObject*)PyArray_EMPTY(1, dims, NPY_INT, 0);
    }
    else
    {
        points_py = (PyArrayObject*)PyArray_SimpleNew(1, dims, NPY_INT);
        memcpy( PyArray_DATA(points_py), &points[0], points.size()*sizeof(int) );
    }
    return PyArray_Return(points_py);
}


PyObject* OccLine_py::orderPointsOnLine_py( PyObject* start_py, PyObject* end_py,
        PyObject* pointsOnLine_py )
{
    Vector3f start = numpy_to_eigen_matrix<float, 3, 1>( start_py, "start", NPY_FLOAT );
    Vector3f end = numpy_to_eigen_matrix<float, 3, 1>( end_py, "end", NPY_FLOAT );
    LineCoords line( start, end, image );

    std::vector<int> pointsOnLine = numpy_to_std_vector<int>( pointsOnLine_py, "pointsOnLine", NPY_INT );
    pointIds.clear();
    lineTs.clear();
    orderPointsOnLine( line, pointsOnLine, pointIds, lineTs );

    npy_intp dims[1] = {static_cast<npy_intp>(pointIds.size())};
    PyArrayObject* lineT_py = (PyArrayObject*) PyArray_SimpleNew(1, dims, NPY_FLOAT);
    PyArrayObject* linePid_py = (PyArrayObject*) PyArray_SimpleNew(1, dims, NPY_INT);
    memcpy( PyArray_DATA(linePid_py), &pointIds[0], pointIds.size()*sizeof(int) );
    memcpy( PyArray_DATA(lineT_py), &lineTs[0], lineTs.size()*sizeof(float) );
    PyObject* tup = PyTuple_Pack(2, lineT_py, linePid_py);
    return tup;
}



boost::python::tuple OccLine_py::traverseLineForIntercept_py(
        PyObject* lineT_py, PyObject* linePid_py, PyObject* start_py, PyObject* end_py )
{
    lineTs = numpy_to_std_vector<float>( lineT_py, "lineT", NPY_FLOAT );
    pointIds = numpy_to_std_vector<int>( linePid_py, "linePid", NPY_INT );
    if( pointIds.size() != lineTs.size() )
    {
        PyErr_SetString(PyExc_ValueError, "lineT must be same length as linePid");
        throw_error_already_set();
    }
    Vector3f start = numpy_to_eigen_matrix<float, 3, 1>( start_py, "start", NPY_FLOAT );
    Vector3f end = numpy_to_eigen_matrix<float, 3, 1>( end_py, "end", NPY_FLOAT );
    LineCoords line( start, end, image );

    float value = line.length/2;
    unsigned char status = EMPTY;
    int closestPid;
    traverseLineForIntercept( pointIds, lineTs, line, closestPid, value, status );
    return make_tuple( closestPid, value, status );
}



OccLineParams_py::OccLineParams_py( PyObject* params )
{
    wNearThresh = get_attribute_value<double>(params, "wNearThresh");
    angleNoDataThresh = get_attribute_value<double>(params, "angleNoDataThresh");
    lineLengthToCheck = get_attribute_value<double>(params, "lineLengthToCheck");
    pcaInterceptThresh = get_attribute_value<double>(params, "pcaInterceptThresh");
}



