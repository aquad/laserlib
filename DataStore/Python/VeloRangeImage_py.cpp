/*! VeloRangeImage_py.cpp
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
#include "LaserPy/numpy_helpers.h"

#include <cmath>
#include <boost/python.hpp>

#include "VeloRangeImage_py.h"
#include "VelodyneDb_py.h"




void export_VeloRangeImage()
{
    using namespace boost::python;
    class_<VeloRangeImage>("VeloRangeImage", no_init);

    class_< VeloRangeImage_py, boost::shared_ptr<VeloRangeImage_py>, bases<VeloRangeImage> >(
                "VeloRangeImage",
                "VeloRangeImage(xRes, maxPointsPerPixel, db)\n\n"
                "Store point ids in an image.\n\n"
                "Each pixel contains several points. The image goes "
                "vertically from db.vc.min() to db.vc.max(), horizontally from 0 "
                "to 2*pi.\n\n"
                "Parameters\n"
                "----------\n"
                "xRes : int\n"
                "   x resolution\n"
                "maxPointsPerPixel : int\n"
                "db : :class:`pyception.sensors.velodyne.db_xml`\n",
                no_init)

            .def( "__init__", boost::python::make_constructor( &VeloRangeImage_py_constructor ) )

            .def("AddPoints", &VeloRangeImage_py::AddPoints_py,
                 "AddPoints(w,id)\n\n"
                 "Parameters\n"
                 "----------\n"
                 "w : (n,) ndarray, uint16\n"
                 "  azimuth, degrees * 500\n"
                 "id : (n,) ndarray, uint8\n"
                 "  laser id\n")

            .def("AddPointsRad", &VeloRangeImage_py::AddPointsRad_py,
                 "AddPointsRad(w,id)\n\n"
                 "Parameters\n"
                 "----------\n"
                 "w : (n,) ndarray, float64\n"
                 "  azimuth, rad\n"
                 "id : (n,) ndarray, uint8\n"
                 "  laser id\n")

            .def("AddPointsRadEl", &VeloRangeImage_py::AddPointsRadEl_py,
                 "AddPointsRadEl(w,el)\n\n"
                 "Parameters\n"
                 "----------\n"
                 "w : (n,) ndarray, float64\n"
                 "  azimuth, rad\n"
                 "el : (n,) ndarray, float64\n"
                 "  elevation, rad\n")

            .def("Clear", &VeloRangeImage_py::Clear)
            .def("GetNearestPixels", &VeloRangeImage_py::GetNearestPixels_py,
                 "GetNearestPixels(az, el, x, y)\n\n"
                 "Get the nearest pixels to the specified bearing coords\n\n"
                 "Parameters\n"
                 "----------\n"
                 "az : (n,) ndarray, float64\n"
                 "  azimuth (rad)\n"
                 "el : (n,) ndarray float64\n"
                 "  elevation (rad)\n"
                 "x : (n,) ndarray int32\n"
                 "  (output) x pixel coords\n"
                 "y : (n,) ndarray int32\n"
                 "  (output) y pixel coords\n")

            .def("GetNearestPixelsPoints", &VeloRangeImage_py::GetNearestPixelsPoints_py,
                 "GetNearestPixelsPoints(az, el, result, nPoints)\n\n"
                 "Get the points in the nearest pixels to the specified bearing coords\n\n"
                 "Parameters\n"
                 "----------\n"
                 "az : (n,) ndarray, float64\n"
                 "  azimuth (rad)\n"
                 "el : (n,) ndarray float64\n"
                 "  elevation (rad)\n"
                 "result : (n,k) ndarray, int32\n"
                 "   (output) point ids in each pixel, up to a max of k points per pixel\n"
                 "nPoints : (n,) ndarray, int32\n"
                 "   (output) number of points found in each pixel")

            .def("GetPoints", &VeloRangeImage_py::GetPoints_py,
                 "GetPoints(x, y, result, nPoints)\n\n"
                 "Parameters\n"
                 "----------\n"
                 "x : (n,) ndarray, int32\n"
                 "   horizontal pixel indices\n"
                 "y : (n,) ndarray, int32\n"
                 "   vertical pixel indices\n"
                 "result : (n,k) ndarray, int32\n"
                 "   (output) point ids in each pixel, up to a max of k points per pixel\n"
                 "nPoints : (n,) ndarray, int32\n"
                 "   (output) number of points found in each pixel")

            .def("GetPointsPerPixel", &VeloRangeImage_py::GetPointsPerPixel_py,
                 "GetPointsPerPixel()\n\n"
                 "Returns\n"
                 "-------\n"
                 "nPoints : (n,) ndarray, int32\n"
                 "  Number of points in each pixel\n")

            .def("GetAllPoints", &VeloRangeImage_py::GetAllPoints_py,
                 "GetAllPoints(x, y)\n\n"
                 "Get all the points in the specified pixels\n\n"
                 "Parameters\n"
                 "----------\n"
                 "x : (n,) ndarray, int32\n"
                 "   horizontal pixel indices\n"
                 "y : (n,) ndarray, int32\n"
                 "   vertical pixel indices\n\n"
                 "Returns\n"
                 "-------\n"
                 "pids : (n,) ndarray, int32\n"
                 "  Point ids contained in pixels\n")

            .def("Point3dToRangeImage", &VeloRangeImage_py::Point3dToRangeImage_py,
                 "Point3dToRangeImage(p3d, p2d)\n\n"
                 "Convert 3D points to 2D (bearing coords)\n\n"
                 "Parameters\n"
                 "----------\n"
                 "p3d : (n,3) ndarray, float64\n"
                 "p2d : (n,2) ndarray, float64\n"
                 "   (output) bearing coordinates\n")

            .def("GetElevationBorders", &VeloRangeImage_py::GetElevationBorders_py,
                 "GetElevationBorders()\n\n"
                 "Get upper & lower elevations bounding the range image\n\n"
                 "Returns\n"
                 "-------\n"
                 "minEl : double\n"
                 "maxEl : double\n")

            .enable_pickling()
            .def("__getinitargs__", &VeloRangeImage_py::__getinitargs__)
            //.def("__getnewargs__", &VeloRangeImage_py::__getnewargs__)
            .def("__getstate__", &VeloRangeImage_py::__getstate__)
            .def("__setstate__", &VeloRangeImage_py::__setstate__)
            .def_readonly("db", &VeloRangeImage_py::db_pyobj)
            .def_readwrite("image", &VeloRangeImage_py::image_py)
            .def_readwrite("nPointsInPixel", &VeloRangeImage_py::nPointsInPixel_py);


    def("NNImageQuery", &NNImageQuery_py,
        "NNImageQuery(image, image2dPoints, queryPoints)\n\n"
        "For a set of arbitrary 2D (azimuth/elevation) points, find the nearest "
        "neighbours (in 2D) in the range image. Only checks points within the "
        "same grid cell.\n\n"
        "Parameters\n"
        "----------\n"
        "image : :class:`VeloRangeImage`\n"
        "   Range image with points added.\n"
        "image2dPoints : (n,2) ndarray, float64\n"
        "   Azimuth + elevation (rad) array of all points in the range image.\n"
        "queryPoints : (k,2) ndarray, float64\n"
        "   Azimuth + elevation (rad) array of points to match\n\n"
        "Returns\n"
        "-------\n"
        "matches : (k,) ndarray, int32\n"
        "   The indices of nearest neighbours in the scan.\n\n"
        "Notes\n"
        "-----\n"
        "To do simple occupancy checking of a set of 3D 'model' points:\n\n"
        "- Use :meth:`VeloRangeImage.Point3dToRangeImage` to convert arbitrary 3D points to 2D.\n"
        "- Call this function to get 2D neighbours in the scan.\n"
        "- Compute the range of each model point (just the 3D norm).\n\n"
        "- Compute the difference between model and scan ranges. If a model point is far behind a scan point, it's occluded.\n"
        "(note- no associated 'pure c++' function)\n"
        );
}



VeloRangeImage_py::VeloRangeImage_py(int xRes, int maxPointsPerPixel, VelodyneDb& db)
    : VeloRangeImage(xRes, maxPointsPerPixel, db)
{
    //make python objects for pickling
    npy_intp imageDims[3] = {64, xRes, maxPointsPerPixel};
    image_py = (PyArrayObject*) PyArray_SimpleNewFromData(3, imageDims, NPY_INT, image);

    npy_intp nPointsDims[2] = {64, xRes};
    nPointsInPixel_py = (PyArrayObject*) PyArray_SimpleNewFromData(2, nPointsDims, NPY_INT, nPointsInPixel);
}


boost::shared_ptr<VeloRangeImage_py> VeloRangeImage_py_constructor(
        int xRes, int maxPointsPerPixel, PyObject* db_pyobj)
{
    VelodyneDb_py db_py(db_pyobj);
    VelodyneDb& db = *dynamic_cast<VelodyneDb*>( &db_py );
    boost::shared_ptr<VeloRangeImage_py> range_image_py(
        new VeloRangeImage_py( xRes, maxPointsPerPixel, db ) );
    //messy, but no neater way...
    range_image_py->db_pyobj = db_pyobj;
    Py_INCREF(db_pyobj);
    return range_image_py;
}


VeloRangeImage_py::~VeloRangeImage_py()
{
    Py_XDECREF(db_pyobj);
    Py_XDECREF(image_py);
    Py_XDECREF(nPointsInPixel_py);
}


void VeloRangeImage_py::AddPoints_py(PyObject* w_py, PyObject* id_py)
{
    unsigned short* w = numpy_to_ptr<unsigned short>(w_py, "w", NPY_USHORT);
    int n = PyArray_DIM((PyArrayObject*)w_py, 0);
    unsigned char* id = numpy_to_ptr<unsigned char>(id_py, "id", NPY_UBYTE, n);
    AddPoints( w, id, n );
}


void VeloRangeImage_py::AddPointsRad_py(PyObject* w_py, PyObject* id_py)
{
    double* w = numpy_to_ptr<double>(w_py, "w", NPY_DOUBLE);
    int n = PyArray_DIM((PyArrayObject*)w_py, 0);
    unsigned char* id = numpy_to_ptr<unsigned char>(id_py, "id", NPY_UBYTE, n);
    AddPoints( w, id, n );
}

void VeloRangeImage_py::AddPointsRadEl_py(PyObject* w_py, PyObject* el_py)
{
    double* w = numpy_to_ptr<double>(w_py, "w", NPY_DOUBLE);
    int n = PyArray_DIM((PyArrayObject*)w_py, 0);
    double* el = numpy_to_ptr<double>(el_py, "el", NPY_DOUBLE, n);
    AddPoints( w, el, n );
}


void VeloRangeImage_py::GetNearestPixels_py(PyObject* az, PyObject* el, PyObject* x, PyObject* y)
{
    double* azp = numpy_to_ptr<double>( az, "az", NPY_DOUBLE );
    int n = PyArray_DIM((PyArrayObject*)az, 0);
    double* elp = numpy_to_ptr<double>( el, "el", NPY_DOUBLE, n );
    int* xp = numpy_to_ptr<int>( x, "x", NPY_INT, n );
    int* yp = numpy_to_ptr<int>( y, "y", NPY_INT, n );

    for( int i=0 ; i<n ; i++ )
    {
        GetNearestPixel(azp[i], elp[i], xp[i], yp[i]);
    }
}


void VeloRangeImage_py::GetNearestPixelsPoints_py(PyObject* az, PyObject* el, PyObject* result, PyObject* nPoints)
{
    double* azp = numpy_to_ptr<double>( az, "az", NPY_DOUBLE );
    int n = PyArray_DIM((PyArrayObject*)az, 0);
    double* elp = numpy_to_ptr<double>( el, "el", NPY_DOUBLE, n );
    int* resultp = numpy_to_ptr<int>( result, "result", NPY_INT, n );
    int* nPointsp = numpy_to_ptr<int>( nPoints, "nPoints", NPY_INT, n );

    int maxPoints = PyArray_DIM((PyArrayObject*)result, 1);
    for( int i=0 ; i<n ; i++ )
    {
        int* imp;
        GetNearestPixelsPoints(azp[i], elp[i], imp, nPointsp[i]);
        for( int j=0 ; j<nPointsp[i] ; j++ )
        {
            resultp[i*maxPoints + j] = imp[j];
        }
    }
}


void VeloRangeImage_py::GetPoints_py(PyObject* x, PyObject* y, PyObject* result, PyObject* nPoints)
{
    int* xp = numpy_to_ptr<int>( x, "x", NPY_INT );
    int n = PyArray_DIM((PyArrayObject*)x, 0);
    int* yp = numpy_to_ptr<int>( y, "y", NPY_INT, n );
    int* resultp = numpy_to_ptr<int>( result, "result", NPY_INT, n, -2 );
    int* nPointsp = numpy_to_ptr<int>( nPoints, "nPoints", NPY_INT, n );

    int maxPoints = 1;
    if( PyArray_NDIM( (PyArrayObject*)result ) == 2 )
        maxPoints = PyArray_DIM((PyArrayObject*)result, 1);
    for( int i=0 ; i<n ; i++ )
    {
        int* imp;
        GetPoints(xp[i], yp[i], imp, nPointsp[i]);
        for( int j=0 ; j<nPointsp[i] ; j++ )
        {
            resultp[i*maxPoints + j] = imp[j];
        }
    }
}


PyObject* VeloRangeImage_py::GetPointsPerPixel_py()
{
    npy_intp dims[1] = {xRes * 64};
    PyArrayObject *ppp = (PyArrayObject*)PyArray_SimpleNewFromData( 1, dims, NPY_INT, nPointsInPixel );
    Py_INCREF(ppp);
    return PyArray_Return(ppp);
}


PyObject* VeloRangeImage_py::GetAllPoints_py(PyObject* x, PyObject* y)
{
    int* xp = numpy_to_ptr<int>( x, "x", NPY_INT );
    int n = PyArray_DIM((PyArrayObject*)x, 0);
    int* yp = numpy_to_ptr<int>( y, "y", NPY_INT, n );

    std::vector<int> points;
    points.reserve(200);
    for( int i=0 ; i<n ; i++ )
    {
        int* imp;
        int nPoints;
        GetPoints(xp[i], yp[i], imp, nPoints);
        for( int j=0 ; j<nPoints ; j++ )
        {
            points.push_back(imp[j]);
        }
    }

    npy_intp dims[2] = {static_cast<npy_intp>(points.size())};
    PyArrayObject *points_py = (PyArrayObject*)PyArray_SimpleNew(1, dims, NPY_INT);
    memcpy( PyArray_DATA(points_py), &(points[0]), points.size()*sizeof(int) );
    return PyArray_Return(points_py);
}




void VeloRangeImage_py::Point3dToRangeImage_py( PyObject* p3d_py, PyObject* p2d_py )
{
    double* p3d_p = numpy_to_ptr<double>( p3d_py, "p3d", NPY_DOUBLE, -1, 3 );
    int n = PyArray_DIM((PyArrayObject*)p3d_py, 0);
    double* p2d_p = numpy_to_ptr<double>( p2d_py, "p2d", NPY_DOUBLE, n, 2 );
    for( int i=0 ; i<n ; i++ )
    {
        Point3dToRangeImage( p3d_p + i*3, p2d_p + i*2 );
    }
}


boost::python::tuple VeloRangeImage_py::GetElevationBorders_py()
{
    return boost::python::make_tuple( elBounds.front(), elBounds.back() );
}


PyObject* VeloRangeImage_py::__getinitargs__()
{
    PyObject* tup = PyTuple_Pack(3, PyInt_FromLong(xRes), PyInt_FromLong(maxPointsPerPixel), db_pyobj);
    return tup;
}


PyObject* VeloRangeImage_py::__getstate__()
{
    PyObject* tup = PyTuple_Pack(2, image_py, nPointsInPixel_py);
    return tup;
}

void VeloRangeImage_py::__setstate__(PyObject* state)
{
    PyArrayObject* state_image_py = (PyArrayObject*) PyTuple_GetItem(state,0);
    PyArrayObject* state_nPointsInPixel_py = (PyArrayObject*) PyTuple_GetItem(state,1);
    //copy across image data
    memcpy(image, PyArray_DATA(state_image_py), maxPointsPerPixel*xRes*64*sizeof(int) );
    memcpy(nPointsInPixel, PyArray_DATA(state_nPointsInPixel_py), xRes*64*sizeof(int) );
}



PyObject* NNImageQuery_py( VeloRangeImage& image, PyObject* image2dPoints_py, PyObject* queryPoints_py)
{
    //for each query point, find the nearest 2d neighbour within the associated pixel.
    double* image2dPoints = numpy_to_ptr<double>( image2dPoints_py, "queryPoints", NPY_DOUBLE, -1, 2 );
    double* queryPoints = numpy_to_ptr<double>( queryPoints_py, "queryPoints", NPY_DOUBLE, -1, 2 );
    int nQuery = PyArray_DIM((PyArrayObject*)queryPoints_py, 0);

    //result arrays:
    npy_intp dims[1] = {nQuery};
    PyArrayObject* ind_py = (PyArrayObject*) PyArray_SimpleNew(1, dims, NPY_INT);
    PyArrayObject* valid_py = (PyArrayObject*) PyArray_SimpleNew(1, dims, NPY_BOOL);

    int* ind = (int*)PyArray_DATA(ind_py);
    bool* valid = (bool*)PyArray_DATA(valid_py);

    for( int i=0 ; i<nQuery ; i++ )
    {
        int nPoints=0;
        int* result;
        double qAz = queryPoints[i*2];
        double qEl = queryPoints[i*2 + 1];
        image.GetNearestPixelsPoints( qAz, qEl, result, nPoints );
        double minDiff = 10;
        int minId = -1;
        for( int j=0 ; j<nPoints ; j++ )
        {
            int id = result[j];
            double azDiff = image2dPoints[id] - qAz;
            double elDiff = image2dPoints[id+1] - qEl;
            double dist = sqrt( std::pow(azDiff,2) + std::pow(elDiff,2) );
            if( dist < minDiff )
            {
                minDiff = dist;
                minId = id;
            }
        }
        if( minId == -1 )
        {
            valid[i] = false;
        }
        else
        {
            valid[i] = true;
            ind[i] = minId;
        }
    }

    PyObject* tup = PyTuple_Pack(2, ind_py, valid_py);
    Py_DECREF(ind_py);
    Py_DECREF(valid_py);
    return tup;
}

