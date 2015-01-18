/*! VeloImageSelect_py.cpp
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
 * \date       23-05-2011
*/

#include <Python.h>
#define PY_ARRAY_UNIQUE_SYMBOL _LaserPy_ARRAY_API
#define NO_IMPORT_ARRAY
#include "LaserPy/numpy_to_eigen.h"
#include "LaserPy/numpy_helpers.h"
#include <boost/python.hpp>
#include "Common/ArrayTypes.h"
#include "VeloImageSelect_py.h"
#include "VelodyneDb_py.h"



void export_VeloImageSelect()
{
    using namespace boost::python;

    //not really used- might delete
    class_<ImagePlusSelector_py, boost::shared_ptr<ImagePlusSelector_py>, bases<Selector> >("ImagePlusSelector", no_init)
            .def( "__init__", boost::python::make_constructor( &ImagePlusSelector_py_constructor ) )
            .def("SelectRegion", &ImagePlusSelector_py::SelectRegion_py)
            .def("SetRadius", &ImagePlusSelector_py::SetRadius);


    class_< ImageSphereSelector_py, boost::shared_ptr<ImageSphereSelector_py>, bases<Selector> >("ImageSphereSelector", no_init)
            .def( "__init__", boost::python::make_constructor( &ImageSphereSelector_py_constructor ),
                  "ImageSphereSelector(image, db, w, id, D, P, rad)\n\n"
                  "Select spherical regions using the range image.\n\n"
                  "Parameters\n"
                  "----------\n"
                  "image : :class:`VeloRangeImage`\n"
                  "db : :class:`pyception.sensors.velodyne.db_xml`\n"
                  "w : ndarray (n,) float64\n"
                  " azimuth angles, rad\n"
                  "id : ndarray (n,) uint8\n"
                  " laser ids [0-63]\n"
                  "D : ndarray (n,) float64\n"
                  " depth, m\n"
                  "P : ndarray (n,3), float64\n"
                  " 3D points\n"
                  "rad : float\n"
                  " radius (m) of selection region\n")

            .def("SelectRegion", &ImageSphereSelector_py::SelectRegion_py,
                 "SelectRegion(centre)\n\n"
                 "Parameters\n"
                 "----------\n"
                 "centre : int\n"
                 "  point id of sphere centre\n\n"
                 "Returns\n"
                 "-------\n"
                 "region : ndarray (m,) int32\n"
                 "  point ids in region\n")

            .def("SetRadius", &ImageSphereSelector_py::SetRadius,
                 "SetRadius( rad )\n");


    def("RangeImageMatch", &RangeImageMatch_py,
        "RangeImageMatch(image, t, P, srcLid, srcW, srcT, srcP)\n\n"
        "Find the matching points from a subset (eg between scan and segmented object).\n\n"
        ":param image: :class:`VeloRangeImage` with data added\n"
        ":param t: (n,) datetime64, time, aligned with the range image data\n"
        ":param P: (n,3) float64, points, aligned with the range image data\n"
        ":param srcLid: (m,) uint8, laser ids of points to find\n"
        ":param srcW: (m,) float64, azimuth (rad) of points to find\n"
        ":param srcT: (m,) datetime64, time of points to find\n"
        ":param srcP: (m,3) float64, points to find\n"
        ":returns: (m,) array of matching point ids, int32\n");
}


using namespace Eigen;


boost::shared_ptr<ImagePlusSelector_py> ImagePlusSelector_py_constructor( VeloRangeImage& image, PyObject* db_pyobj,
            PyObject* w_py, PyObject* id_py, PyObject* D_py, double rad )
{
    VelodyneDb_py db_py(db_pyobj);
    VelodyneDb& db = *dynamic_cast<VelodyneDb*>(&db_py);
    double* w = numpy_to_ptr<double>(w_py, "w", NPY_DOUBLE);
    int n = PyArray_DIM((PyArrayObject*)w_py, 0);
    unsigned char* id = numpy_to_ptr<unsigned char>(id_py, "id", NPY_UBYTE, n);
    double* D = numpy_to_ptr<double>(D_py, "D", NPY_DOUBLE, n);

    return boost::shared_ptr<ImagePlusSelector_py>( new ImagePlusSelector_py( image, db, w, id, D, rad ) );
}




boost::shared_ptr<ImageSphereSelector_py> ImageSphereSelector_py_constructor( VeloRangeImage& image, PyObject* db_pyobj,
            PyObject* w_py, PyObject* id_py, PyObject* D_py, PyObject* P_py, double rad )
{
    VelodyneDb_py db_py(db_pyobj);
    VelodyneDb& db = *dynamic_cast<VelodyneDb*>(&db_py);

    double* w = numpy_to_ptr<double>(w_py, "w", NPY_DOUBLE);
    int n = PyArray_DIM((PyArrayObject*)w_py, 0);
    unsigned char* id = numpy_to_ptr<unsigned char>(id_py, "id", NPY_UBYTE, n);
    double* D = numpy_to_ptr<double>(D_py, "D", NPY_DOUBLE, n);
    Mat3<double>::type P = numpy_to_eigen<double, Dynamic, 3>( P_py, "P", NPY_DOUBLE, n );

    return boost::shared_ptr<ImageSphereSelector_py>( new ImageSphereSelector_py( image, db, w, id, D, P, rad ) );
}



PyObject* RangeImageMatch_py( VeloRangeImage& image, PyObject* t_py, PyObject* P_py,
                              PyObject* srcLid_py, PyObject* srcW_py, PyObject* srcT_py, PyObject* srcP_py )
{
    MapVecXll t = numpy_to_eigen<long long, Dynamic, 1>( t_py, "t", NPY_DATETIME );
    Mat3<double>::type P = numpy_to_eigen<double, Dynamic, 3>( P_py, "P", NPY_DOUBLE, t.rows() );
    MapVecXuc srcLid = numpy_to_eigen<unsigned char, Dynamic, 1>( srcLid_py, "srcLid", NPY_UBYTE );
    int nSrc = srcLid.rows();
    MapVecXd srcW = numpy_to_eigen<double, Dynamic, 1>( srcW_py, "srcW", NPY_DOUBLE, nSrc );
    MapVecXll srcT = numpy_to_eigen<long long, Dynamic, 1>( srcT_py, "srcT", NPY_DATETIME, nSrc );
    Mat3<double>::type srcP = numpy_to_eigen<double, Dynamic, 3>( srcP_py, "srcP", NPY_DOUBLE, nSrc );

    npy_intp dims[1] = {static_cast<npy_intp>(srcLid.size())};
    PyArrayObject* matches_py = (PyArrayObject *)PyArray_SimpleNew(1, dims, NPY_INT );
    MapVecXi matches( (int *) PyArray_DATA(matches_py), PyArray_DIM(matches_py,0) );

    RangeImageMatch( image, t, P, srcLid, srcW, srcT, srcP, matches );
    return PyArray_Return(matches_py);
}

