/*! BearingGraph_py.cpp
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
 * \date       18-05-2011
*/

#include <Python.h>
#define PY_ARRAY_UNIQUE_SYMBOL _LaserPy_ARRAY_API
#define NO_IMPORT_ARRAY
#include "LaserPy/numpy_to_eigen.h"
#include "LaserPy/numpy_helpers.h"
#include <boost/python.hpp>

#include "BearingGraph_py.h"
#include "DataStore/BearingGraph.h"
#include "VelodyneDb_py.h"
#include "Features/GraphSurfNorm.h"
#include "Common/ArrayTypes.h"


void export_BearingGraph()
{
    using namespace boost::python;

    class_<BearingGraphBuilder_py, boost::noncopyable>(
            "BearingGraphBuilder",
            "BearingGraphBuilder(db[, maxNumPoints=300000, wThresh=0.0349])\n\n"
            "Builds a bearing graph from velodyne data.\n\n"
            "Parameters\n"
            "----------\n"
            "db : :class:`~pyception.sensors.velodyne.db_xml` \n"
            "maxNumPoints : int\n"
            "    Maximum number of points in a velodyne scan (buffer size)\n"
            "wThresh : float\n"
            "    Angular threshold (rad), don't connect points with "
            "a larger difference in azimuth.\n",
            init<PyObject*, optional<unsigned int, float> >() )

            .def("BuildGraph", &BearingGraphBuilder_py::BuildGraph,
                 "BuildGraph(id, w)\n\n"
                 "Build the graph (connect all neighbouring points)\n\n"
                 "Parameters\n"
                 "----------\n"
                 "id : ndarray (n,) uint8\n"
                 "    laser ids [0-63]\n"
                 "w : ndarray (n,) float64\n"
                 "    azimuth (rad)\n\n"
                 "Returns\n"
                 "-------\n"
                 "graph : ndarray (n,4) int32\n"
                 "    graph structure. For example, row 6 indicates the (left, "
                 "right, up, down) neighbours of point 6. -1 indicates no neighbour. This "
                 "array is owned by the class, and overwritten on the next call "
                 "to BuildGraph.")

            .def("CleanGraph", &BearingGraphBuilder_py::CleanGraph, CleanGraph_overloads(
                 "CleanGraph(P, D[, maxLength=5.0, relThresh=3.0, convThresh=0.5])\n\n"
                 "Remove long links in an attempt to separate foreground & background. "
                 "A number of parameters specify whether to disconnect a link.\n\n"
                 "Parameters\n"
                 "----------\n"
                 "P : ndarray (n,3) float64\n"
                 "    3D points\n"
                 "D : ndarray (n,) float64\n"
                 "    Velodyne range (metres)\n"
                 "maxLength : float\n"
                 "    Disconnect if longer than this (m)\n"
                 "relThresh : float\n"
                 "    Disconnect if the link is more than this many times "
                 "longer than the one in the opposite direction (eg. horizontal "
                 "or vertical link).\n"
                 "convThresh : float\n"
                 "    Disconnect if both are more than this long, and convex/concave\n\n"
                 "Returns\n"
                 "-------\n"
                 "graph : ndarray (n,4) int32") )

            .def("CleanGraphFast", &BearingGraphBuilder_py::CleanGraphFast, CleanGraphFast_overloads(
                 "CleanGraphFast(Dint[, maxLength=5.0*500, relThresh=3.0, convThresh=0.5*500])\n\n"
                 "Older version of CleanGraph, tried to do things fast using"
                 "only integer ranges, but actually fails quite a bit.\n\n"
                 "Parameters\n"
                 "----------\n"
                 "Dint : ndarray (n,) uint16\n"
                 "    Velodyne range (metres * 500)\n"
                 "maxLength : int\n"
                 "    Disconnect if longer than this (m*500)\n"
                 "relThresh : float\n"
                 "    Disconnect if the link is more than this many times "
                 "longer than the one in the opposite direction (eg. horizontal "
                 "or vertical link).\n"
                 "convThresh : int\n"
                 "    Disconnect if both are more than this long, and convex/concave\n\n"
                 "Returns\n"
                 "-------\n"
                 "graph : ndarray (n,4) int32") )


            .enable_pickling()
            .def("__getinitargs__", &BearingGraphBuilder_py::__getinitargs__);


    enum_<Direction>("Direction")
        .value("LEFT", LEFT)
        .value("RIGHT", RIGHT)
        .value("UP", UP)
        .value("DOWN", DOWN);


    def("CalcSurfNorms", &CalcSurfNorms_py,
        "CalcSurfNorms(neighs, P, sn, valid)\n\n"
        "Calculate surface normals using the graph (cross product method from "
        "[moosmann2009segmentation]_)\n\n"
        "Parameters\n"
        "----------\n"
        "neighs : (n,4) ndarray, int32\n"
        "   Graph structure (preferably nice and cleaned)\n"
        "P : (n,3) ndarray, float64\n"
        "   3D points\n"
        "sn : (n,3) ndarray, float64\n"
        "   (output) surface normals. Will be (0,0,0) where the calculation failed\n"
        "valid : (n,) ndarray, bool\n"
        "   (output) Indicates where the surface normals are valid. Some points will "
        "have insufficient neighbours to calculate a surface normal\n");


    def("BlurSurfNorms", &BlurSurfNorms_py, BlurSurfNorms_overloads(
        "BlurSurfNorms(neighs, P, sn, valid, sn_blurred[, sd=1.0])\n\n"
        "Gaussian blur surface normals based on 3D distance & the connectivity graph\n\n"
        "Parameters\n"
        "----------\n"
        "neighs : (n,4) ndarray, int32\n"
        "   Graph structure\n"
        "P : (n,3) ndarray, float64\n"
        "   3D points\n"
        "sn : (n,3) ndarray, float64\n"
        "   surface normals to blur\n"
        "valid : (n,) ndarray, bool\n"
        "   Valid surface normals (as from :func:`CalcSurfNorms`)\n"
        "sn_blurred : (n,3) ndarray, float64\n"
        "   (output) blurred surface normals\n"
        "sd : double\n"
        "   Standard deviation of gaussian function used in blurring") );
}


using namespace Eigen;


BearingGraphBuilder_py::BearingGraphBuilder_py( PyObject* db_pyobj, unsigned int maxNumPoints, float wThresh )
    :   db_pyobj_(db_pyobj),
        maxNumPoints_(maxNumPoints),
        wThresh_(wThresh)
{
    VelodyneDb_py db_py(db_pyobj);
    VelodyneDb& db = *dynamic_cast<VelodyneDb*>(&db_py);
    bearGraph.reset( new BearingGraphBuilder(db, maxNumPoints, wThresh) );
}


PyObject* BearingGraphBuilder_py::BuildGraph( PyObject* id_py, PyObject* w_py )
{
    PyObject* graph_py;
    unsigned char* id_p = numpy_to_ptr<unsigned char>( id_py, "id", NPY_UBYTE );
    unsigned int nPoints = PyArray_DIM((PyArrayObject*)id_py, 0);
    double* w_p = numpy_to_ptr<double>( w_py, "w", NPY_DOUBLE, nPoints );

    Py_BEGIN_ALLOW_THREADS;
    bearGraph->BuildGraph(id_p, w_p, nPoints);
    Py_END_ALLOW_THREADS;

    npy_intp dims[2] = {nPoints, 4};
    graph_py = PyArray_SimpleNewFromData(2, dims, NPY_INT, bearGraph->GetGraphPtr());
    return graph_py;
}



PyObject* BearingGraphBuilder_py::CleanGraph( PyObject* P_py, PyObject* D_py, float maxLength, float relThresh, float convThresh )
{
    Mat3<double>::type P = numpy_to_eigen<double,Dynamic,3>( P_py, "P", NPY_DOUBLE );
    MapVecXd D = numpy_to_eigen<double,Dynamic,1>( D_py, "D", NPY_DOUBLE, P.rows() );

    Py_BEGIN_ALLOW_THREADS;
    bearGraph->CleanGraph( P, D, maxLength, relThresh, convThresh );
    Py_END_ALLOW_THREADS;

    npy_intp dims[2] = {P.rows(), 4};
    PyObject* graph_py = PyArray_SimpleNewFromData(2, dims, NPY_INT, bearGraph->GetGraphPtr());
    return PyArray_Return( (PyArrayObject*)graph_py );
}



PyObject* BearingGraphBuilder_py::CleanGraphFast( PyObject* D_py, int maxLength, float relThresh, int convThresh )
{
    unsigned short* D_p = numpy_to_ptr<unsigned short>( D_py, "D", NPY_USHORT );
    unsigned int nPoints = PyArray_DIM((PyArrayObject*)D_py, 0);

    Py_BEGIN_ALLOW_THREADS;
    bearGraph->CleanGraphFast( D_p, nPoints, maxLength, relThresh, convThresh );
    Py_END_ALLOW_THREADS;

    npy_intp dims[2] = {nPoints, 4};
    PyObject* graph_py = PyArray_SimpleNewFromData(2, dims, NPY_INT, bearGraph->GetGraphPtr());
    return PyArray_Return( (PyArrayObject*)graph_py );
}



PyObject* BearingGraphBuilder_py::__getinitargs__()
{
    return PyTuple_Pack(3, db_pyobj_, PyInt_FromLong(maxNumPoints_), PyFloat_FromDouble(wThresh_));
}





void CalcSurfNorms_py( PyObject* neighs_py, PyObject* P_py, PyObject* sn_py, PyObject* valid_py )
{
    Graph neighs = numpy_to_eigen<int, Dynamic, 4>( neighs_py, "neighs", NPY_INT );
    int n = neighs.rows();
    Mat3<double>::type P = numpy_to_eigen<double, Dynamic, 3>( P_py, "P", NPY_DOUBLE, n );
    Mat3<double>::type sn = numpy_to_eigen<double, Dynamic, 3>( sn_py, "sn", NPY_DOUBLE, n );
    Vect<bool>::type valid = numpy_to_eigen<bool, Dynamic, 1>( valid_py, "valid", NPY_BOOL, n );

    CalcSurfNorms( neighs, P, sn, valid );
}



void BlurSurfNorms_py( PyObject* neighs_py, PyObject* P_py, PyObject* sn_py,
                      PyObject* valid_py, PyObject* sn_blurred_py, double sd )
{
    Graph neighs = numpy_to_eigen<int, Dynamic, 4>( neighs_py, "neighs", NPY_INT );
    int n = neighs.rows();
    Mat3<double>::type P = numpy_to_eigen<double, Dynamic, 3>( P_py, "P", NPY_DOUBLE, n );
    Mat3<double>::type sn = numpy_to_eigen<double, Dynamic, 3>( sn_py, "sn", NPY_DOUBLE, n );
    Vect<bool>::type valid = numpy_to_eigen<bool, Dynamic, 1>( valid_py, "valid", NPY_BOOL, n );
    Mat3<double>::type sn_blurred = numpy_to_eigen<double, Dynamic, 3>( sn_blurred_py, "sn_blurred", NPY_DOUBLE, n );

    BlurSurfNorms( neighs, P, sn, valid, sn_blurred, sd );
}


