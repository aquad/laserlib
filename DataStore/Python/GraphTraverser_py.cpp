/*! GraphTraverser_py.cpp
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
 * \date       10-11-2010
*/

#include <Python.h>
#define PY_ARRAY_UNIQUE_SYMBOL _LaserPy_ARRAY_API
#define NO_IMPORT_ARRAY
#include "LaserPy/numpy_to_eigen.h"
#include <boost/python.hpp>
#include "GraphTraverser_py.h"



void export_GraphTraverser()
{
    using namespace boost::python;

    class_<GraphSphereSelector_py, boost::shared_ptr<GraphSphereSelector_py>,
        bases<Selector> >( "GraphSphereSelector",
            "GraphSphereSelector(graph, P, rad)\n\n"
            "Select 3D spherical regions using the bearing graph\n\n"
            "Parameters\n"
            "----------\n"
            "graph : (n,4) ndarray, int32\n"
            "P : (n,3) ndarray, float64\n"
            "    3D point cloud\n"
            "rad : double",
            no_init)

            .def( "__init__", boost::python::make_constructor( &GraphSphereSelector_py_constructor ) )

            .def("SelectRegion", &GraphSphereSelector_py::SelectRegion_py,
                "SelectRegion(i)\n\n"
                "Select spherical region about existing point.\n\n"
                "Parameters\n"
                "----------\n"
                "i : integer\n"
                "    Point id at the centre of the sphere to select")

            .def("SetRadius", &GraphSphereSelector_py::SetRadius,
                "SetRadius(r)");


    class_<PreFourNeighSelector_py, boost::shared_ptr<PreFourNeighSelector_py>,
        bases<Selector> >( "PreFourNeighSelector",
            "PreFourNeighSelector(neighs)\n\n"
            "Uses the output of :func:`Get4Neighs` as a :class:`Selector`, allowing "
            "functions that take in selectors to use those results.\n\n"
            "Parameters\n"
            "----------\n"
            "neighs : (n,4) ndarray, int32\n"
            "   left/right/up/down neighbours, from :func:`Get4Neighs`",
            no_init)

            .def( "__init__", boost::python::make_constructor( &PreFourNeighSelector_py_constructor ) )

            .def("SelectRegion", &PreFourNeighSelector_py::SelectRegion_py,
                "SelectRegion(i)\n\n"
                "Select spherical region about existing point.\n\n"
                "Parameters\n"
                "----------\n"
                "i : integer\n"
                "    Point id at the centre of the sphere to select");


    def("Get4Neighs", &Get4Neighs_py,
            "Get4Neighs(neighs, graph, P, rad)\n\n"
            "Given a bearing graph (specifying direct, immediate connectivity "
            "of points), select neighbours left/right/up/down from each point, a "
            "given distance away (or slightly greater). The graph is traversed "
            "to find these neighbours.\n\n"
            "Parameters\n"
            "----------\n"
            "neighs : (n,4) ndarray, int32\n"
            "   (output) left/right/up/down neighbours of specified distance away.\n"
            "graph : (n,4) ndarray, int32\n"
            "   Bearing graph (directly connected, immediate neighbours\n"
            "P : (n,3) ndarray, float64\n"
            "   3D points\n"
            "rad : double\n"
            "   3D distance (m) that each left/right/up/down neighbour should -at least- be");

    def("Get4NeighsValid", &Get4NeighsValid_py,
            "Get4NeighsValid(neighs, graph, P, valid, rad)\n\n"
            "As in :func:`Get4Neighs`, but don't pick invalid points as neighbours.\n\n"
            "Parameters\n"
            "----------\n"
            "neighs : (n,4) ndarray, int32\n"
            "   (output) left/right/up/down neighbours of specified distance away.\n"
            "graph : (n,4) ndarray, int32\n"
            "   Bearing graph (directly connected, immediate neighbours\n"
            "P : (n,3) ndarray, float64\n"
            "   3D points\n"
            "valid : (n,) ndarray, bool\n"
            "rad : double\n"
            "   3D distance (m) that each left/right/up/down neighbour should -at least- be");

    def("Get4NeighsClosest", &Get4NeighsClosest_py);

    def("ConvexSegment", &ConvexSegment_py);

}



using namespace Eigen;


boost::shared_ptr<GraphSphereSelector_py> GraphSphereSelector_py_constructor( PyObject* graph_py, PyObject* P_py, double rad )
{
    Graph graph = numpy_to_eigen<int, Dynamic, 4>( graph_py, "graph", NPY_INT );
    Mat3<double>::type P = numpy_to_eigen<double, Dynamic, 3>( P_py, "P", NPY_DOUBLE, graph.rows() );
    return boost::shared_ptr<GraphSphereSelector_py>( new GraphSphereSelector_py(graph, P, rad) );
}



boost::shared_ptr<PreFourNeighSelector_py> PreFourNeighSelector_py_constructor( PyObject* neighs_all_py )
{
    Graph neighs_all = numpy_to_eigen<int, Dynamic, 4>( neighs_all_py, "neighs_all", NPY_INT );
    return boost::shared_ptr<PreFourNeighSelector_py>( new PreFourNeighSelector_py(neighs_all) );
}



void Get4Neighs_py( PyObject* neighs_py, PyObject* graph_py, PyObject* P_py, double rad )
{
    Graph neighs = numpy_to_eigen<int, Dynamic, 4>( neighs_py, "neighs", NPY_INT );
    int n = neighs.rows();
    Graph graph = numpy_to_eigen<int, Dynamic, 4>( graph_py, "graph", NPY_INT, n );
    Mat3<double>::type P = numpy_to_eigen<double, Dynamic, 3>( P_py, "P", NPY_DOUBLE, n );

    Get4Neighs( neighs, graph, P, rad );
}


void Get4NeighsClosest_py( PyObject* neighs_py, PyObject* graph_py )
{
    Graph neighs = numpy_to_eigen<int, Dynamic, 4>( neighs_py, "neighs", NPY_INT );
    Graph graph = numpy_to_eigen<int, Dynamic, 4>( graph_py, "graph", NPY_INT, neighs.rows() );
    Get4NeighsClosest( neighs, graph );
}



void Get4NeighsValid_py( PyObject* neighs_py, PyObject* graph_py, PyObject* P_py, PyObject* valid_py, double rad )
{
    Graph neighs = numpy_to_eigen<int, Dynamic, 4>( neighs_py, "neighs", NPY_INT );
    int n = neighs.rows();
    Graph graph = numpy_to_eigen<int, Dynamic, 4>( graph_py, "graph", NPY_INT, n );
    Mat3<double>::type P = numpy_to_eigen<double, Dynamic, 3>( P_py, "P", NPY_DOUBLE, n );
    Vect<bool>::type valid = numpy_to_eigen<bool, Dynamic, 1>( valid_py, std::string("valid"), NPY_BOOL, n );

    Get4NeighsValid( neighs, graph, P, valid.data(), rad );
}



void ConvexSegment_py( PyObject* segs_py, PyObject* convex_py,
                       PyObject* graph_py, PyObject* surfNorms_py, double eta3 )
{
    Vect<int>::type segs = numpy_to_eigen<int, Dynamic, 1>( segs_py, "segs", NPY_INT );
    int n = segs.rows();
    Graph convex = numpy_to_eigen<int, Dynamic, 4>( convex_py, "convex", NPY_INT, n );
    Graph graph = numpy_to_eigen<int, Dynamic, 4>( graph_py, "graph", NPY_INT, n );
    Mat3<double>::type surfNorms = numpy_to_eigen<double, Dynamic, 3>( surfNorms_py, "surfNorms", NPY_DOUBLE, n );

    ConvexSegment( segs, convex, graph, surfNorms, eta3 );
}
