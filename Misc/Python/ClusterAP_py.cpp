/*! ClusterAP_py.cpp
 *
 * Copyright (C) 2013 Alastair Quadros.
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
 * \date       10-07-2013
*/

#include <Python.h>
#define PY_ARRAY_UNIQUE_SYMBOL _LaserPy_ARRAY_API
#define NO_IMPORT_ARRAY
#include "LaserPy/numpy_to_eigen.h"
#include <boost/python.hpp>
#include "ClusterAP_py.h"


void export_ClusterAP()
{
    using namespace boost::python;

    class_< ClusterAP_py, boost::shared_ptr<ClusterAP_py>, boost::noncopyable >(
            "ClusterAP",
            "ClusterAP(sim[, showProgress=True, damping=0.9, minIter=15, convergeIter=20, maxIter=200, nThreads=-1])\n\n"
            "Affinity propogation clustering, requires only a similarity matrix. Call :meth:`run`, then :meth:`getAssignments`.\n"
            "Warning: all returned arrays have memory owned by this class (no memory copying).\n\n"
            "Parameters\n"
            "----------\n"
            "sim : ndarray (n,n) float32\n"
            "   similarity matrix\n"
            "showProgress : bool\n"
            "damping : float\n"
            "   0-1\n"
            "minIter : int\n"
            "   minimum number of iterations\n"
            "convergeIter : int\n"
            "   If the exemplars haven't changed after this many iterations, finish\n"
            "maxIter : int\n"
            "   maximum number of iterations\n"
            "nThreads : int\n"
            "   If >0, set number of OpenMP threads\n",
            no_init)
            //init<PyObject*, optional<bool, float, int, int, int, int> >())

            .def( "__init__", boost::python::make_constructor(
                        &ClusterAP_py_constructor, default_call_policies(),
                        (arg("sim"), arg("showProgress")=true, arg("damping")=0.9, arg("minIter")=15,
                         arg("convergeIter")=20, arg("maxIter")=200, arg("nThreads")=-1) ) )

            .def("run", &ClusterAP_py::run,
                 "run()\n\n"
                 "Returns\n"
                 "-------\n"
                 "iters : int\n\n")

            .def("runIters", &ClusterAP_py::runIters,
                 "runIters( [nIters=1] )\n\n"
                 "Returns\n"
                 "-------\n"
                 "changed : bool\n"
                 "  Whether exemplars changed\n")

            .def("getAssignments", &ClusterAP_py::getAssignments_py,
                 "getAssignments()\n\n"
                 "Returns\n"
                 "-------\n"
                 "assign : ndarray (n,) int32\n")

            .def("getAssignmentScores", &ClusterAP_py::getAssignmentScores_py,
                 "getAssignmentScores()\n\n"
                 "Similarity for each assignment.\n\n"
                 "Returns\n"
                 "-------\n"
                 "assignScores : ndarray (n,) float32\n")

            .def("getAvailabilities", &ClusterAP_py::getAvailabilities_py,
                 "getAvailabilities()\n\n"
                 "Returns\n"
                 "-------\n"
                 "avail : ndarray (n,n) float32\n")

            .def("getResponsibilities", &ClusterAP_py::getResponsibilities_py,
                 "getResponsibilities()\n\n"
                 "Returns\n"
                 "-------\n"
                 "respon : ndarray (n,n) float32\n");

}

using namespace Eigen;


ClusterAP_py::ClusterAP_py( MapMatXf& sim, bool showProgress, float damping, int minIter, int convergeIter, int maxIter, int nThreads )
    : ClusterAP( sim, showProgress, damping, minIter, convergeIter, maxIter, nThreads )
{}


boost::shared_ptr<ClusterAP_py> ClusterAP_py_constructor( PyObject* sim_py, bool showProgress, float damping,
                                                          int minIter, int convergeIter, int maxIter, int nThreads )
{
    int n = PyArray_DIM((PyArrayObject*)sim_py, 0);
    MapMatXf sim = numpy_to_eigen<float, Dynamic, Dynamic>( sim_py, "sim", NPY_FLOAT, n, n ); // must be square
    return boost::shared_ptr<ClusterAP_py>( new ClusterAP_py( sim, showProgress, damping, minIter, convergeIter, maxIter, nThreads ) );
}



PyObject* ClusterAP_py::getAssignments_py()
{
    MapVecXi assign = getAssignments();
    npy_intp dims[1] = {assign.rows()};
    PyObject *assign_py = PyArray_SimpleNewFromData(1, dims, NPY_INT, assign.data());
    return PyArray_Return( (PyArrayObject*)assign_py );
}


PyObject* ClusterAP_py::getAssignmentScores_py()
{
    MapVecXf assignScores = getAssignmentScores();
    npy_intp dims[1] = {assignScores.rows()};
    PyObject *assignScores_py = PyArray_SimpleNewFromData(1, dims, NPY_FLOAT, assignScores.data());
    return PyArray_Return( (PyArrayObject*)assignScores_py );
}


PyObject* ClusterAP_py::getAvailabilities_py()
{
    npy_intp dims[2] = {size(), size()};
    PyObject *avail_py = PyArray_SimpleNewFromData(2, dims, NPY_FLOAT, avail_.data());
    return PyArray_Return( (PyArrayObject*)avail_py );
}


PyObject* ClusterAP_py::getResponsibilities_py()
{
    npy_intp dims[2] = {size(), size()};
    PyObject *respon_py = PyArray_SimpleNewFromData(2, dims, NPY_FLOAT, respon_.data());
    return PyArray_Return( (PyArrayObject*)respon_py );
}

