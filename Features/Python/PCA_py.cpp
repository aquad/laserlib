/*! PCA_py.cpp
 *
 * Copyright (C) 2010 Alastair Quadros.
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
 * \date       24-11-2010
*/

#include <Python.h>
#define PY_ARRAY_UNIQUE_SYMBOL _LaserPy_ARRAY_API
#define NO_IMPORT_ARRAY
#include "LaserPy/numpy_to_eigen.h"
#include <iostream>
#include <omp.h>
#include <boost/python.hpp>
#include "PCA_py.h"
#include "Features/PCA.h"
#include "DataStore/Selector.h"
#include "DataStore/Python/VeloImageSelect_py.h"
#include "LaserLibConfig.h" //cmake option for openmp



void export_PCA()
{
    using namespace boost::python;
    class_<PCA_py, boost::shared_ptr<PCA_py> >("PCA",
                                               "PCA(P[, nThreads=1])\n\n",
                                               init< PyObject*, optional<int> >())

            .def("ComputeAll", &PCA_py::ComputeAll,
                 "ComputeAll(sel, keys, evals, evects, meanP)\n\n"
                 "Compute PCA at all keypoints.\n\n"
                 "Parameters\n"
                 "----------\n"
                 "sel : :class:`Selector`\n"
                 "  Selector (specifies region/radius to compute PCA on)\n"
                 "keys : (n,) ndarray, int32\n"
                 "  Keypoints, referencing points provided to `sel`\n"
                 "evals : (n,3) ndarray, float32\n"
                 "  (output) eigenvalues at each keypoint\n"
                 "evects : (n,3,3) ndarray, float32\n"
                 "  (output) eigenvectors\n"
                 "meanP : (n,3) ndarray, float64\n"
                 "  (output) mean 3D point of each region\n")

            .def("ComputeAllVariableSize", &PCA_py::ComputeAllVariableSize,
                 "ComputeAllVariableSize(sel, keys, rad, evals, evects, meanP)\n\n"
                 "Compute PCA at all keypoints, with a specified radius for each one.\n\n"
                 "Parameters\n"
                 "----------\n"
                 "sel : :class:`ImageSphereSelector`\n"
                 "keys : (n,) ndarray, int32\n"
                 "  Keypoints, referencing points provided to `sel`\n"
                 "rad : (n,) ndarray, float32\n"
                 "  Radius of each region\n"
                 "evals : (n,3) ndarray, float32\n"
                 "  (output) eigenvalues at each keypoint\n"
                 "evects : (n,3,3) ndarray, float32\n"
                 "  (output) eigenvectors\n"
                 "meanP : (n,3) ndarray, float64\n"
                 "  (output) mean 3D point of each region\n");


    def("minRadiusSelection", &minRadiusSelection_py,
        "minRadiusSelection(graph, P, rad, valid)\n\n"
        "Determine the minimum size sphere about each point that PCA can be computed on."
        "The selection must include a neighbouring horizontal and vertical point"
        "(if they exist- valid to compute PCA on a pole with no horizontal neighbours)."
        "This uses the bearing graph from :class:`BearingGraphBuilder`.\n\n"
        "Parameters\n"
        "----------\n"
        "graph : (n,4) ndarray, int32\n"
        "   Graph from :class:`BearingGraphBuilder`\n"
        "P : (n,3) ndarray, float64\n"
        "   3D points\n"
        "rad : (n,) ndarray, float32\n"
        "   (output) Radii\n"
        "valid : (n,) ndarray, bool\n"
        "   (output) Some regions are not valid (insufficient connectivity to neighbours)\n");


    def("surfNormPCA", &surfNormPCA<float>, surfNormPCA_ol_f(
        "surfNormPCA(sel, P, sn[, ids])\n\n"
        "Compute surface normals with PCA\n\n"
        "Parameters\n"
        "----------\n"
        "sel : :class:`Selector`\n"
        "P : ndarray float32/float64 (n,3)\n"
        "sn : ndarray float32 (n,3)\n"
        "   (Output) surface normals. If *ids* is specified, is the length of *ids* rather than *P*\n"
        "ids : ndarray int32 (k,)\n"
        "   Optional, specify a subset of points to compute.\n"
        "surfThresh : float\n"
        "   Optional, filters out bad results (long thin regions with no strong surface normals). "
        "Lower is stricter. Invalid surface normals are set to (0,0,0).\n"));

    def("surfNormPCA", &surfNormPCA<double>, surfNormPCA_ol_d());
}


using namespace Eigen;

PCA_py::PCA_py( PyObject* P_py, int nThreads )
    :   P_( numpy_to_eigen<double, Dynamic, 3>( P_py, "P", NPY_DOUBLE ) ),
        nThreads_(nThreads)
{}


void PCA_py::ComputeAll( Selector& sel, PyObject* keys_py, PyObject* evals_py,
        PyObject* evects_py, PyObject* meanP_py )
{
    Vect<int>::type keys = numpy_to_eigen<int, Dynamic, 1>( keys_py, "keys", NPY_INT );
    int nKeys = keys.rows();
    Mat3<float>::type evals = numpy_to_eigen<float, Dynamic, 3>( evals_py, "evals", NPY_FLOAT, nKeys );
    MapMatXf evects = numpy_to_eigen<float, Dynamic, Dynamic>( evects_py, "evects", NPY_FLOAT, nKeys, 9 );
    Mat3<double>::type meanP = numpy_to_eigen<double, Dynamic, 3>( meanP_py, "meanP", NPY_DOUBLE, nKeys );

    Py_BEGIN_ALLOW_THREADS;
    PCA pca(P_);

    bool useOpenMP = false;
    #ifdef LaserLib_USE_OPENMP
    if( nThreads_ > 0 )
        omp_set_num_threads(nThreads_);
    if( nThreads_ != 1 )
        useOpenMP = true;
    #endif

    if( useOpenMP )
    {
        int i;
        #pragma omp parallel
        {
            //copy the selector (one for each thread)
            //pca class can be shared
            boost::shared_ptr<Selector> thisSel( sel.clone() );
            #pragma omp for
            for(i=0 ; i<keys.rows() ; i++)
            {
                Map<Vector3f> evalBlock( evals.row(i).data() );
                MapMat3f evectBlock( evects.row(i).data() );
                Map<Vector3d> meanPBlock( meanP.row(i).data() );

                std::vector<int>& neigh = thisSel->SelectRegion( keys(i) );
                pca.compute( neigh, evalBlock, evectBlock, meanPBlock );
            }
        }
    }
    else
    {
        //single-thread (simpler)
        for(int i=0 ; i<keys.rows() ; i++)
        {
            Map<Vector3f> evalBlock( evals.row(i).data() );
            MapMat3f evectBlock( evects.row(i).data() );
            Map<Vector3d> meanPBlock( meanP.row(i).data() );

            std::vector<int>& neigh = sel.SelectRegion( keys(i) );
            pca.compute( neigh, evalBlock, evectBlock, meanPBlock );
        }
    }

    Py_END_ALLOW_THREADS;
}




void minRadiusSelection_py( PyObject* graph_py, PyObject* P_py, PyObject* rad_py, PyObject* valid_py )
{
    Graph graph = numpy_to_eigen<int, Dynamic, 4>( graph_py, "graph", NPY_INT );
    int n = graph.rows();
    Mat3<double>::type P = numpy_to_eigen<double, Dynamic, 3>( P_py, "P", NPY_DOUBLE, n );
    Vect<float>::type rad = numpy_to_eigen<float, Dynamic, 1>( rad_py, "rads", NPY_FLOAT, n );
    Vect<bool>::type valid = numpy_to_eigen<bool, Dynamic, 1>( valid_py, "valid", NPY_BOOL, n );
    minRadiusSelection( graph, P, rad, valid );
}




void PCA_py::ComputeAllVariableSize( ImageSphereSelector_py& sel, PyObject* keys_py, PyObject* rad_py,
                                    PyObject* evals_py, PyObject* evects_py, PyObject* meanP_py )
{
    Vect<int>::type keys = numpy_to_eigen<int, Dynamic, 1>( keys_py, "keys", NPY_INT );
    int nKeys = keys.rows();
    Vect<float>::type rad = numpy_to_eigen<float, Dynamic, 1>( rad_py, "rad", NPY_FLOAT, nKeys );
    checkNumpyArray( evals_py, "evals", NPY_FLOAT, nKeys, 3 );
    checkNumpyArray( evects_py, "evects", NPY_FLOAT, nKeys, 3, 3 );
    checkNumpyArray( meanP_py, "meanP", NPY_DOUBLE, nKeys, 3 );

    PCA pca(P_);
    ImageSphereSelector_py selLocal(sel);
    int i;
    #pragma omp parallel for firstprivate(pca, selLocal)
    for(i=0 ; i<nKeys ; i++)
    {
        sel.SetRadius( rad(i) );
        Map<Vector3f> evalBlock( (float*)PyArray_GETPTR2((PyArrayObject*)evals_py, i, 0) );
        MapMat3f evectBlock( (float*)PyArray_GETPTR3((PyArrayObject*)evects_py, i, 0, 0) );
        Map<Vector3d> meanPBlock( (double*)PyArray_GETPTR2((PyArrayObject*)meanP_py, i, 0) );
        std::vector<int>& neigh = selLocal.SelectRegion( keys(i) );
        pca.compute( neigh, evalBlock, evectBlock, meanPBlock );
    }
}

