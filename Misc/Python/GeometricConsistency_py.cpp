/*! geocon_py.cpp
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
 * \date       15-07-2011
*/

#include <Python.h>
#define PY_ARRAY_UNIQUE_SYMBOL _LaserPy_ARRAY_API
#define NO_IMPORT_ARRAY
#include "LaserPy/numpy_to_eigen.h"

#include <boost/python.hpp>
#include <boost/graph/adjacency_list.hpp>
#include "GeometricConsistency_py.h"



void export_GeometricConsistency()
{
    using namespace boost::python;
    // -- old
    def("FindConsistentSet", &FindConsistentSet_py);
    def("FindComponents", &FindComponents_py);
    def("FindMaxClique", &FindMaxClique_py);
    // -------


    class_<CorrGraph, boost::shared_ptr<CorrGraph> >("CorrGraph", no_init);

    def("BuildCorrGraph", &BuildCorrGraph_py,
        "BuildCorrGraph(matches, testP, testVect, trainP, trainVect, distThresh, angleThresh, minDist)\n\n");

    def("BuildCorrGraphZAligned", &BuildCorrGraphZAligned_py,
        "BuildCorrGraphZAligned(matches, testP, testVect, trainP, trainVect, distThresh, angleThresh, minDist)\n\n");

    def("CorrGraphNumEdges", &CorrGraphNumEdges,
        "CorrGraphNumEdges(graph)\n\n"
        "Parameters\n"
        "----------\n"
        "graph : CorrGraph\n\n"
        "Returns\n"
        "-------\n"
        "nEdges : int\n");


    class_<bron_kerbosch_py>("bron_kerbosch", init<CorrGraph&>())
            .def("find_clique", &bron_kerbosch_py::find_clique_py);

    def("PairsOneToOne", &PairsOneToOne_py,
        "PairsOneToOne()\n\n"
        "Given a set of point pair matches, each with an associated feature distance, reduce them"
        "to one-to-one matches of the highest values.\n\n"
        "Parameters\n"
        "----------\n"
        "matches : ndarray (n,2) int32\n"
        "   each row is a matching pair (point ids)\n"
        "fDist : ndarray (n,) float32\n"
        "   feature distances of *matches*\n\n"
        "Returns\n"
        "-------\n"
        "matchesUnique : ndarray (k,) int32\n"
        "   references matches that are unique (one to one)\n");
}



using namespace boost;
using namespace Eigen;

PyObject* FindConsistentSet_py( PyObject* seeds_py,
                               PyObject* testIds_py, PyObject* testP_py, PyObject* testVect_py,
                               PyObject* trainIds_py, PyObject* trainP_py, PyObject* trainVect_py,
                               PyObject* rmse_py, float distThresh, float angleThresh, float rmseWeight)
{
    Vect<int>::type seeds = numpy_to_eigen<int, Dynamic, 1>( seeds_py, "seeds", NPY_INT );
    Vect<int>::type testIds = numpy_to_eigen<int, Dynamic, 1>( testIds_py, "testIds", NPY_INT );
    Mat3<float>::type testP = numpy_to_eigen<float, Dynamic, 3>( testP_py, "testP", NPY_FLOAT );
    Mat3<float>::type testVect = numpy_to_eigen<float, Dynamic, 3>( testVect_py, "testVect", NPY_FLOAT, testP.rows() );

    Vect<int>::type trainIds = numpy_to_eigen<int, Dynamic, 1>( trainIds_py, "trainIds", NPY_INT, testIds.rows() );
    Mat3<float>::type trainP = numpy_to_eigen<float, Dynamic, 3>( trainP_py, "trainP", NPY_FLOAT );
    Mat3<float>::type trainVect = numpy_to_eigen<float, Dynamic, 3>( trainVect_py, "trainVect", NPY_FLOAT, trainP.rows() );

    MapMatXf rmse = numpy_to_eigen<float, Dynamic, Dynamic>( rmse_py, "rmse", NPY_FLOAT, testP.rows(), trainP.rows() );

    float bestScore = 0.0;
    int bestSeed = 0;
    std::map<int,int> bestPairs;
    FindConsistentSet( seeds, testIds, testP, testVect, trainIds, trainP, trainVect, rmse,
                      bestScore, bestSeed, bestPairs, distThresh, angleThresh, rmseWeight );

    //convert pairs to python
    npy_intp dims[2] = {static_cast<npy_intp>(bestPairs.size()), 0};
    PyArrayObject* bestTestIds = (PyArrayObject*) PyArray_SimpleNew(1, dims, NPY_INT);
    int* bestTestIds_p = (int*)PyArray_DATA(bestTestIds);
    PyArrayObject* bestTrainIds = (PyArrayObject*) PyArray_SimpleNew(1, dims, NPY_INT);
    int* bestTrainIds_p = (int*)PyArray_DATA(bestTrainIds);

    std::map<int,int>::iterator it = bestPairs.begin();
    for( int i=0 ; i<bestPairs.size() ; i++ )
    {
        bestTestIds_p[i] = it->first;
        bestTrainIds_p[i] = it->second;
        it++;
    }
    PyObject* tup = PyTuple_Pack(4, PyFloat_FromDouble(bestScore), bestTestIds, bestTrainIds, PyInt_FromLong(bestSeed));
    Py_DECREF(bestTestIds);
    Py_DECREF(bestTrainIds);
    return tup;
}




void FindComponents_py( PyObject* matches_py,
                 PyObject* testP_py, PyObject* testVect_py,
                 PyObject* trainP_py, PyObject* trainVect_py,
                 float distThresh, float angleThresh, PyObject* components_py )
{
    pairMatrix matches = numpy_to_eigen<int, Dynamic, 2>( matches_py, "matches", NPY_INT );
    Mat3<float>::type testP = numpy_to_eigen<float, Dynamic, 3>( testP_py, "testP", NPY_FLOAT );
    Mat3<float>::type testVect = numpy_to_eigen<float, Dynamic, 3>( testVect_py, "testVect", NPY_FLOAT, testP.rows() );
    Mat3<float>::type trainP = numpy_to_eigen<float, Dynamic, 3>( trainP_py, "trainP", NPY_FLOAT );
    Mat3<float>::type trainVect = numpy_to_eigen<float, Dynamic, 3>( trainVect_py, "trainVect", NPY_FLOAT, trainP.rows() );
    Vect<int>::type components = numpy_to_eigen<int, Dynamic, 1>( components_py, "components", NPY_INT, matches.rows() );

    FindComponents( matches, testP, testVect, trainP, trainVect,
                   distThresh, angleThresh, components );
}




PyObject* FindMaxClique_py( PyObject* matches_py,
                 PyObject* testP_py, PyObject* testVect_py,
                 PyObject* trainP_py, PyObject* trainVect_py,
                 float distThresh, float minDist, float angleThresh, int minCliqueSize, int maxEdges )
{
    pairMatrix matches = numpy_to_eigen<int, Dynamic, 2>( matches_py, "matches", NPY_INT );
    Mat3<float>::type testP = numpy_to_eigen<float, Dynamic, 3>( testP_py, "testP", NPY_FLOAT );
    Mat3<float>::type testVect = numpy_to_eigen<float, Dynamic, 3>( testVect_py, "testVect", NPY_FLOAT, testP.rows() );
    Mat3<float>::type trainP = numpy_to_eigen<float, Dynamic, 3>( trainP_py, "trainP", NPY_FLOAT );
    Mat3<float>::type trainVect = numpy_to_eigen<float, Dynamic, 3>( trainVect_py, "trainVect", NPY_FLOAT, trainP.rows() );

    std::vector<int> maxClique;
    FindMaxClique( matches, testP, testVect, trainP, trainVect,
                   distThresh, minDist, angleThresh, minCliqueSize, maxEdges, maxClique );

    npy_intp dims[1] = {static_cast<npy_intp>(maxClique.size())};
    PyArrayObject *maxClique_py = (PyArrayObject*)PyArray_SimpleNew(1, dims, NPY_INT);
    memcpy( PyArray_DATA(maxClique_py), &(maxClique[0]), maxClique.size()*sizeof(int) );
    return PyArray_Return(maxClique_py);
}





boost::shared_ptr< CorrGraph > BuildCorrGraph_py( PyObject* matches_py,
                     PyObject* testP_py, PyObject* testVect_py,
                     PyObject* trainP_py, PyObject* trainVect_py,
                     float distThresh, float angleThresh, float minDist )
{
    pairMatrix matches = numpy_to_eigen<int, Dynamic, 2>( matches_py, "matches", NPY_INT );
    Mat3<float>::type testP = numpy_to_eigen<float, Dynamic, 3>( testP_py, "testP", NPY_FLOAT );
    Mat3<float>::type testVect = numpy_to_eigen<float, Dynamic, 3>( testVect_py, "testVect", NPY_FLOAT, testP.rows() );
    Mat3<float>::type trainP = numpy_to_eigen<float, Dynamic, 3>( trainP_py, "trainP", NPY_FLOAT );
    Mat3<float>::type trainVect = numpy_to_eigen<float, Dynamic, 3>( trainVect_py, "trainVect", NPY_FLOAT, trainP.rows() );

    boost::shared_ptr< CorrGraph > corrGraph = BuildCorrGraph( matches, testP, testVect, trainP, trainVect,
                     distThresh, angleThresh, minDist );
    return corrGraph;
}


boost::shared_ptr< CorrGraph > BuildCorrGraphZAligned_py( PyObject* matches_py,
                     PyObject* testP_py, PyObject* testVect_py,
                     PyObject* trainP_py, PyObject* trainVect_py,
                     float distThresh, float angleThresh, float minDist )
{
    pairMatrix matches = numpy_to_eigen<int, Dynamic, 2>( matches_py, "matches", NPY_INT );
    Mat3<float>::type testP = numpy_to_eigen<float, Dynamic, 3>( testP_py, "testP", NPY_FLOAT );
    Mat3<float>::type testVect = numpy_to_eigen<float, Dynamic, 3>( testVect_py, "testVect", NPY_FLOAT, testP.rows() );
    Mat3<float>::type trainP = numpy_to_eigen<float, Dynamic, 3>( trainP_py, "trainP", NPY_FLOAT );
    Mat3<float>::type trainVect = numpy_to_eigen<float, Dynamic, 3>( trainVect_py, "trainVect", NPY_FLOAT, trainP.rows() );

    boost::shared_ptr< CorrGraph > corrGraph = BuildCorrGraphZAligned( matches, testP, testVect, trainP, trainVect,
                     distThresh, angleThresh, minDist );
    return corrGraph;
}



int CorrGraphNumEdges( CorrGraph& g )
{
    return num_edges(g);
}



PyObject* PairsOneToOne_py( PyObject* matches_py, PyObject* fDist_py )
{
    pairMatrix matches = numpy_to_eigen<int, Dynamic, 2>( matches_py, "matches", NPY_INT );
    Vect<float>::type fDist = numpy_to_eigen<float, Dynamic, 1>( fDist_py, "fDist", NPY_FLOAT, matches.rows() );

    std::vector<int> matchesUnique;
    PairsOneToOne( matches, fDist, matchesUnique );

    npy_intp dims[1] = {static_cast<npy_intp>(matchesUnique.size())};
    PyArrayObject *matchesUnique_py = (PyArrayObject*)PyArray_SimpleNew(1, dims, NPY_INT);
    memcpy( PyArray_DATA(matchesUnique_py), &(matchesUnique[0]), matchesUnique.size() * sizeof(int) );
    return PyArray_Return(matchesUnique_py);
}

