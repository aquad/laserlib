/*! narf_py.cpp
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
 * \date       25-11-2010
*/

#include <Python.h>
#define PY_ARRAY_UNIQUE_SYMBOL _LaserPy_ARRAY_API
#define NO_IMPORT_ARRAY

#include "LaserPy/numpy_to_eigen.h"
#include <boost/python.hpp>
#include <omp.h>

#include "PCL/fpfh_interface.h"
#include "fpfh_py.h"
#include "Common/ProgressIndicator.h"
#include "LaserLibConfig.h"

using namespace boost::python;
using namespace Eigen;


void export_fpfh()
{
    def("FPFH", &fpfh_py);
    def("FPFH_Knn", &fpfh_knn_py);

    class_<FPFHObjectKnn_py, boost::shared_ptr<FPFHObjectKnn_py> >("FPFHObjectKnn", no_init)
            .def( "__init__", boost::python::make_constructor( &FPFHObjectKnn_py_constructor ) )
            .def("ClassifySet", &FPFHObjectKnn_py::ClassifySet);
}



void fpfh_py( PyObject* P_py, PyObject* sn_py, Selector& sel,
              PyObject* keys_py, PyObject* noBins_py, PyObject* hist_py )
{
    Mat3<double>::type P = numpy_to_eigen<double, Dynamic, 3>( P_py, "P", NPY_DOUBLE );
    Mat3<double>::type sn = numpy_to_eigen<double, Dynamic, 3>( sn_py, "sn", NPY_DOUBLE, P.rows() );
    Vect<int>::type keys = numpy_to_eigen<int, Dynamic, 1>( keys_py, "keys", NPY_INT );

    checkNumpyArray( noBins_py, "noBins", NPY_INT, 3 );
    Eigen::Vector3i noBins;
    int* noBinsPtr = (int*)PyArray_DATA( (PyArrayObject*)noBins_py );
    noBins(0) = noBinsPtr[0];
    noBins(1) = noBinsPtr[1];
    noBins(2) = noBinsPtr[2];

    MapMatXf hist = numpy_to_eigen<float, Dynamic, Dynamic>( hist_py, "hist", NPY_FLOAT, keys.rows(), noBins.sum() );

    fpfh( P, sn, sel, keys, noBins, hist );
}



void fpfh_knn_py( PyObject* test_py, PyObject* train_py, PyObject* matches_py, PyObject* values_py )
{
    int nTest = PyArray_DIM((PyArrayObject*)test_py,0);
    int nBins = PyArray_DIM((PyArrayObject*)test_py,1);
    MapMatXf test = numpy_to_eigen<float, Dynamic, Dynamic>( test_py, "test", NPY_FLOAT );
    MapMatXf train = numpy_to_eigen<float, Dynamic, Dynamic>( train_py, "train", NPY_FLOAT, -1, nBins );
    MapMatXi matches = numpy_to_eigen<int, Dynamic, Dynamic>( matches_py, "matches", NPY_INT, nTest, -1 );
    MapMatXf values = numpy_to_eigen<float, Dynamic, Dynamic>( values_py, "values", NPY_FLOAT, nTest, matches.cols() );

    fpfh_knn_classifier knn(test, train);
    knn.Classify( test.rows(), train.rows(), matches, values );
}






FPFHObjectKnn_py::FPFHObjectKnn_py( std::vector<MapMatXf>& trainObjData, std::vector<int>& nTrainPts,
                  bool showProgress, int nThreads )
    : FPFHObjectKnn( trainObjData, nTrainPts, showProgress ),
      nThreads_(nThreads)
{}

boost::shared_ptr<FPFHObjectKnn_py> FPFHObjectKnn_py_constructor( PyObject* dataset_py, bool showProgress, int nThreads )
{
    std::vector<MapMatXf> dataset;
    std::vector<int> nTrainPts;

    //go through each object in the dataset
    if( !PyList_Check(dataset_py) )
    {
        PyErr_SetString(PyExc_ValueError, "dataset must be a list" );
        throw_error_already_set();
    }
    Py_ssize_t length = PyList_Size(dataset_py);
    for( Py_ssize_t i=0 ; i<length ; i++ )
    {
        //retrieve fpfh data
        PyObject* objData = PyList_GetItem( dataset_py, i );
        MapMatXf fpfh_data = get_attribute_numpy_array<float, Dynamic, Dynamic>( objData, "fpfh", NPY_FLOAT );
        dataset.push_back(fpfh_data);
        nTrainPts.push_back( fpfh_data.rows() );
    }

    return boost::shared_ptr<FPFHObjectKnn_py>( new FPFHObjectKnn_py( dataset, nTrainPts, showProgress, nThreads ) );
}



void FPFHObjectKnn_py::ClassifySet( PyObject* testDataset_py, int k )
{
    std::vector<MapMatXf> testDataset;
    //retrieve fpfh data from each test object
    if( !PyList_Check(testDataset_py) )
    {
        PyErr_SetString(PyExc_ValueError, "dataset must be a list" );
        throw_error_already_set();
    }
    Py_ssize_t length = PyList_Size(testDataset_py);
    for( Py_ssize_t i=0 ; i<length ; i++ )
    {
        //retrieve fpfh data
        PyObject* objData = PyList_GetItem( testDataset_py, i );
        MapMatXf fpfh_data = get_attribute_numpy_array<float, Dynamic, Dynamic>( objData, "fpfh", NPY_FLOAT );
        testDataset.push_back(fpfh_data);
    }

    //progress indicator should be for the whole set
    bool showOverallProgress = GetShowProgress();
    if( showOverallProgress ){ SetShowProgress(false); }
    int nTestTotal = 0;
    for( int i=0 ; i<testDataset.size() ; i++ )
        nTestTotal += testDataset[i].rows();

    //do each test object in parallel- requires a copy of self for each thread
    FPFHObjectKnn selfCopy(*this);

    ProgressIndicator prog(nTestTotal, 5);
    if( showOverallProgress ){ prog.start(); }

    #ifdef LaserLib_USE_OPENMP
    if( nThreads_ > 0 )
        omp_set_num_threads(nThreads_);
    #endif

    int i;
    #pragma omp parallel for firstprivate(selfCopy) schedule(dynamic,1)
    for( i=0 ; i<testDataset.size() ; i++ )
    {
        MapMatXf& testObjData = testDataset[i];

        //create results arrays
        npy_intp dims[2] = {static_cast<npy_intp>(testObjData.rows()), k};
        PyObject* objMatches_py = PyArray_SimpleNew(2, dims, NPY_INT);
        PyObject* pointMatches_py = PyArray_SimpleNew(2, dims, NPY_INT);
        PyObject* distances_py =  PyArray_SimpleNew(2, dims, NPY_FLOAT);

        MapMatXi objMatches( (int*)PyArray_DATA( (PyArrayObject*)objMatches_py ),
                             testObjData.rows(), k );
        MapMatXi pointMatches( (int*)PyArray_DATA( (PyArrayObject*)pointMatches_py ),
                               testObjData.rows(), k );
        MapMatXf distances( (float*)PyArray_DATA( (PyArrayObject*)distances_py ),
                            testObjData.rows(), k );

        selfCopy.SetTestObject( testObjData );
        selfCopy.Classify( objMatches, pointMatches, distances );

        //add results arrays to object data
        PyObject* objData = PyList_GetItem( testDataset_py, (Py_ssize_t)i );
        PyObject_SetAttrString( objData, "match_objects", objMatches_py ); //could error check...
        PyObject_SetAttrString( objData, "match_points", pointMatches_py );
        PyObject_SetAttrString( objData, "match_dists", distances_py );
        //may have to decref...

        prog += testObjData.rows();
    }
    if( showOverallProgress )
    {
        prog.stop();
        SetShowProgress(true);
    }
}


