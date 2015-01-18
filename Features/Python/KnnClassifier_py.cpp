/*! KnnClassifier_py.cpp
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
 * \date       15-08-2012
*/

#include <Python.h>
#define PY_ARRAY_UNIQUE_SYMBOL _LaserPy_ARRAY_API
#define NO_IMPORT_ARRAY
#include "LaserPy/numpy_to_eigen.h"

#include <boost/python.hpp>
#include "Common/ProgressIndicator.h"
#include "Common/ArrayTypes.h"
#include "KnnClassifier_py.h"

using namespace Eigen;
using namespace boost::python;


PyObject* ObjectKnnClassifier_py::Classify_py( int k )
{
    //create results arrays
    int nPoints = GetNTest();
    npy_intp dims[2] = {nPoints, k};
    PyArrayObject* objMatches_py = (PyArrayObject*) PyArray_SimpleNew(2, dims, NPY_INT);
    PyArrayObject* pointMatches_py = (PyArrayObject*) PyArray_SimpleNew(2, dims, NPY_INT);
    PyArrayObject* distances_py = (PyArrayObject*) PyArray_SimpleNew(2, dims, NPY_FLOAT);

    MapMatXi objMatches( (int*)PyArray_DATA(objMatches_py), nPoints, k );
    MapMatXi pointMatches( (int*)PyArray_DATA(pointMatches_py), nPoints, k );
    MapMatXf distances( (float*)PyArray_DATA(distances_py), nPoints, k );

    Classify( objMatches, pointMatches, distances );

    PyObject* tup = PyTuple_Pack(3, objMatches_py, pointMatches_py, distances_py);
    return tup;
}


PyObject* ObjectKnnClassifier_py::ClassifyObj( PyObject* testObjData_py, int k )
{
    SetTestObject_py( testObjData_py );
    return Classify_py(k);
}


void ObjectKnnClassifier_py::ClassifySet( PyObject* testObjDataset_py, int k )
{
    if( !PyList_Check(testObjDataset_py) )
    {
        PyErr_SetString(PyExc_ValueError, "test dataset must be a list" );
        throw_error_already_set();
    }
    int nTestObjects = (int)PyList_Size(testObjDataset_py);

    // progress indicator needs total number of features being compared
    int nTestTotal = 0;
    if( GetShowProgress() )
    {
        for( int i=0 ; i<nTestObjects ; i++ )
        {
            PyObject* testObj = PyList_GetItem( testObjDataset_py, i );
            SetTestObject_py( testObj );
            nTestTotal += GetNTest();
        }
    }
    ProgressIndicator prog(nTestTotal, 5);
    if( GetShowProgress() ){ prog.start(); }

    //Cannot use openmp due to python calls. Cannot convert python stuff to
    //pure c++ without some crazy template crap. Ohwell, should do parallel
    //stuff on the python side anyway.
    bool showOverallProgress = GetShowProgress();
    SetShowProgress(false); //disable per-object progress indication
    for( int i=0 ; i<nTestObjects ; i++ )
    {
        PyObject* testObj = PyList_GetItem( testObjDataset_py, (Py_ssize_t)i );
        SetTestObject_py( testObj );

        //create results arrays
        int nPoints = GetNTest();
        npy_intp dims[2] = {nPoints, k};
        PyObject* objMatches_py = PyArray_SimpleNew(2, dims, NPY_INT);
        PyObject* pointMatches_py = PyArray_SimpleNew(2, dims, NPY_INT);
        PyObject* distances_py = PyArray_SimpleNew(2, dims, NPY_FLOAT);

        MapMatXi objMatches( (int*)PyArray_DATA( (PyArrayObject*)objMatches_py ),
                             nPoints, k );
        MapMatXi pointMatches( (int*)PyArray_DATA( (PyArrayObject*)pointMatches_py ),
                               nPoints, k );
        MapMatXf distances( (float*)PyArray_DATA( (PyArrayObject*)distances_py ),
                            nPoints, k );

        Classify( objMatches, pointMatches, distances );

        PyObject* time_py = PyLong_FromLong( computeTime.total_microseconds() );

        //add results arrays to object data
        PyObject_SetAttrString( testObj, "match_objects", objMatches_py );
        PyObject_SetAttrString( testObj, "match_points", pointMatches_py );
        PyObject_SetAttrString( testObj, "match_dists", distances_py );
        PyObject_SetAttrString( testObj, "match_time", time_py );

        if( showOverallProgress ){ prog += nPoints; }
    }
    if( showOverallProgress )
    {
        prog.stop();
        SetShowProgress(true);
    }
}


long ObjectKnnClassifier_py::GetComputeTime()
{
    return computeTime.total_microseconds();
}



//-----------------------

MatchData::MatchData( MapMatXi& object, MapMatXi& point, MapMatXf& dist )
    : object_(object), point_(point), dist_(dist) {}


MatchData::MatchData( PyObject* object_py, PyObject* point_py, PyObject* dist_py )
    : object_( numpy_to_eigen<int, Eigen::Dynamic, Eigen::Dynamic>(object_py, "object", NPY_INT ) ),
      point_( numpy_to_eigen<int, Eigen::Dynamic, Eigen::Dynamic>(point_py, "point", NPY_INT) ),
      dist_( numpy_to_eigen<float, Eigen::Dynamic, Eigen::Dynamic>(dist_py, "dist", NPY_FLOAT) )
{}


