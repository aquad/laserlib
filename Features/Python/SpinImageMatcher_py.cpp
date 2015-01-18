/*! SpinImageMatcher_py.cpp
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
 * \date       12-01-2013
*/

#include <Python.h>
#define PY_ARRAY_UNIQUE_SYMBOL _LaserPy_ARRAY_API
#define NO_IMPORT_ARRAY
#include "LaserPy/numpy_to_eigen.h"

#include <boost/python.hpp>
#include "SpinImageMatcher_py.h"
#include "Common/ArrayTypes.h"

using namespace boost::python;
using namespace Eigen;


void export_SpinImageMatcher()
{
    def("MatchSpinSets", &MatchSpinSets_py,
        "MatchSpinSets(test, training, values, lamb, metric)\n\n");

    def("SpinCorrelation", &SpinCorrelation_py,
        "SpinCorrelation(P, Q)\n\n");

    def("SpinCorrAtanh", &SpinCorrAtanh_py,
        "SpinCorrelation(P, Q)\n\n");

    def("SpinSimilarity", &SpinSimilarity_py,
        "SpinSimilarity(P, Q, lamb)\n\n");

    def("SpinKnn", &SpinKnn_py);
}



//matches only 1 test spin images to a bank of training spin images.
//can only do one at a time, as 'values' is already quite large.
void MatchSpinSets_py( PyObject* test_py, PyObject* train_py, PyObject* values_py, float lamb, SpinMetric metric )
{
    MapVecXf test = numpy_to_eigen<float, Dynamic, 1>( test_py, "test", NPY_FLOAT );
    int nCells = test.rows();
    MapMatXf train = numpy_to_eigen<float, Dynamic, Dynamic>( train_py, "train", NPY_FLOAT, -1, nCells );
    int nImages = train.rows();
    MapVecXf values = numpy_to_eigen<float, Dynamic, 1>( values_py, "values", NPY_FLOAT, nImages );

    for(int i=0 ; i<nImages ; i++)
    {
        values[i] = SpinDistance( test.data(), train.row(i).data(), nCells, lamb, metric );
    }
}



float SpinCorrelation_py( PyObject* Ppy, PyObject* Qpy)
{
    int nCells = PyArray_DIM((PyArrayObject*)Ppy,0);
    float* P_p = numpy_to_ptr<float>( Ppy, "P", NPY_FLOAT );
    float* Q_p = numpy_to_ptr<float>( Qpy, "Q", NPY_FLOAT, nCells );
    return SpinCorrelation( P_p, Q_p, nCells );
}

float SpinCorrAtanh_py( PyObject* Ppy, PyObject* Qpy)
{
    int nCells = PyArray_DIM((PyArrayObject*)Ppy,0);
    float* P_p = numpy_to_ptr<float>( Ppy, "P", NPY_FLOAT );
    float* Q_p = numpy_to_ptr<float>( Qpy, "Q", NPY_FLOAT, nCells );
    return SpinCorrAtanh( P_p, Q_p, nCells );
}

float SpinSimilarity_py( PyObject* Ppy, PyObject* Qpy, float lamb)
{
    int nCells = PyArray_DIM((PyArrayObject*)Ppy,0);
    float* P_p = numpy_to_ptr<float>( Ppy, "P", NPY_FLOAT );
    float* Q_p = numpy_to_ptr<float>( Qpy, "Q", NPY_FLOAT, nCells );
    return SpinSimilarity( P_p, Q_p, nCells, lamb );
}



//With multiple test spin images, and a set of training spin images, do a comparison between each, keeping the top knn values.
//'matches' and 'values' has the same number of rows as 'test' and knn columns.
//'test' and 'training' are (N,64) matrices.
void SpinKnn_py( PyObject* test_py, PyObject* training_py,
                 PyObject* matches_py, PyObject* values_py, float lamb, SpinMetric metric )
{
    MapMatXf test = numpy_to_eigen<float, Dynamic, Dynamic>( test_py, "test", NPY_FLOAT );
    int n = test.rows();
    int nCells = test.cols();
    MapMatXf training = numpy_to_eigen<float, Dynamic, Dynamic>( training_py, "training", NPY_FLOAT, -1, nCells );
    MapMatXi matches = numpy_to_eigen<int, Dynamic, Dynamic>( matches_py, "matches", NPY_INT, n );
    int k = matches.cols();
    MapMatXf values = numpy_to_eigen<float, Dynamic, Dynamic>( values_py, "values", NPY_FLOAT, n, k );

    SpinSetKnn classifier(test, training, lamb, metric);
    classifier.Classify( test.rows(), training.rows(), matches, values );
}

