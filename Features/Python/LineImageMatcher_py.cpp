/*! LineImageMatcher_py.cpp
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
#include "LaserPy/numpy_to_eigen.h"
#include "LaserPy/numpy_to_std.h"

#include <iostream>
#include <boost/python.hpp>
#include <omp.h>

#include "LineImage_py.h"
#include "LineImageMatcher_py.h"
#include "Common/ProgressIndicator.h"
#include "LaserLibConfig.h"

using namespace Eigen;
using namespace boost::python;


void export_LineImageMatcher()
{
    def("match_line_images", &match_line_images,
        "match_line_images(params, metric, values1, status1, valuesSet, statusSet, rmse, known)\n\n"
        "Match one point's line image (values1, status1) to a set of others.\n\n"
        ":param params: radial_feature.line_image.lineImageParams() class instance\n"
        ":param metric: comparison function number, 1-5\n"
        ":param values1: line image values, numpy array, shape=(nLines,), dtype=float32\n"
        ":param status1: line image status, numpy array, shape=(nLines,), dtype=uint8\n"
        ":param valuesSet: set of line image values, numpy array, shape=(n,nLines), dtype=float32\n"
        ":param statusSet: set of line image status, numpy array, shape=(n,nLines), dtype=uint8\n"
        ":param rmse: comparison output, numpy array, shape=(n,), dtype=float32\n"
        ":param known: comparison output, numpy array, shape=(n,), dtype=float32\n");

    def("match_line_images_keys", &match_line_images_keys,
        "match_line_images_keys(params, metric, values1, status1, valuesSet, statusSet, keys, rmse, known)\n\n"
        "As :func:`match_line_images`, but only match to the features from the set indexed by :obj:`keys`."
        "(rmse, known) are still the full size of valuesSet.\n\n"
        ":param keys: references valuesSet, statusSet, array (m,) int32");

    def("match_line_image_sets", &match_line_image_sets,
        "match_line_image_sets(params, metric, values1, status1, values2, status2, rmse, known)\n\n"
        "As :func:`match_line_images`, but between two sets of line images, such that values1[0] is compared to values2[0] etc.\n\n"
        ":param values1: (n,nLines)\n"
        ":param status1: (n,nLines)\n"
        ":param values2: (n,nLines)\n"
        ":param status2: (n,nLines)\n");

    def("match_line_images_all", &match_line_images_all, match_line_images_all_overloads(
        "match_line_images_all(params, metric, values1, status1, values2, status2, rmse, known[, nThreads=-1])\n\n"
        "As :func:`match_line_image_sets`, but every value is compared to every other.\n\n"
        ":param values1: (n,nLines)\n"
        ":param status1: (n,nLines)\n"
        ":param values2: (m,nLines)\n"
        ":param status2: (m,nLines)\n"
        ":param rmse: (n,m)\n"
        ":param known: (n,m)\n"
        ":param nThreads: number of threads to use, if compiled with openmp (-1: use as many as possible)\n") );
}



void match_line_images( PyObject* params_py, int metric, PyObject* values1_py, PyObject* status1_py,
        PyObject* valuesSet_py, PyObject* statusSet_py,
        PyObject* rmse_py, PyObject* known_py )
{
    LineImageParams_py params_py_derived(params_py);
    LineImageParams& params = *dynamic_cast<LineImageParams*>(&params_py_derived);

    float* values1 = numpy_to_ptr<float>( values1_py, "values1", NPY_FLOAT, params.nLines );
    unsigned char* status1 = numpy_to_ptr<unsigned char>( status1_py, "status1", NPY_UBYTE, params.nLines );

    float* valuesSet = numpy_to_ptr<float>( valuesSet_py, "valuesSet", NPY_FLOAT, -1, params.nLines );
    int nItems = PyArray_DIM((PyArrayObject*)valuesSet_py, 0);
    unsigned char* statusSet = numpy_to_ptr<unsigned char>( statusSet_py, "statusSet", NPY_UBYTE, nItems, params.nLines );

    float* rmse = numpy_to_ptr<float>( rmse_py, "rmse", NPY_FLOAT, nItems );
    float* known = numpy_to_ptr<float>( known_py, "known", NPY_FLOAT, nItems );

    boost::shared_ptr<LineImageMatcher> matcher = MakeLIMatcher( params, metric );

    matcher->match_rmse_one_many( values1, status1, valuesSet, statusSet, nItems, rmse, known );
}




void match_line_images_keys( PyObject* params_py, int metric, PyObject* values1_py, PyObject* status1_py,
        PyObject* valuesSet_py, PyObject* statusSet_py, PyObject* keys_py,
        PyObject* rmse_py, PyObject* known_py )
{
    LineImageParams_py params_py_derived(params_py);
    LineImageParams& params = *dynamic_cast<LineImageParams*>(&params_py_derived);

    float* values1 = numpy_to_ptr<float>( values1_py, "values1", NPY_FLOAT, params.nLines );
    unsigned char* status1 = numpy_to_ptr<unsigned char>( status1_py, "status1", NPY_UBYTE, params.nLines );

    float* valuesSet = numpy_to_ptr<float>( valuesSet_py, "valuesSet", NPY_FLOAT, -1, params.nLines );
    int nItems = PyArray_DIM((PyArrayObject*)valuesSet_py, 0);
    unsigned char* statusSet = numpy_to_ptr<unsigned char>( statusSet_py, "statusSet", NPY_UBYTE, nItems, params.nLines );

    std::vector<int> keys = numpy_to_std_vector<int>(keys_py, "keys", NPY_INT);

    float* rmse = numpy_to_ptr<float>( rmse_py, "rmse", NPY_FLOAT, nItems );
    float* known = numpy_to_ptr<float>( known_py, "known", NPY_FLOAT, nItems );

    boost::shared_ptr<LineImageMatcher> matcher = MakeLIMatcher( params, metric );
    matcher->match_rmse_one_many_keys( values1, status1,
                    valuesSet, statusSet, keys, rmse, known );
}




void match_line_image_sets( PyObject* params_py, int metric, PyObject* values1_py, PyObject* status1_py,
        PyObject* values2_py, PyObject* status2_py, PyObject* rmse_py, PyObject* known_py )
{
    LineImageParams_py params_py_derived(params_py);
    LineImageParams& params = *dynamic_cast<LineImageParams*>(&params_py_derived);

    int nElements = PyArray_DIM((PyArrayObject*)values1_py,0);
    checkNumpyArray( values1_py, "values1", NPY_FLOAT, -1, params.nLines );
    checkNumpyArray( status1_py, "status1", NPY_UBYTE, nElements, params.nLines );
    checkNumpyArray( values2_py, "valuesSet", NPY_FLOAT, nElements, params.nLines );
    checkNumpyArray( status2_py, "statusSet", NPY_UBYTE, nElements, params.nLines );
    checkNumpyArray( rmse_py, "rmse", NPY_FLOAT, nElements );
    checkNumpyArray( known_py, "known", NPY_FLOAT, nElements );

    boost::shared_ptr<LineImageMatcher> matcher = MakeLIMatcher( params, metric );

    for(int i=0 ; i<nElements ; i++)
    {
        float rmse, known;
        matcher->match_rmse( (float*)PyArray_GETPTR2((PyArrayObject*)values1_py, i, 0),
                             (unsigned char*)PyArray_GETPTR2((PyArrayObject*)status1_py, i, 0),
                             (float*)PyArray_GETPTR2((PyArrayObject*)values2_py, i, 0),
                             (unsigned char*)PyArray_GETPTR2((PyArrayObject*)status2_py, i, 0),
                             rmse, known );

        *(float*)PyArray_GETPTR1((PyArrayObject*)rmse_py, i) = rmse;
        *(float*)PyArray_GETPTR1((PyArrayObject*)known_py, i) = known;
    }
}


// before openmp
/*
void match_line_images_all( PyObject* params_py, int metric, PyObject* values1_py, PyObject* status1_py,
        PyObject* values2_py, PyObject* status2_py, PyObject* rmse_py, PyObject* known_py )
{
    LineImageParams_py params_py_derived(params_py);
    LineImageParams& params = *dynamic_cast<LineImageParams*>(&params_py_derived);

    int n1 = PyArray_DIM((PyArrayObject*)values1_py, 0);
    int n2 = PyArray_DIM((PyArrayObject*)values2_py, 0);

    MapMatXf values1 = numpy_to_eigen<float, Dynamic, Dynamic>( values1_py, "values1", NPY_FLOAT, n1, params.nLines );
    MapMatXf values2 = numpy_to_eigen<float, Dynamic, Dynamic>( values2_py, "values2", NPY_FLOAT, n2, params.nLines );
    MapMatXuc status1 = numpy_to_eigen<unsigned char, Dynamic, Dynamic>( status1_py, "status1", NPY_UBYTE, n1, params.nLines );
    MapMatXuc status2 = numpy_to_eigen<unsigned char, Dynamic, Dynamic>( status2_py, "status2", NPY_UBYTE, n2, params.nLines );
    MapMatXf rmse = numpy_to_eigen<float, Dynamic, Dynamic>( rmse_py, "rmse", NPY_FLOAT, n1, n2 );
    MapMatXf known = numpy_to_eigen<float, Dynamic, Dynamic>( known_py, "known", NPY_FLOAT, n1, n2 );

    boost::shared_ptr<LineImageMatcher> matcher = MakeLIMatcher( params, metric );

    //note: not all distances are symmetric now, so do everything...
    int i;
    #pragma omp parallel for
    for(i=0 ; i<n1 ; i++)
    {
        //for(int j=i+1 ; j<n2 ; j++)
        for(int j=0 ; j<n2 ; j++)
        {
            float rmse_ij, known_ij;
            matcher->match_rmse( values1.row(i).data(), status1.row(i).data(),
                                 values2.row(j).data(), status2.row(j).data(), rmse_ij, known_ij );
            rmse(i,j) = rmse_ij;
            known(i,j) = known_ij;
            //rmse(j,i) = rmse_ij;
            //known(j,i) = known_ij;
        }
    }
}
*/


void match_line_images_all( PyObject* params_py, int metric, PyObject* values1_py, PyObject* status1_py,
        PyObject* values2_py, PyObject* status2_py, PyObject* rmse_py, PyObject* known_py, int nThreads )
{
    LineImageParams_py params_py_derived(params_py);
    LineImageParams& params = *dynamic_cast<LineImageParams*>(&params_py_derived);

    int n1 = PyArray_DIM((PyArrayObject*)values1_py, 0);
    int n2 = PyArray_DIM((PyArrayObject*)values2_py, 0);

    MapMatXf values1 = numpy_to_eigen<float, Dynamic, Dynamic>( values1_py, "values1", NPY_FLOAT, n1, params.nLines );
    MapMatXf values2 = numpy_to_eigen<float, Dynamic, Dynamic>( values2_py, "values2", NPY_FLOAT, n2, params.nLines );
    MapMatXuc status1 = numpy_to_eigen<unsigned char, Dynamic, Dynamic>( status1_py, "status1", NPY_UBYTE, n1, params.nLines );
    MapMatXuc status2 = numpy_to_eigen<unsigned char, Dynamic, Dynamic>( status2_py, "status2", NPY_UBYTE, n2, params.nLines );
    MapMatXf rmse = numpy_to_eigen<float, Dynamic, Dynamic>( rmse_py, "rmse", NPY_FLOAT, n1, n2 );
    MapMatXf known = numpy_to_eigen<float, Dynamic, Dynamic>( known_py, "known", NPY_FLOAT, n1, n2 );
    Py_BEGIN_ALLOW_THREADS;

    boost::shared_ptr<LineImageMatcher> match = MakeLIMatcher( params, metric );
    std::vector< boost::shared_ptr<LineImageMatcher> > matchers;
    matchers.push_back(match);

    int i;
    int threadId=0;
    #ifdef LaserLib_USE_OPENMP
    if( nThreads > 0 )
        omp_set_num_threads(nThreads);
    #endif
    #pragma omp parallel private(threadId) shared(matchers)
    {
        //copy the matcher, one for each thread
        #ifdef LaserLib_USE_OPENMP
        #pragma omp single
        {
            int nThreadsUsed = omp_get_num_threads();
            for( int j=1 ; j<nThreadsUsed ; j++ )
            {
                boost::shared_ptr<LineImageMatcher> thisMatcher( match->clone() );
                matchers.push_back( thisMatcher );
            }
        }
        #endif
        #pragma omp for
        for(i=0 ; i<n1 ; i++)
        {
            #ifdef LaserLib_USE_OPENMP
            threadId = omp_get_thread_num();
            #endif
            for(int j=0 ; j<n2 ; j++)
            {
                float rmse_ij, known_ij;
                matchers[threadId]->match_rmse( values1.row(i).data(), status1.row(i).data(),
                                                values2.row(j).data(), status2.row(j).data(),
                                                rmse_ij, known_ij );
                rmse(i,j) = rmse_ij;
                known(i,j) = known_ij;
            }
        }
    }
    Py_END_ALLOW_THREADS;
}




ObjLineImagesAligned ObjLineImagesAligned_from_py( PyObject* data_py )
{
    Mat3<float>::type alignVectors =
            get_attribute_numpy_array<float, Dynamic, 3>(
                data_py, "alignVectors", NPY_FLOAT);
    int nPoints = alignVectors.rows();

    Vect<unsigned char>::type alignType =
            get_attribute_numpy_array<unsigned char, Dynamic, 1>(
                data_py, "alignType", NPY_UBYTE, nPoints);

    MapMatXf values =
        get_attribute_numpy_array<float, Dynamic, Dynamic>(
                data_py, "values", NPY_FLOAT, nPoints);
    int nLines = values.cols();

    MapMatXuc status =
        get_attribute_numpy_array<unsigned char, Dynamic, Dynamic>(
                data_py, "status", NPY_UBYTE, nPoints, nLines);

    // apparently, the compiler is smart enough to not make unnecessary copies
    return ObjLineImagesAligned( alignVectors, alignType, values, status, nPoints );
}



void ObjLineImagesAlignedDataSet_from_py( PyObject* dataset_py, std::vector<ObjLineImagesAligned>& dataset )
{
    if( !PyList_Check(dataset_py) )
    {
        PyErr_SetString(PyExc_ValueError, "dataset must be a list" );
        throw_error_already_set();
    }
    Py_ssize_t length = PyList_Size(dataset_py);
    for( Py_ssize_t i=0 ; i<length ; i++ )
    {
        dataset.push_back( ObjLineImagesAligned_from_py( PyList_GetItem( dataset_py, i ) ) );
    }
}



ObjLineImages ObjLineImages_from_py( PyObject* data_py )
{
    MapMatXf values =
        get_attribute_numpy_array<float, Dynamic, Dynamic>(
                data_py, "values", NPY_FLOAT);
    int nPoints = values.rows();
    int nLines = values.cols();

    MapMatXuc status =
        get_attribute_numpy_array<unsigned char, Dynamic, Dynamic>(
                data_py, "status", NPY_UBYTE, nPoints, nLines);

    return ObjLineImages( values, status, nPoints );
}



void ObjLineImagesDataSet_from_py( PyObject* dataset_py, std::vector<ObjLineImages>& dataset )
{
    if( !PyList_Check(dataset_py) )
    {
        PyErr_SetString(PyExc_ValueError, "dataset must be a list" );
        throw_error_already_set();
    }
    Py_ssize_t length = PyList_Size(dataset_py);
    for( Py_ssize_t i=0 ; i<length ; i++ )
    {
        dataset.push_back( ObjLineImages_from_py( PyList_GetItem( dataset_py, i ) ) );
    }
}


