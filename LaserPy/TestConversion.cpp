/*! TestConversion.cpp
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
 * \date       16-01-2013
*/

#include <Python.h>
#define PY_ARRAY_UNIQUE_SYMBOL _LaserPy_ARRAY_API
#define NO_IMPORT_ARRAY

#include "numpy_to_eigen.h"
#include "numpy_to_std.h"

#include <boost/python.hpp>

#include "TestConversion.h"

using namespace boost::python;


void export_TestConversion()
{
    def("_test_checkNumpyArray", &checkNumpyArray_py, checkNumpyArray_py_overloads(
        "_test_checkNumpyArray(arr, name, type_num, shape0[, shape1=0, shape2=0, shape3=0])\n\n"
        "Will throw an exception if the array is the wrong type/shape, or not c-contiguous.\n\n") );


    def("_test_numpy_to_std_vector", &numpy_to_std_vector_py,
        "_test_numpy_to_std_vector(arr, cTypeName)\n\n"
        "Convert array to an std vector, then back again.\n\n"
        "Parameters\n"
        "----------\n"
        "arr : ndarray\n"
        "cTypeName : str\n"
        "   int, float, double, long long\n\n"
        "Returns\n"
        "-------\n"
        "outArr : ndarray\n"
        "   If successful, a flat, type-casted *arr*");


    def("_test_numpy_float_to_eigen", &test_numpy_float_to_eigen,
        "_test_numpy_float_to_eigen(arr, name, np_rows, np_cols)\n\n"
        "Tests numpy_to_eigen function, which converts an nd array to a 2d eigen matrix, "
        "where higher dimensions are flattened into the dimension 1. Will throw on error\n\n"
        "Parameters\n"
        "----------\n"
        "arr : ndarray, float32\n"
        "name : str\n"
        "np_rows : int\n"
        "   number of rows to check the array for, -1 = don't care.\n"
        "np_cols : int\n"
        "   number of cols (flattened) to check the array for\n");
}



void checkNumpyArray_py( PyObject* arr, const std::string& name,
                        int type_num, int shape0, int shape1, int shape2,
                        int shape3 )
{
    checkNumpyArray( arr, name, (NPY_TYPES)type_num, shape0, shape1, shape2, shape3 );
}



PyObject* numpy_to_std_vector_py(PyObject* arr, const std::string& name, int type_num)
{
    PyArrayObject* retArray;
    npy_intp dims[2];
    dims[0] = 0;

    NPY_TYPES npy_type_num = (NPY_TYPES)type_num;
    switch(npy_type_num)
    {
        case NPY_INT:
        {
            std::vector<int> vec = numpy_to_std_vector<int>(arr, name, npy_type_num);
            dims[0] = vec.size();
            retArray = (PyArrayObject*) PyArray_SimpleNew(1, dims, NPY_INT);
            memcpy( PyArray_DATA(retArray), &(vec[0]), vec.size()*sizeof(int) );
            break;
        }
        case NPY_FLOAT:
        {
            std::vector<float> vec = numpy_to_std_vector<float>(arr, name, npy_type_num);
            dims[0] = vec.size();
            retArray = (PyArrayObject*) PyArray_SimpleNew(1, dims, NPY_FLOAT);
            memcpy( PyArray_DATA(retArray), &(vec[0]), vec.size()*sizeof(float) );
            break;
        }
        case NPY_DOUBLE:
        {
            std::vector<int> vec = numpy_to_std_vector<int>(arr, name, npy_type_num);
            dims[0] = vec.size();
            retArray = (PyArrayObject*) PyArray_SimpleNew(1, dims, NPY_DOUBLE);
            memcpy( PyArray_DATA(retArray), &(vec[0]), vec.size()*sizeof(double) );
            break;
        }
        case NPY_LONGLONG:
        {
            std::vector<long long> vec = numpy_to_std_vector<long long>(arr, name, npy_type_num);
            dims[0] = vec.size();
            retArray = (PyArrayObject*) PyArray_SimpleNew(1, dims, NPY_LONGLONG);
            memcpy( PyArray_DATA(retArray), &(vec[0]), vec.size()*sizeof(long long) );
            break;
        }
        default:
            retArray = (PyArrayObject*)PyArray_EMPTY(1, dims, NPY_INT, 0);
    }
    return PyArray_Return(retArray);
}


void test_numpy_float_to_eigen( PyObject* arr, const std::string& name, int np_rows, int np_cols )
{
    using namespace Eigen;
    Map< MatrixXf > evects = numpy_to_eigen<float, Dynamic, Dynamic>( arr, name, NPY_FLOAT, np_rows, np_cols );
}



