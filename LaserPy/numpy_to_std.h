/*! numpy_to_std.h
 * Helper functions for converting numpy arrays to std array-like structures.
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
 * \date       08-03-2011
*/

#ifndef NUMPY_TO_STD
#define NUMPY_TO_STD

#include <Python.h>
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <numpy/arrayobject.h>
#include <vector>
#include <string>
#include <boost/python.hpp>

#include "numpy_helpers.h"


/* Convert numpy array to a vector of specified type.
 *
 * Input array can be of any shape (is flattened), and not c-contiguous.
*/
template <typename T>
std::vector<T> numpy_to_std_vector(PyObject* arr, const std::string& name, NPY_TYPES type_num)
{
    // check it's a numpy array
    if( !PyArray_Check(arr) )
    {
        PyErr_SetString(PyExc_TypeError, "numpy_to_std_vector: input is not a numpy array");
        boost::python::throw_error_already_set();
    }
    PyArrayObject* arr_npy = (PyArrayObject*)arr;

    // check type
    if( PyArray_DESCR(arr_npy)->type_num != type_num )
    {
        if( PyArray_DESCR(arr_npy)->type_num == NPY_LONG && PyArray_ITEMSIZE(arr_npy) == 4 && type_num == NPY_INT)
        {
            //numpy stupidity: on 32bit, an array made with type numpy.int32 is NPY_LONG
        }
        else
        {
            std::stringstream errStr;
            errStr << name << "- invalid array datatype, is " <<
                      numpy_type_to_string(PyArray_DESCR(arr_npy)->type_num) <<
                      " (" << PyArray_ITEMSIZE(arr_npy) << " bytes per item), should be " <<
                      numpy_type_to_string(type_num);
            PyErr_SetString(PyExc_ValueError, errStr.str().c_str() );
            boost::python::throw_error_already_set();
        }
    }

    // Handle zero-sized arrays specially
    if (PyArray_SIZE(arr_npy) == 0) {
        return std::vector<T>();
    }

    // Create the iterator.
    NpyIter* iter = NpyIter_New(arr_npy, NPY_ITER_READONLY|
                             NPY_ITER_EXTERNAL_LOOP|
                             NPY_ITER_REFS_OK,
                        NPY_KEEPORDER, NPY_NO_CASTING,
                        NULL);
    if (iter == NULL) {
        PyErr_SetString(PyExc_ValueError, "numpy_to_std_vector: could not create an iterator");
        boost::python::throw_error_already_set();
    }

    std::vector<T> dest;
    dest.resize( PyArray_SIZE(arr_npy) );
    typename std::vector<T>::iterator destIt = dest.begin();

    NpyIter_IterNextFunc *iternext = NpyIter_GetIterNext(iter, NULL);
    char** dataptr = NpyIter_GetDataPtrArray(iter);
    npy_intp* strideptr = NpyIter_GetInnerStrideArray(iter);
    npy_intp* sizeptr = NpyIter_GetInnerLoopSizePtr(iter);
    npy_intp iop, nop = NpyIter_GetNOp(iter);

    do
    {
        char* data = *dataptr;
        npy_intp count = *sizeptr;
        npy_intp stride = *strideptr;
        while (count--)
        {
            for (iop = 0; iop < nop; ++iop, data+=stride)
            {
                *destIt++ = *(T*)data;
            }
        }
    } while (iternext(iter));

    NpyIter_Deallocate(iter);
    return dest;
}




#endif //NUMPY_TO_STD
