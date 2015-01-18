/*! numpy_to_eigen.h
 * Helper functions for converting numpy arrays to eigen arrays.
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

#ifndef NUMPY_TO_EIGEN
#define NUMPY_TO_EIGEN

#include <Python.h>
#include <string>
#include <sstream>

#define EIGEN_DEFAULT_TO_ROW_MAJOR 1
#include <Eigen/Core>

#include <boost/python.hpp>

#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <numpy/arrayobject.h>

#include "numpy_helpers.h"



/*! Copy a numpy array to an eigen matrix.
 *
 *  If an array has more than 2 dimensions, they are flattened into the 2nd dimension (columns).
 *  Size must be either Dynamic or known at compile time.
 *
 *  (np_rows, np_cols) is for checking the dimensions of the numpy array. -1
 *  means "don't care", defaulting to the size restrictions of the eigen type,
 *  if any (not quite the same meaning as in checkNumpyArray()).
 */
template <typename T, int rows, int cols>
Eigen::Matrix<T, rows, cols>
        numpy_to_eigen_matrix( PyObject* arr,
                       const std::string& name, NPY_TYPES type_num,
                       int np_rows=-1, int np_cols=-1 )
{
    if( !PyArray_Check(arr) )
    {
        std::stringstream errStr;
        errStr << name << " is not a numpy array";
        PyErr_SetString(PyExc_TypeError, errStr.str().c_str() );
        boost::python::throw_error_already_set();
    }
    PyArrayObject* arr_npy = (PyArrayObject*)arr;

    checkType( arr, name, type_num );

    // Determine the size of the eigen matrix from the numpy array.
    int nDims = PyArray_NDIM(arr_npy);
    int actual_rows = PyArray_DIM(arr_npy, 0);
    int actual_cols = 1;
    if( nDims == 2 )
        actual_cols = PyArray_DIM(arr_npy, 1);
    else if( nDims > 2 )
    {
        // flatten further dimensions into dimension 2
        for( int i=1 ; i<nDims ; i++ )
            actual_cols *= PyArray_DIM(arr_npy,i);
    }

    // Check numpy array dimensions are compatible with templated eigen dimensions.
    // Only occurs when returned eigen type is fixed size, but not the same as
    // the numpy array.
    if( (cols != -1) && (actual_cols != cols) ||
            (rows != -1) && (actual_rows != rows) )
    {
        std::stringstream errStr;
        errStr << "Array " << name << " is size (" <<
            actual_rows << "," << actual_cols <<
            "), but required eigen size is (" <<
            rows << "," << cols << ")";
        PyErr_SetString(PyExc_ValueError, errStr.str().c_str() );
        boost::python::throw_error_already_set();
    }

    // Check extra constraints on array dimensions (np_rows, np_cols).
    // checkNumpyArray interprets -1 to mean "don't care, but the dimension
    // must exist"; in this case, the (2nd) dimension needn't exist.
    int check_rows = np_rows;
    int check_cols = np_cols;
    if( nDims == 1 )
        check_cols = 0;

    if( np_rows != -2 )
        checkDim( arr, name, 0, check_rows );
    if( (np_cols != -2) || (nDims > 2) ) //if flattened, don't check 2nd dim
        checkDim( arr, name, 1, check_cols );

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

    Eigen::Matrix<T, rows, cols> dest( actual_rows, actual_cols );
    T* destptr = dest.data();

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
                *destptr++ = *(T*)data;
            }
        }
    } while (iternext(iter));

    NpyIter_Deallocate(iter);
    return dest;
}




/*! Convert a numpy array to an eigen map (no copy).
 *
 *  If an array has more than 2 dimensions, they are flattened into the 2nd dimension (columns).
 *  Size must be either Dynamic or known at compile time.
 *
 *  (np_rows, np_cols) is for checking the dimensions of the numpy array. -1
 *  means "don't care", defaulting to the size restrictions of the eigen type,
 *  if any (not quite the same meaning as in checkNumpyArray()).
 */
template <typename T, int rows, int cols>
Eigen::Map< Eigen::Matrix<T, rows, cols> >
        numpy_to_eigen( PyObject* arr,
                       const std::string& name, NPY_TYPES type_num,
                       int np_rows=-1, int np_cols=-1 )
{
    if( !PyArray_Check(arr) )
    {
        std::stringstream errStr;
        errStr << name << " is not a numpy array";
        PyErr_SetString(PyExc_TypeError, errStr.str().c_str() );
        boost::python::throw_error_already_set();
    }
    PyArrayObject* arr_npy = (PyArrayObject*)arr;

    checkType( arr, name, type_num );

    // Determine the size of the eigen matrix from the numpy array.
    int nDims = PyArray_NDIM(arr_npy);
    int actual_rows = PyArray_DIM(arr_npy, 0);
    int actual_cols = 1;
    if( nDims == 2 )
        actual_cols = PyArray_DIM(arr_npy, 1);
    else if( nDims > 2 )
    {
        // flatten further dimensions into dimension 2
        for( int i=1 ; i<nDims ; i++ )
            actual_cols *= PyArray_DIM(arr_npy,i);
    }

    // Check numpy array dimensions are compatible with templated eigen dimensions.
    // Only occurs when returned eigen type is fixed size, but not the same as
    // the numpy array.
    if( (cols != -1) && (actual_cols != cols) ||
            (rows != -1) && (actual_rows != rows) )
    {
        std::stringstream errStr;
        errStr << "Array " << name << " is size (" <<
            actual_rows << "," << actual_cols <<
            "), but required eigen size is (" <<
            rows << "," << cols << ")";
        PyErr_SetString(PyExc_ValueError, errStr.str().c_str() );
        boost::python::throw_error_already_set();
    }

    // Check extra constraints on array dimensions (np_rows, np_cols).
    // checkNumpyArray interprets -1 to mean "don't care, but the dimension
    // must exist"; in this case, the (2nd) dimension needn't exist.
    int check_rows = np_rows;
    int check_cols = np_cols;
    if( nDims == 1 )
        check_cols = 0;

    if( np_rows != -1 )
        checkDim( arr, name, 0, check_rows );
    if( (np_cols != -1) && (nDims == 2) )
        checkDim( arr, name, 1, check_cols );
    // If there are more than 2 dimensions, flattened
    if( (np_cols != -1) && (nDims > 2) && (check_cols != actual_cols) )
    {
        std::stringstream errStr;
        errStr << name << "- invalid array shape in dimension 2 & up, is " <<
            actual_cols << " (flattened into dimension 2), should be " << check_cols;
        PyErr_SetString(PyExc_ValueError, errStr.str().c_str() );
        boost::python::throw_error_already_set();
    }

    T* data_p = (T*)PyArray_DATA(arr_npy);

    return Eigen::Map< Eigen::Matrix<T, rows, cols> >(
            data_p, actual_rows, actual_cols );
}



/*! For when a python object contains a numpy array as an attribute.
 *
 *  See numpy_to_eigen()
 */
template <typename T, int rows, int cols>
Eigen::Map< Eigen::Matrix<T, rows, cols> >
        get_attribute_numpy_array( PyObject* container,
                                   const std::string& name, NPY_TYPES type_num,
                                   int np_rows=-1, int np_cols=-1 )
{
    PyObject* arr = PyObject_GetAttrString(container, name.c_str());
    if( arr == NULL )
    {
        std::stringstream errStr;
        errStr << "no attribute named " << name;
        PyErr_SetString(PyExc_AttributeError, errStr.str().c_str() );
        boost::python::throw_error_already_set();
    }
    Eigen::Map< Eigen::Matrix<T, rows, cols> > outMap =
            numpy_to_eigen<T,rows,cols>( arr, name, type_num, np_rows, np_cols );
    Py_DECREF(arr);
    return outMap;
}

#endif //NUMPY_TO_EIGEN
