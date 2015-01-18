/*! numpy_helpers.h
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

#ifndef NUMPY_HELPERS
#define NUMPY_HELPERS

#include <Python.h>
#include <string>

#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <numpy/arrayobject.h>


std::string numToOrdinal( int num );
std::string numpy_type_to_string( int type );
void checkDim( PyObject* arr, const std::string& name, int dim, int size );
void checkType( PyObject* arr, const std::string& name, NPY_TYPES type_num );

/*!
  Ensure numpy array is of specified type, shape, and c-contiguous.

  :param arr: numpy array
  :param name: array name, goes in error message for easy debugging
  :param type_num: eg. NPY_INT, see numpy/ndarraytypes.h
  :param shape0: shape in dimension 0.
        Special values:
        0 : this dimension should not exist.
        -1: this dimension should exist, but no specific size is needed.
        -2: don't care about existance and shape.
  */
void checkNumpyArray( PyObject* arr, const std::string& name,
        NPY_TYPES type_num, int shape0, int shape1=0, int shape2=0,
        int shape3=0 );


template <typename T>
T* numpy_to_ptr( PyObject* arr, const std::string& name, NPY_TYPES type_num,
                 int shape0=-1, int shape1=0, int shape2=0, int shape3=0 )
{
    checkNumpyArray( arr, name, type_num, shape0, shape1, shape2, shape3 );
    return (T*)PyArray_DATA( (PyArrayObject*)arr );
}



// Warning: returned pyobject ownership goes to caller (call decref if not keeping)
PyObject* get_attribute( PyObject* container, const std::string& name );


// maybe move to a new python-only file?
//! Get an attribute of basic number type
template <typename T>
T get_attribute_value( PyObject* container, const std::string& name );



#endif //NUMPY_HELPERS
