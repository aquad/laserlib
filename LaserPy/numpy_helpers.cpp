/*! numpy_helpers.cpp
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
#include "numpy_helpers.h"
#include <string>
#include <sstream>
#include <boost/python.hpp>

using namespace boost::python;


std::string numToOrdinal( int num )
{
    std::stringstream ordinal;
    ordinal << num;
    if( num == 1 ){ ordinal << "st"; }
    else if( num == 2 ){ ordinal << "nd"; }
    else if( num == 3 ){ ordinal << "rd"; }
    else { ordinal << num << "th"; }
    return ordinal.str();
}


//! Convert NPY_TYPES to string for error messages
std::string numpy_type_to_string( int type )
{
    std::string name;
    switch(type)
    {
        case NPY_BOOL:
            name = "bool";
            break;
        case NPY_BYTE:
            name = "int8";
            break;
        case NPY_UBYTE:
            name = "uint8";
            break;
        case NPY_SHORT:
            name = "int16";
            break;
        case NPY_USHORT:
            name = "uint16";
            break;
        case NPY_INT:
            name = "int32";
            break;
        case NPY_UINT:
            name = "uint32";
            break;
        case NPY_LONG:
            name = "int32";
            break;
        case NPY_ULONG:
            name = "uint32";
            break;
        case NPY_LONGLONG:
            name = "int64";
            break;
        case NPY_ULONGLONG:
            name = "uint64";
            break;
        case NPY_FLOAT:
            name = "float32";
            break;
        case NPY_DOUBLE:
            name = "float64";
            break;
        case NPY_LONGDOUBLE:
            name = "float128";
            break;
        case NPY_STRING:
            name = "string";
            break;
        case NPY_UNICODE:
            name = "unicode";
            break;
        case NPY_DATETIME:
            name = "datetime";
            break;
        default:
            std::stringstream namestream;
            namestream << "enum NPY_TYPES=" << type;
            name = namestream.str();
            break;
    }
    return name;
}



void checkDim( PyObject* arr, const std::string& name, int dim, int size )
{
    PyArrayObject* arr_npy = (PyArrayObject*)arr;
    int nDims = PyArray_NDIM(arr_npy);
    if( (size == -1 || size > 0 ) && nDims < dim+1 ) //dimension must exist
    {
        std::stringstream errStr;
        errStr << name << "- array must have a " << numToOrdinal(dim+1) << " dimension" <<
                  ", only has " << nDims << " dimensions";
        PyErr_SetString(PyExc_ValueError, errStr.str().c_str() );
        throw_error_already_set();
    }

    if( size > 0 )
    {
        if( PyArray_DIM(arr_npy,dim) != size )
        {
            std::stringstream errStr;
            errStr << name << "- invalid array shape in dimension " << dim << ", is "
                    << PyArray_DIM(arr_npy,dim) << ", should be " << size;
            PyErr_SetString(PyExc_ValueError, errStr.str().c_str() );
            throw_error_already_set();
        }
    }
    else if( size==0 && nDims >= dim+1 ) //can't have this dimension
    {
        std::stringstream errStr;
        errStr << name << "- array should not have a " << numToOrdinal(dim+1) << " dimension";
        PyErr_SetString(PyExc_ValueError, errStr.str().c_str() );
        throw_error_already_set();
    }
}



void checkType( PyObject* arr, const std::string& name, NPY_TYPES type_num )
{
    PyArrayObject* arr_npy = (PyArrayObject*)arr;
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
}



void checkNumpyArray( PyObject* arr, const std::string& name, NPY_TYPES type_num,
                      int shape0, int shape1, int shape2, int shape3 )
{
    if( !PyArray_Check(arr) )
    {
        std::stringstream errStr;
        errStr << name << " is not a numpy array";
        PyErr_SetString(PyExc_TypeError, errStr.str().c_str() );
        throw_error_already_set();
    }

    checkType( arr, name, type_num );

    if( shape0 != -2 )
        checkDim( arr, name, 0, shape0 );
    if( shape1 != -2 )
        checkDim( arr, name, 1, shape1 );
    if( shape2 != -2 )
        checkDim( arr, name, 2, shape2 );
    if( shape3 != -2 )
        checkDim( arr, name, 3, shape3 );

    if( !PyArray_ISCONTIGUOUS((PyArrayObject*)arr) )
    {
        std::stringstream errStr;
        errStr << name << "- not c contiguous [use numpy.array(data, order='C')]";
        PyErr_SetString(PyExc_ValueError, errStr.str().c_str() );
        throw_error_already_set();
    }
}


PyObject* get_attribute( PyObject* container, const std::string& name )
{
    PyObject* pyob = PyObject_GetAttrString(container, name.c_str());
    if( pyob == NULL )
    {
        std::stringstream errStr;
        errStr << "no attribute named " << name;
        PyErr_SetString(PyExc_AttributeError, errStr.str().c_str() );
        boost::python::throw_error_already_set();
    }
    return pyob;
}


template <>
double get_attribute_value( PyObject* container, const std::string& name )
{
    PyObject* pyob = get_attribute(container, name);
    double retVal = PyFloat_AsDouble( pyob );
    Py_DECREF(pyob);
    return retVal;
}


template <>
float get_attribute_value( PyObject* container, const std::string& name )
{
    return (float)get_attribute_value<double>( container, name );
}

template <>
int get_attribute_value( PyObject* container, const std::string& name )
{
    PyObject* pyob = get_attribute(container, name);
    int retVal = PyInt_AsLong( pyob );
    Py_DECREF(pyob);
    return retVal;
}

