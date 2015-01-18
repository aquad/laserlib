/*! VelodyneDb_py.h
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
 * \date       10-05-2011
*/

#ifndef VELODYNE_DB_PY
#define VELODYNE_DB_PY

#include <Python.h>
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <numpy/arrayobject.h>
#include <boost/python.hpp>
#include "DataStore/VelodyneDb.h"


struct VelodyneDb_py : VelodyneDb
{
    VelodyneDb_py( PyObject* db_py )
    {
        PyArrayObject* pyob = (PyArrayObject*) PyObject_GetAttrString(db_py, "dc");
        if( pyob == NULL )
        {
            PyErr_SetString(PyExc_AttributeError, "VelodyneDb: no dc attribute" );
            boost::python::throw_error_already_set();
        }
        memcpy( dc, (double *)PyArray_DATA( pyob ), 64*sizeof(double) );
        Py_DECREF(pyob);

        pyob = (PyArrayObject*) PyObject_GetAttrString(db_py, "rc");
        if( pyob == NULL )
        {
            PyErr_SetString(PyExc_AttributeError, "VelodyneDb: no rc attribute" );
            boost::python::throw_error_already_set();
        }
        memcpy( rc, (double *)PyArray_DATA( pyob ), 64*sizeof(double) );
        Py_DECREF(pyob);

        pyob = (PyArrayObject*) PyObject_GetAttrString(db_py, "vc");
        if( pyob == NULL )
        {
            PyErr_SetString(PyExc_AttributeError, "VelodyneDb: no vc attribute" );
            boost::python::throw_error_already_set();
        }
        memcpy( vc, (double *)PyArray_DATA( pyob ), 64*sizeof(double) );
        Py_DECREF(pyob);

        pyob = (PyArrayObject*) PyObject_GetAttrString(db_py, "voffc");
        if( pyob == NULL )
        {
            PyErr_SetString(PyExc_AttributeError, "VelodyneDb: no voffc attribute" );
            boost::python::throw_error_already_set();
        }
        memcpy( voffc, (double *)PyArray_DATA( pyob ), 64*sizeof(double) );
        Py_DECREF(pyob);

        pyob = (PyArrayObject*) PyObject_GetAttrString(db_py, "hoffc");
        if( pyob == NULL )
        {
            PyErr_SetString(PyExc_AttributeError, "VelodyneDb: no hoffc attribute" );
            boost::python::throw_error_already_set();
        }
        memcpy( hoffc, (double *)PyArray_DATA( pyob ), 64*sizeof(double) );
        Py_DECREF(pyob);
    }
};

#endif //VELODYNE_DB_PY
