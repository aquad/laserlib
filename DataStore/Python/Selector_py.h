/*! Selector_py.h
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
 * \date       23-05-2011
*/

#ifndef SELECTOR_PY_HEADER_GUARD
#define SELECTOR_PY_HEADER_GUARD

#include "../export.h"
#include <Python.h>
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <numpy/arrayobject.h>
#include <boost/python.hpp>

#include "DataStore/Selector.h"



//still pure virtual
class LASERLIB_DATASTORE_EXPORT Selector_py : virtual public Selector
{
public:
    Selector_py(){}

    PyObject* SelectRegion_py( int centre )
    {
        std::vector<int>& neighs = SelectRegion(centre);
        npy_intp dims[1] = {static_cast<npy_intp>(neighs.size())};
        PyArrayObject* neighs_py = (PyArrayObject *)PyArray_SimpleNew(1, dims, NPY_INT );
        memcpy( PyArray_DATA(neighs_py), &(neighs[0]), sizeof(int)*neighs.size() );
        return PyArray_Return(neighs_py);
    }
};



//needed to have a python binding for pure virtual Selector,
// so we can downcast derived classes automatically.
struct LASERLIB_DATASTORE_EXPORT Selector_callback : public Selector
{
    // constructor storing initial self parameter
    Selector_callback(PyObject *p) : self(p) {}

    // In case hello is returned by-value from a wrapped function
    Selector_callback(PyObject *p, const Selector& x)
        : Selector(x), self(p) {}

    // Override virtual function to call back into Python
    std::vector<int>& SelectRegion( unsigned int centre )
        { return boost::python::call_method<std::vector<int>&>(self, "SelectRegion", centre); }

    // same for clone()
    boost::shared_ptr<Selector> clone()
        { return boost::python::call_method< boost::shared_ptr<Selector> >(self, "clone"); }

 private:
    std::vector<int> neigh;
    PyObject* self;
};




//this class allows a SelectRegion function to be defined in python, for use in c++
class LASERLIB_DATASTORE_EXPORT Selector_from_py : public Selector
{
public:
    Selector_from_py()
        { neigh.reserve(300); }

    virtual PyObject* SelectRegion_py( int centre )=0;

    std::vector<int>& SelectRegion( int centre )
    {
        PyArrayObject* neigh_py = (PyArrayObject*) SelectRegion_py(centre);
        neigh.resize( PyArray_DIM(neigh_py,0) );
        memcpy( &(neigh[0]), PyArray_DATA(neigh_py), sizeof(int)*neigh.size() );
        return neigh;
    }

    //clone?

private:
    std::vector<int> neigh;
};


//needed to have a python binding for pure virtual Selector_from_py,
// so we can downcast derived classes automatically.
struct LASERLIB_DATASTORE_EXPORT Selector_from_py_callback : public Selector_from_py
{
    // constructor storing initial self parameter
    Selector_from_py_callback(PyObject *p) : self(p) {}

    // In case hello is returned by-value from a wrapped function
    Selector_from_py_callback(PyObject *p, const Selector_from_py& x)
        : Selector_from_py(x), self(p) {}

    // Override virtual function to call back into Python
    PyObject* SelectRegion_py( int centre )
        { return boost::python::call_method<PyObject*>(self, "SelectRegion", centre); }
    //note that SelectRegion here refers to a python function, not the c++ SelectRegion, which is not visible from python.

 private:
    PyObject* self;
};


#endif //SELECTOR_PY_HEADER_GUARD
