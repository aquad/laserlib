/*! LineDistrib_py.cpp
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
 * \date       24-07-2012
*/

#include <Python.h>
#define PY_ARRAY_UNIQUE_SYMBOL _LaserPy_ARRAY_API
#define NO_IMPORT_ARRAY
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <numpy/arrayobject.h>
#include <boost/python.hpp>

#include "LineDistrib_py.h"
#include "LineImage_py.h"
#include "LaserPy/numpy_to_eigen.h"

using namespace boost::python;


void export_LineDistrib()
{
    //for automatic downcasting
    class_<LineDistrib>("LineDistrib", no_init);

    class_<LineDistrib_py, boost::shared_ptr<LineDistrib_py>, bases<LineDistrib> >("LineDistrib",
             "LineDistrib(params, binLength)\n\n",
             no_init)

            .def( "__init__", boost::python::make_constructor( &LineDistrib_py_constructor ) )

            .def("Copy", &LineDistrib_py::Copy,
                 "Copy()\n")

            .def("Set", &LineDistrib_py::Set_py,
                 "Set(values, status)\n\n")

            .def("Merge", &LineDistrib_py::Merge,
                 "Merge(other)\n")

            .def("GetBins", &LineDistrib_py::GetBins,
                 "GetBins()\n")

            .def("GetOdds", &LineDistrib_py::GetOdds,
                 "GetOdds()\n")

            .enable_pickling()
            .def("__getinitargs__", &LineDistrib_py::__getinitargs__)
            .def("__getstate__", &LineDistrib_py::__getstate__)
            .def("__setstate__", &LineDistrib_py::__setstate__);
}



LineDistrib_py::LineDistrib_py( LineImageParams& params, float binLength )
    :   LineDistrib( params, binLength )
{}


boost::shared_ptr<LineDistrib_py> LineDistrib_py_constructor(
        PyObject* params_py, float binLength )
{
    //these are temporary, but they get copied in the ComputeLineImage constructor.
    LineImageParams_py params_py_derived(params_py);
    LineImageParams& params = *dynamic_cast<LineImageParams*>(&params_py_derived);

    boost::shared_ptr<LineDistrib_py> returnObj(
            new LineDistrib_py( params, binLength ) );
    //for pickling, need to store constructor python args
    returnObj->params_py = params_py;
    Py_INCREF(params_py);
    returnObj->binLength_py = binLength;

    return returnObj;
}


LineDistrib_py::LineDistrib_py( const LineDistrib_py& rhs )
    :   LineDistrib( rhs ),
        params_py( rhs.params_py ),
        binLength_py( rhs.binLength_py )
{
    Py_INCREF(params_py);
}


boost::shared_ptr<LineDistrib_py> LineDistrib_py::Copy()
{
    return boost::shared_ptr<LineDistrib_py>( new LineDistrib_py( *this ) );
}


void LineDistrib_py::Set_py( PyObject* values_py, PyObject* status_py )
{
    float* values = (float*)PyArray_DATA((PyArrayObject*)values_py);
    unsigned char* status = (unsigned char*)PyArray_DATA((PyArrayObject*)status_py);
    Set(values, status);
}


PyObject* LineDistrib_py::GetBins()
{
    PyObject* outList = PyList_New(bins_.size());
    npy_intp dims[2];
    for( int i=0; i < bins_.size() ; i++ )
    {
        dims[0] = bins_[i].size();
        PyObject* lineArray = PyArray_SimpleNewFromData( 1, dims, NPY_FLOAT, &(bins_[i][0]) );
        int val = PyList_SetItem(outList, i, lineArray);
    }
    return outList;
}


PyObject* LineDistrib_py::GetOdds()
{
    npy_intp dims[1] = {static_cast<npy_intp>(logOddsIntercept_.size())};
    PyObject* odds_py = PyArray_SimpleNewFromData(1, dims, NPY_FLOAT, &(logOddsIntercept_[0]));
    return odds_py;
}


PyObject* LineDistrib_py::__getinitargs__()
{
    PyObject* tup = PyTuple_Pack(2, params_py, PyFloat_FromDouble(binLength_py));
    return tup;
}


PyObject* LineDistrib_py::__getstate__()
{
    PyObject* bins_py = GetBins();
    PyObject* odds_py = GetOdds();
    PyObject* tup = PyTuple_Pack(2, bins_py, odds_py);
    return tup;
}


void LineDistrib_py::__setstate__(PyObject* state)
{
    PyObject* bins_py = PyTuple_GetItem(state,0);
    PyArrayObject* odds_py = (PyArrayObject*) PyTuple_GetItem(state,1);
    //copy across data
    for( int i=0 ; i<bins_.size() ; i++ )
    {
        PyArrayObject* line_py = (PyArrayObject*) PyList_GetItem( bins_py, i );
        memcpy( &(bins_[i][0]), PyArray_DATA(line_py), bins_[i].size()*sizeof(float) );
    }
    memcpy( &(logOddsIntercept_[0]), PyArray_DATA(odds_py), logOddsIntercept_.size()*sizeof(float) );
}


