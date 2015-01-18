/*! PCAFrame_py.h
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
 * \date       05-09-2012
*/

#ifndef PCA_FRAME_PY
#define PCA_FRAME_PY

#include <Python.h>
#include <numpy/arrayobject.h>
#include <boost/python/overloads.hpp>
#include "Features/PCAFrame.h"


class PCAFrames_py : public PCAFrames
{
public:
    PCAFrames_py( int n )
        :   PCAFrames(n)
    {}

    PyObject* get_P()
    {
        npy_intp dims[2] = {size,3};
        PyObject* obj = PyArray_SimpleNewFromData(2, dims, NPY_FLOAT, P.data());
        return PyArray_Return( (PyArrayObject*) obj );
    }

    PyObject* get_R()
    {
        npy_intp dims[3] = {size,3,3};
        PyObject* obj = PyArray_SimpleNewFromData(3, dims, NPY_FLOAT, R.data());
        return PyArray_Return( (PyArrayObject*) obj );
    }

    PyObject* get_frameType()
    {
        npy_intp dims[1] = {size};
        PyObject* obj = PyArray_SimpleNewFromData(1, dims, NPY_UBYTE, frameType.data());
        return PyArray_Return( (PyArrayObject*) obj );
    }

    PyObject* get_alignVect()
    {
        npy_intp dims[2] = {size,3};
        PyObject* obj = PyArray_SimpleNewFromData(2, dims, NPY_FLOAT, alignVect.data());
        return PyArray_Return( (PyArrayObject*) obj );
    }

    PyObject* get_pcaId()
    {
        npy_intp dims[1] = {size};
        PyObject* obj = PyArray_SimpleNewFromData(1, dims, NPY_INT, pcaId.data());
        return PyArray_Return( (PyArrayObject*) obj );
    }

    PyObject* __getstate__();
    void __setstate__(PyObject* state);
    PyObject* __getinitargs__();
};


void ComputePCAFrames_py(
        PyObject* meanP_py, PyObject* evals_py, PyObject* evects_py, PCAFrames_py& frames,
        float surfThresh = 0.3, float linThresh = 0.5, float ssRad = 0.01);

// macro for default arguments
BOOST_PYTHON_FUNCTION_OVERLOADS(ComputePCAFrames_overloads, ComputePCAFrames_py, 4, 7)


#endif //PCA_FRAME_PY
