/*! PCAGrid_py.cpp
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
 * \date       29-01-2012
*/

#include <Python.h>
#define PY_ARRAY_UNIQUE_SYMBOL _LaserPy_ARRAY_API
#define NO_IMPORT_ARRAY
#include "LaserPy/numpy_to_eigen.h"

#include "Features/PCAGrid.h"
#include <boost/python.hpp>


using namespace Eigen;

class PCAGrid_py : public PCAGrid
{
public:
    PCAGrid_py( Mat3<float>::type meanP, Mat3<float>::type evals, MapMat33Xf evects,
                float gridLength, int cellSide )
        :   PCAGrid(meanP, evals, evects, gridLength, cellSide)
    {}

    void ComputeAll( PyObject* R_py, PyObject* P_py,
                     PyObject* gridP_py, PyObject* gridEvals_py,
                     PyObject* gridEvects_py, PyObject* valid_py );
};



boost::shared_ptr<PCAGrid_py> PCAGrid_py_constructor(
        PyObject* meanP_py, PyObject* evals_py, PyObject* evects_py,
        float gridLength, int cellSide )
{
    Mat3<float>::type meanP = numpy_to_eigen<float, Dynamic, 3>( meanP_py, "meanP", NPY_FLOAT );
    Mat3<float>::type evals = numpy_to_eigen<float, Dynamic, 3>( evals_py, "evals", NPY_FLOAT, meanP.rows() );
    MapMat33Xf evects = numpy_to_eigen<float, Dynamic, 9>( evects_py, "evects", NPY_FLOAT, meanP.rows() );

    return boost::shared_ptr<PCAGrid_py>(
        new PCAGrid_py( meanP, evals, evects, gridLength, cellSide ) );
}



void PCAGrid_py::ComputeAll( PyObject* R_py, PyObject* P_py,
                             PyObject* gridP_py, PyObject* gridEvals_py,
                             PyObject* gridEvects_py, PyObject* valid_py )
{
    int n = PyArray_DIM((PyArrayObject*)R_py, 0);
    checkNumpyArray( R_py, "R", NPY_FLOAT, n, 3, 3 );
    Mat3<float>::type P = numpy_to_eigen<float, Dynamic, 3>( P_py, "P", NPY_FLOAT, n );

    checkNumpyArray( gridP_py, "gridP", NPY_FLOAT, n, nCells, 3 );
    checkNumpyArray( gridEvals_py, "gridEvals", NPY_FLOAT, n, nCells, 3 );
    checkNumpyArray( gridEvects_py, "gridEvects", NPY_FLOAT, n, nCells, 3, 3 );
    checkNumpyArray( valid_py, "valid", NPY_UBYTE, n, nCells );

    //not parallelised- would need to copy/re-build kdtree
    for( int i=0 ; i<n ; i++ )
    {
        Map<Matrix3f> R( (float*)PyArray_GETPTR3((PyArrayObject*)R_py, i, 0, 0), 3,3 );
        Vector3f point = P.row(i);

        float* gridP_i = (float*)PyArray_GETPTR3((PyArrayObject*)gridP_py, i, 0, 0);
        float* gridEvals_i = (float*)PyArray_GETPTR3((PyArrayObject*)gridEvals_py, i, 0, 0);
        float* gridEvects_i = (float*)PyArray_GETPTR4((PyArrayObject*)gridEvects_py, i, 0, 0, 0);
        unsigned char* valid_i = (unsigned char*)PyArray_GETPTR2((PyArrayObject*)valid_py, i, 0);

        PCAGridElement data( Mat3<float>::type( gridP_i, nCells, 3 ),
                             Mat3<float>::type( gridEvals_i, nCells, 3 ),
                             MapMat33Xf( gridEvects_i, nCells, 9 ),
                             Vect<unsigned char>::type( valid_i, nCells ) );

        compute( R, point, data );
    }
}



void export_PCAGrid()
{
    using namespace boost::python;
    class_<PCAGrid_py, boost::shared_ptr<PCAGrid_py> >("PCAGrid", no_init)
            .def( "__init__", boost::python::make_constructor( &PCAGrid_py_constructor ) )
            .def("ComputeAll", &PCAGrid_py::ComputeAll);
}
