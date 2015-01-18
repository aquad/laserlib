/*! TestConversion.h
 * Test the ability to check and convert numpy arrays to eigen/std types.
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

#ifndef TEST_CONVERSION_PY
#define TEST_CONVERSION_PY

#include <Python.h>
#include <boost/python/overloads.hpp>
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <numpy/ndarraytypes.h>
#include <string>


void checkNumpyArray_py( PyObject* arr, const std::string& name,
                        int type_num, int shape0, int shape1=0, int shape2=0,
                        int shape3=0 );

// macro for default arguments
BOOST_PYTHON_FUNCTION_OVERLOADS(checkNumpyArray_py_overloads, checkNumpyArray_py, 4, 7)


PyObject* numpy_to_std_vector_py(PyObject* arr, const std::string& name, int type_num);

void test_numpy_float_to_eigen( PyObject* arr, const std::string& name, int np_rows, int np_cols );

#endif //TEST_CONVERSION_PY
