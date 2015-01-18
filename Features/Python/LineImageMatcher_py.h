/*! LineImageMatcher_py.h
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

#ifndef LINE_IMAGE_MATCHER_PY
#define LINE_IMAGE_MATCHER_PY

#include <Python.h>
#include <boost/python/overloads.hpp>
#include "Features/LineImageMatcher.h"


void match_line_images( PyObject* params, int metric, PyObject* values1, PyObject* status1,
        PyObject* valuesSet, PyObject* statusSet,
        PyObject* rmse, PyObject* known );


void match_line_images_keys( PyObject* params, int metric, PyObject* values1, PyObject* status1,
        PyObject* valuesSet, PyObject* statusSet, PyObject* keys,
        PyObject* rmse, PyObject* known );


void match_line_image_sets( PyObject* params, int metric, PyObject* values1, PyObject* status1,
        PyObject* values2, PyObject* status2, PyObject* rmse, PyObject* known );


void match_line_images_all( PyObject* params_py, int metric, PyObject* values1_py, PyObject* status1_py,
        PyObject* values2_py, PyObject* status2_py, PyObject* rmse_py, PyObject* known_py, int nThreads=1 );


// macro for default arguments
BOOST_PYTHON_FUNCTION_OVERLOADS(match_line_images_all_overloads, match_line_images_all, 8, 9)


/*! Convert a python class to a ObjLineImagesAligned struct.
  Python object must have these attributes:
  alignVectors - surface normals or linear vectors, numpy array, shape=(n,3), dtype=float32.
  alignType - whether the vector is a surface normal (0) or linear vector (1). numpy array, shape=(n,), dtype=uint8.
  values - line image values, numpy array, shape=(n,nLines), dtype=float32.
  status - line image status, numpy array, shape=(n,nLines), dtype=uint8.
  */
ObjLineImagesAligned ObjLineImagesAligned_from_py( PyObject* data_py );


/*! Convert a python class to a ObjLineImages struct.
  Python object must have these attributes:
  values - line image values, numpy array, shape=(n,nLines), dtype=float32.
  status - line image status, numpy array, shape=(n,nLines), dtype=uint8.
  */
ObjLineImages ObjLineImages_from_py( PyObject* data_py );


//! Convert a list of ObjLineImagesAligned from python, see ObjLineImagesAligned_from_py"()".
void ObjLineImagesAlignedDataSet_from_py( PyObject* dataset_py, std::vector<ObjLineImagesAligned>& dataset );


//! Convert a list of ObjLineImages from python, see ObjLineImages_from_py"()".
void ObjLineImagesDataSet_from_py( PyObject* dataset_py, std::vector<ObjLineImages>& dataset );



#endif //LINE_IMAGE_MATCHER_PY
