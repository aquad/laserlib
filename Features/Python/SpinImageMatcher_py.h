/*! SpinImageMatcher_py.h
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
 * \date       12-01-2013
*/

#ifndef SPIN_IMAGE_MATCHER_PY
#define SPIN_IMAGE_MATCHER_PY

#include <Python.h>
#include <numpy/arrayobject.h>
#include "Features/SpinImage.h"
#include "Features/SpinImageKnn.h"


/*! Compare one test spin image against a set of training, using the similarity metric.
test_py: (cellSide x cellSide,) float32 numpy array.
training_py: (n, cellSide x cellSide) float32 numpy array.
values_py: (n,) float32 numpy array (output).
lamb: lambda weighting parameter for metric SIMILARITY (see spin image paper).
*/
void MatchSpinSets_py( PyObject* test_py, PyObject* train_py, PyObject* values_py, float lamb, SpinMetric metric );


float SpinCorrelation_py( PyObject* Ppy, PyObject* Qpy);
float SpinCorrAtanh_py( PyObject* Ppy, PyObject* Qpy);
float SpinSimilarity_py( PyObject* Ppy, PyObject* Qpy, float lamb);

void SpinKnn_py( PyObject* test_py, PyObject* training_py,
                 PyObject* matches_py, PyObject* values_py, float lamb, SpinMetric metric );

#endif //SPIN_IMAGE_MATCHER_PY
