/*! Subsample_py.h
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
 * \date       31-05-2011
*/

#ifndef SUBSAMPLE_PY_HEADER_GUARD
#define SUBSAMPLE_PY_HEADER_GUARD

#include "../export.h"
#include <Python.h>
#include <numpy/arrayobject.h>
#include <boost/python.hpp>

#include "DataStore/Subsample.h"
#include "Selector_py.h"


LASERLIB_DATASTORE_EXPORT void LocalMax_py( Selector& sel, int nTotal, PyObject* items, PyObject* val, PyObject* maxBy );

LASERLIB_DATASTORE_EXPORT PyObject* SubSampleEvenly_py( Selector& sel, int nTotal );

LASERLIB_DATASTORE_EXPORT PyObject* SubSampleKeysEvenly_py( Selector& sel, int nTotal, PyObject* keys_py );

LASERLIB_DATASTORE_EXPORT PyObject* SubsampleBySurfNorm_py( Selector& sel, int nTotal, PyObject* keys_py, PyObject* sn, double thresh);


#endif //SUBSAMPLE_PY_HEADER_GUARD
