/*! ICP_py.h
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
 * \date       11-05-2011
*/

#ifndef ICP_PY_HEADER_GUARD
#define ICP_PY_HEADER_GUARD

#include "../export.h"
#include <Python.h>
#include "Misc/ICP.h"

/*!
Solves point-to-plance ICP.
Wraps ICP_PointPlane_2D()

\param targetP - target points
\param n - surface normals of target points
\param templateP - template points (these are moved to align with the target points)

outputs a 3-tuple: (R, errSqr, isSingular)
\param R - (output) rotation matrix
\param errSqr - (output) resulting ICP error
\param isSingular - (output) if this is true, the rotation matrix was not found.
*/
LASERLIB_MISC_EXPORT PyObject* ICP_PointPlane_2D_py( PyObject* targetP, PyObject* n, PyObject* templateP );

LASERLIB_MISC_EXPORT PyObject* ICP_PointPlane_3D_py( PyObject* targetP, PyObject* n, PyObject* templateP );

#endif //ICP_PY_HEADER_GUARD
