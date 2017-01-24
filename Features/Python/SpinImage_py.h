/*! SpinImage_py.h
 *
 * Copyright (C) 2010 Alastair Quadros.
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
 * \date       10-11-2010
*/

#ifndef SPIN_IMAGE_PY
#define SPIN_IMAGE_PY

#include "../export.h"
#include <Python.h>
#include "Common/ArrayTypes.h"
#include "DataStore/Selector.h"
#include "LaserLibConfig.h"

#ifdef LaserLib_USE_FLANN
#include "DataStore/Python/FlannKDTree_py.h"
#endif


//! Compute spin images on a set of points.
class LASERLIB_FEATURES_EXPORT SpinImageBatch_py
{
public:
    SpinImageBatch_py( PyObject* P, PyObject* sn,
                       double imageLength, int cellSide,
                       double supportAngle, int nThreads=1 );

    //! Compute about points (from construction) referred to by 'keys'
    void ComputeAllKeys( Selector& sel, PyObject* keys_py, PyObject* images_py );

    #ifdef LaserLib_USE_FLANN
    //! Compute about specified points and surf norms
    void ComputeAll( FlannKDTree_py<double>& sel, PyObject* centreP_py, PyObject* centreSn_py, PyObject* images_py );
    #endif

private:
    Mat3<double>::type P_;
    Mat3<double>::type sn_;
    double imageLength_;
    int cellSide_;
    double supportAngle_;
    int nThreads_;
};

#endif //SPIN_IMAGE_PY
