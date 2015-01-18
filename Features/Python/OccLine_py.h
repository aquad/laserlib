/*! OccImage_py.h
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
 * \date       05-12-2012
*/

#ifndef OCC_IMAGE_PY
#define OCC_IMAGE_PY

#include <Python.h>
#include <boost/python.hpp>
#include <boost/shared_ptr.hpp>

#include "Common/ArrayTypes.h"
#include "Features/OccLine.h"


//! Compute occupancy line (python interface).
class OccLine_py : public OccLine
{
public:
    OccLine_py( OccLineParams& params, Mat3<double>::type& P, Vect<unsigned char>::type& id,
            Vect<double>::type& D, Vect<double>::type& w,
            PCAResults& pcaResults, VelodyneDb& db, VeloRangeImage& image );

    //! Set a boolean mask for all points. False items will not be considered for surface intercepts (eg. clutter around an object).
    void setObjectMask_py( PyObject* mask );

    //! Compute on one or multiple lines
    PyObject* compute_py( PyObject* start_py, PyObject* end_py );

    // internal routines for testing / visualisation
    PyObject* lineTrace_py( PyObject* bounds );
    PyObject* getPointsOnLine_py( PyObject* bounds_py );
    PyObject* orderPointsOnLine_py( PyObject* start_py, PyObject* end_py,
            PyObject* pointsOnLine_py );
    boost::python::tuple traverseLineForIntercept_py( PyObject* lineT_py,
            PyObject* linePid_py, PyObject* start_py, PyObject* end_py );
};


boost::shared_ptr<OccLine_py> OccLine_py_constructor(
        PyObject* params_py, PyObject* P_py, PyObject* id_py,
        PyObject* D_py, PyObject* w_py, PyObject* pcaResults_py,
        PyObject* db_pyobj, VeloRangeImage& image );


//! Construct a OccLineParams object from a python radial_feature.line_image.OccLineParams() class instance
struct OccLineParams_py : OccLineParams
{
    OccLineParams_py( PyObject* params );
};


#endif //OCC_LINE_PY
