/*! LineDistrib_py.h
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

#ifndef LINE_DISTRIB_PY
#define LINE_DISTRIB_PY

#include "../export.h"
#include <Python.h>
#include <numpy/arrayobject.h>
#include <boost/python.hpp>
#include <boost/shared_ptr.hpp>

#include "Common/ArrayTypes.h"
#include "Features/LineDistrib.h"

class LASERLIB_FEATURES_EXPORT LineDistrib_py : public LineDistrib
{
public:
    LineDistrib_py( LineImageParams& params, float binLength );
    LineDistrib_py( const LineDistrib_py& );

    boost::shared_ptr<LineDistrib_py> Copy();
    void Set_py( PyObject* values, PyObject* status );

    //copy across data to numpy arrays
    PyObject* GetBins();
    PyObject* GetOdds();

    //pickling functions
    PyObject* __getinitargs__();
    PyObject* __getstate__();
    void __setstate__(PyObject* state);
    // for pickling init
    PyObject* params_py;
    float binLength_py;
};


LASERLIB_FEATURES_EXPORT boost::shared_ptr<LineDistrib_py> LineDistrib_py_constructor(
        PyObject* params_py, float binLength );
LASERLIB_FEATURES_EXTERN template LineDistrib_py const volatile * LASERLIB_FEATURES_IMPORT boost::get_pointer(LineDistrib_py const volatile *);


#endif //LINE_DISTRIB_PY
