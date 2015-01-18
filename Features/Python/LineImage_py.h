/*! LineImage_py.h
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

#ifndef LINE_IMAGE_PY
#define LINE_IMAGE_PY

#include <Python.h>
#include <boost/shared_ptr.hpp>
//#include <boost/python/overloads.hpp>

#include "Common/ArrayTypes.h"
#include "Features/LineImage.h"

/*!
Compute line images (python interface).
*/
class ComputeLineImage_py : public ComputeLineImage
{
public:
    ComputeLineImage_py( LineImageParams& params, Mat3<double>::type& P,
            Vect<unsigned char>::type& id, Vect<double>::type& D, Vect<double>::type& w,
            PCAResults& pcaResults, VelodyneDb& db, VeloRangeImage& image, int nThreads=1 );
    //! Set a boolean mask for all points. False items will not be considered for surface intercepts (eg. clutter around an object).
    void setObjectMask_py( PyObject* mask );
    //! Compute line image (single or multiple)
    void compute_py( PyObject* R_py, PyObject* P_py, PyObject* values_py, PyObject* status_py );

private:
    int nThreads_;
};


boost::shared_ptr<ComputeLineImage_py> ComputeLineImage_py_constructor(
        PyObject* params_py, PyObject* P_py, PyObject* id_py,
        PyObject* D_py, PyObject* w_py, PyObject* pcaResults_py,
        PyObject* db_pyobj, VeloRangeImage& image, int nThreads=1 );

//BOOST_PYTHON_FUNCTION_OVERLOADS(ComputeLineImage_constructor_overloads, ComputeLineImage_py_constructor, 8, 9)


//! Construct a LineImageParams object from a python radial_feature.line_image.lineImageParams() class instance
struct LineImageParams_py : LineImageParams
{
    LineImageParams_py( PyObject* params );
};


#endif //LINE_IMAGE_PY
