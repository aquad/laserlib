/*! PCA_py.h
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
 * \date       24-11-2010
*/

#ifndef PCA_PY_HEADER_GUARD
#define PCA_PY_HEADER_GUARD

#include "../export.h"
#include <Python.h>
#include <boost/python.hpp>
#include "Common/ArrayTypes.h"
#include "Features/PCA.h"

class Selector;
class ImageSphereSelector_py;


class LASERLIB_FEATURES_EXPORT PCA_py
{
public:
    PCA_py( PyObject* P, int nThreads=1 );

    void ComputeAll( Selector& selector, PyObject* keys, PyObject* evals_py, PyObject* evects_py, PyObject* meanP_py );

    void ComputeAllVariableSize( ImageSphereSelector_py& _sel, PyObject* keys_py, PyObject* rad_py,
                                PyObject* evals_py, PyObject* evects_py, PyObject* meanP_py );
private:
    Mat3<double>::type P_;
    int nThreads_;
};


LASERLIB_FEATURES_EXPORT void minRadiusSelection_py( PyObject* graph_py, PyObject* P_py, PyObject* rad_py, PyObject* valid_py );


//default args
BOOST_PYTHON_FUNCTION_OVERLOADS(surfNormPCA_ol_f, surfNormPCA<float>, 3, 5)

BOOST_PYTHON_FUNCTION_OVERLOADS(surfNormPCA_ol_d, surfNormPCA<double>, 3, 5)

#endif //PCA_PY_HEADER_GUARD
