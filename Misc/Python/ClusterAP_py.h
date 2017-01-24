/*! ClusterAP_py.h
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
 * \date       10-07-2013
*/

#ifndef CLUSTER_AP_PY
#define CLUSTER_AP_PY

#include "../export.h"
#include <Python.h>
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <numpy/arrayobject.h>
#include <boost/shared_ptr.hpp>
#include "Misc/ClusterAP.h"


class LASERLIB_MISC_EXPORT ClusterAP_py : public ClusterAP
{
public:
    ClusterAP_py( MapMatXf& sim_py, bool showProgress=true, float damping=0.9,
            int minIter=15, int convergeIter=20, int maxIter=200, int nThreads=-1 );

    //! Get assignments of points to exemplars
    PyObject* getAssignments_py();
    PyObject* getAssignmentScores_py();

    PyObject* getAvailabilities_py();
    PyObject* getResponsibilities_py();
};

LASERLIB_MISC_EXPORT boost::shared_ptr<ClusterAP_py> ClusterAP_py_constructor( PyObject* sim_py, bool showProgress, float damping,
                                                          int minIter, int convergeIter, int maxIter, int nThreads );


#endif //CLUSTER_AP_PY
