/*! geocon_py.h
 * Geometric Consistency
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
 * \date       15-07-2011
*/

#ifndef GEOMETRIC_CONSISTENCY_PY
#define GEOMETRIC_CONSISTENCY_PY

#include <Python.h>
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <numpy/arrayobject.h>
#include "Misc/GeometricConsistency.h"
#include "Misc/bron-kerbosch.h"
#include <algorithm>

//old functions----
PyObject* FindConsistentSet_py( PyObject* seeds_py,
                               PyObject* testIds_py, PyObject* testP_py, PyObject* testVect_py,
                               PyObject* trainIds_py, PyObject* trainP_py, PyObject* trainVect_py,
                               PyObject* rmse_py, float distThresh=0.2, float angleThresh=0.5, float rmseWeight=3.0);

PyObject* FindMaxClique_py( PyObject* matches_py,
                 PyObject* testP_py, PyObject* testVect_py,
                 PyObject* trainP_py, PyObject* trainVect_py,
                 float distThresh, float minDist, float angleThresh, int minCliqueSize, int maxEdges );


void FindComponents_py( PyObject* matches_py,
                 PyObject* testP_py, PyObject* testVect_py,
                 PyObject* trainP_py, PyObject* trainVect_py,
                 float distThresh, float angleThresh, PyObject* components_py );
//------------------



typedef bron_kerbosch<CorrGraph> bron_kerbosch_corrgraph;

boost::shared_ptr< CorrGraph > BuildCorrGraph_py( PyObject* matches_py,
                     PyObject* testP, PyObject* testVect,
                     PyObject* trainP, PyObject* trainVect,
                     float distThresh, float angleThresh, float minDist );

boost::shared_ptr< CorrGraph > BuildCorrGraphZAligned_py( PyObject* matches_py,
                     PyObject* testP, PyObject* testVect,
                     PyObject* trainP, PyObject* trainVect,
                     float distThresh, float angleThresh, float minDist );



class bron_kerbosch_py : public bron_kerbosch_corrgraph
{
public:
    bron_kerbosch_py( CorrGraph& g ) : bron_kerbosch_corrgraph(g)
    {}

    //! Find a clique of given size
    PyObject* find_clique_py(int min)
    {
        Clique cliq;
        find_clique( cliq, (std::size_t)min );

        npy_intp dims[1] = {static_cast<npy_intp>(cliq.size())};
        PyArrayObject *cliq_py = (PyArrayObject*)PyArray_SimpleNew(1, dims, NPY_INT);
        std::copy( cliq.begin(), cliq.end(), (int*)PyArray_DATA(cliq_py) );
        return PyArray_Return(cliq_py);
    }
};


int CorrGraphNumEdges( CorrGraph& g );


PyObject* PairsOneToOne_py( PyObject* matches_py, PyObject* fDist_py );

#endif //GEOMETRIC_CONSISTENCY_PY
