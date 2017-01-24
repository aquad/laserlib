/*! GraphTraverser_py.h
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

#ifndef GRAPH_TRAVERSER_PY
#define GRAPH_TRAVERSER_PY

#include "../export.h"
#include <Python.h>
#include <boost/shared_ptr.hpp>

#include "DataStore/GraphTraverser.h"
#include "Common/ArrayTypes.h"
#include "Selector_py.h"


LASERLIB_DATASTORE_EXPORT void Get4Neighs_py( PyObject* neighs, PyObject* graph, PyObject* P, double rad );
LASERLIB_DATASTORE_EXPORT void Get4NeighsClosest_py( PyObject* neighs, PyObject* graph );
LASERLIB_DATASTORE_EXPORT void Get4NeighsValid_py( PyObject* neighs, PyObject* graph, PyObject* P, PyObject* valid, double rad );

LASERLIB_DATASTORE_EXPORT void ConvexSegment_py( PyObject* segs, PyObject* convex, PyObject* graph, PyObject* surfNorms, double eta3 );



class LASERLIB_DATASTORE_EXPORT GraphSphereSelector_py : public GraphSphereSelector, public Selector_py
{
public:
    GraphSphereSelector_py( Graph& graph, Mat3<double>::type& P, double rad )
        :   GraphSphereSelector(graph, P, rad) {}
};

LASERLIB_DATASTORE_EXPORT boost::shared_ptr<GraphSphereSelector_py> GraphSphereSelector_py_constructor( PyObject* graph_py, PyObject* P_py, double rad );
LASERLIB_DATASTORE_EXTERN template GraphSphereSelector_py const volatile * LASERLIB_DATASTORE_IMPORT boost::get_pointer(GraphSphereSelector_py const volatile *);


class LASERLIB_DATASTORE_EXPORT PreFourNeighSelector_py : public PreFourNeighSelector, public Selector_py
{
public:
    PreFourNeighSelector_py( Graph& neighs_all )
        :   PreFourNeighSelector( neighs_all ) {}
};

LASERLIB_DATASTORE_EXPORT boost::shared_ptr<PreFourNeighSelector_py> PreFourNeighSelector_py_constructor( PyObject* neighs_all_py );
LASERLIB_DATASTORE_EXTERN template PreFourNeighSelector_py const volatile * LASERLIB_DATASTORE_IMPORT boost::get_pointer(PreFourNeighSelector_py const volatile *);



#endif //GRAPH_TRAVERSER_PY
