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

#include <Python.h>
#include <boost/shared_ptr.hpp>

#include "DataStore/GraphTraverser.h"
#include "Common/ArrayTypes.h"
#include "Selector_py.h"


void Get4Neighs_py( PyObject* neighs, PyObject* graph, PyObject* P, double rad );
void Get4NeighsClosest_py( PyObject* neighs, PyObject* graph );
void Get4NeighsValid_py( PyObject* neighs, PyObject* graph, PyObject* P, PyObject* valid, double rad );

void ConvexSegment_py( PyObject* segs, PyObject* convex, PyObject* graph, PyObject* surfNorms, double eta3 );



class GraphSphereSelector_py : public GraphSphereSelector, public Selector_py
{
public:
    GraphSphereSelector_py( Graph& graph, Mat3<double>::type& P, double rad )
        :   GraphSphereSelector(graph, P, rad) {}
};

boost::shared_ptr<GraphSphereSelector_py> GraphSphereSelector_py_constructor( PyObject* graph_py, PyObject* P_py, double rad );



class PreFourNeighSelector_py : public PreFourNeighSelector, public Selector_py
{
public:
    PreFourNeighSelector_py( Graph& neighs_all )
        :   PreFourNeighSelector( neighs_all ) {}
};

boost::shared_ptr<PreFourNeighSelector_py> PreFourNeighSelector_py_constructor( PyObject* neighs_all_py );



#endif //GRAPH_TRAVERSER_PY
