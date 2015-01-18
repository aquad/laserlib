/*! BearingGraph_py.h
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
 * \date       18-05-2011
*/

#ifndef BEARING_GRAPH_PY
#define BEARING_GRAPH_PY

#include <Python.h>
#include <boost/scoped_ptr.hpp>
#include <boost/python.hpp>

class BearingGraphBuilder;


class BearingGraphBuilder_py
{
public:
    BearingGraphBuilder_py( PyObject* db_pyobj, unsigned int maxNumPoints=300000, float wThresh=0.0349 );
    PyObject* BuildGraph( PyObject* id, PyObject* w );
    PyObject* CleanGraph( PyObject* P, PyObject* D, float maxLength=5.0, float relThresh=3.0, float convThresh=0.5 );
    PyObject* CleanGraphFast( PyObject* D, int maxLength=5*500, float relThresh=3.0, int convThresh=0.5*500 );
    PyObject* __getinitargs__();

private:
    boost::scoped_ptr< BearingGraphBuilder > bearGraph;

    //for pickling
    PyObject* db_pyobj_;
    unsigned int maxNumPoints_;
    float wThresh_;
};

// macro for default arguments
BOOST_PYTHON_MEMBER_FUNCTION_OVERLOADS(CleanGraph_overloads, CleanGraph, 2, 5)
BOOST_PYTHON_MEMBER_FUNCTION_OVERLOADS(CleanGraphFast_overloads, CleanGraphFast, 1, 4)


void CalcSurfNorms_py( PyObject* neighs, PyObject* P, PyObject* sn, PyObject* valid );

void BlurSurfNorms_py( PyObject* neighs_py, PyObject* P_py, PyObject* sn_py,
                      PyObject* valid_py, PyObject* sn_blurred_py, double sd=1.0 );

BOOST_PYTHON_FUNCTION_OVERLOADS(BlurSurfNorms_overloads, BlurSurfNorms_py, 5, 6)


#endif //BEARING_GRAPH_PY
