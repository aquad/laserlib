/*! Selector_py.cpp
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
 * \date       23-05-2011
*/

#define PY_ARRAY_UNIQUE_SYMBOL _LaserPy_ARRAY_API
#define NO_IMPORT_ARRAY
#include "LaserPy/numpy_to_eigen.h"
#include "Common/ArrayTypes.h"

#include "Selector_py.h"
#include <boost/python.hpp>



void SelectAllTest( Selector& sel, int nPoints );
void SelectAllTestKeys( Selector& sel, int nPoints, PyObject* keys_py );


void export_Selector()
{
    using namespace boost::python;
    class_<Selector, Selector_callback, boost::noncopyable>("Selector"); //pure virtual class

    def("SelectAllTest", &SelectAllTest);
    def("SelectAllTestKeys", &SelectAllTestKeys);
}



//----- testing / timing benchmarks ------

void SelectAllTest( Selector& sel, int nPoints )
{
    for( int i=0 ; i<nPoints ; i++ )
    {
        std::vector<int>& neigh = sel.SelectRegion(i);
    }
}


void SelectAllTestKeys( Selector& sel, int nPoints, PyObject* keys_py)
{
    Vect<int>::type keys = numpy_to_eigen<int, Eigen::Dynamic, 1>( keys_py, "keys", NPY_INT );
    for( int i=0 ; i<keys.size() ; i++ )
    {
        int id = keys(i);
        std::vector<int>& neigh = sel.SelectRegion(id);
    }
}

