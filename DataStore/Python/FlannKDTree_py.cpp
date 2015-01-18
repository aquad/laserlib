/*! FlannKDTree_py.cpp
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
 * \date       08-06-2011
*/

#include <Python.h>
#define PY_ARRAY_UNIQUE_SYMBOL _LaserPy_ARRAY_API
#define NO_IMPORT_ARRAY
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <numpy/arrayobject.h>
#include <boost/python.hpp>

#include "Common/ArrayTypes.h"
#include "FlannKDTree_py.h"


using namespace boost::python;
using namespace Eigen;

// TODO: fix naming in doc
template <typename T>
void create_FlannKDTree_py( const char* name )
{
    class_< FlannKDTree_py<T>, 
            boost::shared_ptr< FlannKDTree_py<T> >, 
            bases<Selector> >(
            name,
            "FlannKDTree(P, rad)\n\n"

            "Select arbitrary 3D spherical regions on unstructured point clouds\n\n"
            "Parameters\n"
            "----------\n"
            "P : ndarray (n,3) float64\n"
            "    3D point cloud\n"
            "rad : float\n"
            "    Radius of spherical region to select\n",
            init< typename Mat3<T>::type&, T >() )

            .def(init< typename Mat3<T>::type&, T, std::string& >())

            .def("SelectRegion", &FlannKDTree_py<T>::SelectRegion_py,
                "SelectRegion(i)\n\n"
                "Select spherical region about existing point.\n\n"
                "Parameters\n"
                "----------\n"
                "i : integer\n"
                "    Point id at the centre of the sphere to select\n\n"
                "Returns\n"
                "-------\n"
                "ids : ndarray (n,) int32\n"
                "    Point ids within selection\n")

            .def("Select3D", &FlannKDTree_py<T>::Select3D_py,
                "Select3D(p)\n\n"
                "Select spherical region about arbitrary 3D point.\n\n"
                "Parameters\n"
                "----------\n"
                "p : ndarray (3,) float64\n"
                "    3D point at the centre of the sphere to select\n\n"
                "Returns\n"
                "-------\n"
                "ids : ndarray (n,) int32\n"
                "    Point ids within selection\n")

            .def("setRadius", &FlannKDTree_py<T>::setRadius,
                "setRadius(rad)");
}


void export_FlannKDTree()
{
    create_FlannKDTree_py<double>("FlannKDTree");
    create_FlannKDTree_py<float>("FlannKDTreeF");
}

