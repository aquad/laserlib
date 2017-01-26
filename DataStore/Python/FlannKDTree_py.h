/*! FlannKDTree_py.h
 * A wrapper for Flann's KDTree for selecting regions of points.
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

#ifndef FLANN_KD_TREE_PY
#define FLANN_KD_TREE_PY

#include "../export.h"
#include "DataStore/FlannKDTree.h"
#include "Selector_py.h"

template <typename T>
class FlannKDTree_py : public FlannKDTree<T>, public Selector_py
{
public:
    FlannKDTree_py( typename Mat3<T>::type& P, T rad )
        :   FlannKDTree<T>( P, rad )
    {}

    FlannKDTree_py( typename Mat3<T>::type& P, T rad, std::string& filename )
        :   FlannKDTree<T>( P, rad, filename )
    {}

    PyObject* Select3D_py( Eigen::Matrix<T,3,1>& centreP )
    {
        std::vector<int>& neighs = this->Select3D(centreP);
        npy_intp dims[1] = {static_cast<npy_intp>(neighs.size())};
        PyArrayObject* neighs_py = (PyArrayObject*) PyArray_SimpleNew(1, dims, NPY_INT );
        memcpy( PyArray_DATA(neighs_py), &(neighs[0]), sizeof(int)*neighs.size() );
        return PyArray_Return(neighs_py);
    }
};

LASERLIB_DATASTORE_EXTERN template class LASERLIB_DATASTORE_IMPORT FlannKDTree_py<double>;
LASERLIB_DATASTORE_EXTERN template class LASERLIB_DATASTORE_IMPORT FlannKDTree_py<float>;
LASERLIB_DATASTORE_EXTERN template FlannKDTree_py<double> const volatile * LASERLIB_DATASTORE_IMPORT boost::get_pointer(FlannKDTree_py<double> const volatile *);
LASERLIB_DATASTORE_EXTERN template FlannKDTree_py<float> const volatile * LASERLIB_DATASTORE_IMPORT boost::get_pointer(FlannKDTree_py<float> const volatile *);

#endif //FLANN_KD_TREE_PY
