/*! Rotations_py.cpp
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
 * \date       28-07-2011
*/

#include <Python.h>
#define PY_ARRAY_UNIQUE_SYMBOL _LaserPy_ARRAY_API
#define NO_IMPORT_ARRAY
#include "numpy_to_eigen.h"
#include <Eigen/Geometry>

#include <boost/python.hpp>

#include "Common/ArrayTypes.h"
#include "Rotations_py.h"


void export_Rotations()
{
    using namespace boost::python;
    def("transform_from_rpy", &transform_from_rpy,
        "transform_from_rpy(pos, rpy, p_in, p_out)\n\n"
        "Transform points, each with a frame defined by an offset and rotation. "
        "Transform from this frame.\n\n"
        "Parameters\n"
        "----------\n"
        "pos : ndarray (n,3) float64\n"
        "   Positional offset of each frame.\n"
        "rpy : ndarray (n,3) float64\n"
        "   Roll, pitch, yaw of each frame.\n"
        "p_in : ndarray (n,3) float64\n"
        "   Input points to transform\n"
        "p_out : ndarray (n,3) float64\n"
        "   (output) transformed points.");

    def("transform_to_rpy", &transform_to_rpy,
        "transform_to_rpy(pos, rpy, p_in, p_out)\n\n"
        "Transform points, each with a frame defined by an offset and rotation. "
        "Transform into this frame.\n\n"
        "Parameters\n"
        "----------\n"
        "pos : ndarray (n,3) float64\n"
        "   Positional offset of each frame.\n"
        "rpy : ndarray (n,3) float64\n"
        "   Roll, pitch, yaw of each frame.\n"
        "p_in : ndarray (n,3) float64\n"
        "   Input points to transform\n"
        "p_out : ndarray (n,3) float64\n"
        "   (output) transformed points.");
}


using namespace Eigen;

// roll pitch yaw: Euler angles. Points are first rotated about Z by yaw, then about Y by pitch, then about X by roll.
// the transform is then: outPoints = roll * pitch * yaw * points.
void transform_from_rpy( PyObject* pos_py, PyObject* rpy_py,
        PyObject* p_in_py, PyObject* p_out_py )
{
    Mat3<double>::type pos = numpy_to_eigen<double, Dynamic, 3>( pos_py, "pos", NPY_DOUBLE );
    int n = pos.rows();
    Mat3<double>::type rpy = numpy_to_eigen<double, Dynamic, 3>( rpy_py, "rpy", NPY_DOUBLE, n );
    Mat3<double>::type p_in = numpy_to_eigen<double, Dynamic, 3>( p_in_py, "p_in", NPY_DOUBLE, n );
    Mat3<double>::type p_out = numpy_to_eigen<double, Dynamic, 3>( p_out_py, "p_out", NPY_DOUBLE, n );

    Matrix3d m_rotation;
    for( int i=0 ; i<rpy.rows() ; i++ )
    {
        double sr = sin( rpy(i,0) );
        double cr = cos( rpy(i,0) );
        double sp = sin( rpy(i,1) );
        double cp = cos( rpy(i,1) );
        double sy = sin( rpy(i,2) );
        double cy = cos( rpy(i,2) );

        m_rotation << cp*cy, -cr*sy+sr*sp*cy,  sr*sy+cr*sp*cy,
                      cp*sy,  cr*cy+sr*sp*sy, -sr*cy+cr*sp*sy,
                        -sp,           sr*cp,           cr*cp  ;
        p_out.row(i) = m_rotation * p_in.row(i).transpose();
    }
    p_out += pos;
}



void transform_to_rpy( PyObject* pos_py, PyObject* rpy_py,
        PyObject* p_in_py, PyObject* p_out_py )
{
    Mat3<double>::type pos = numpy_to_eigen<double, Dynamic, 3>( pos_py, "pos", NPY_DOUBLE );
    int n = pos.rows();
    Mat3<double>::type rpy = numpy_to_eigen<double, Dynamic, 3>( rpy_py, "rpy", NPY_DOUBLE, n );
    Mat3<double>::type p_in = numpy_to_eigen<double, Dynamic, 3>( p_in_py, "p_in", NPY_DOUBLE, n );
    Mat3<double>::type p_out = numpy_to_eigen<double, Dynamic, 3>( p_out_py, "p_out", NPY_DOUBLE, n );

    Matrix3d m_rotation;
    for( int i=0 ; i<rpy.rows() ; i++ )
    {
        double sr = sin( rpy(i,0) );
        double cr = cos( rpy(i,0) );
        double sp = sin( rpy(i,1) );
        double cp = cos( rpy(i,1) );
        double sy = sin( rpy(i,2) );
        double cy = cos( rpy(i,2) );

        m_rotation << cp*cy, -cr*sy+sr*sp*cy,  sr*sy+cr*sp*cy,
                      cp*sy,  cr*cy+sr*sp*sy, -sr*cy+cr*sp*sy,
                        -sp,           sr*cp,           cr*cp  ;

        Translation3d m_translation( pos.row(i) );
        p_out.row(i) = m_rotation.transpose() * ( m_translation.inverse() ) * p_in.row(i).transpose();
    }
}

