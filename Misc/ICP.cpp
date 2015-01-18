/*! ICP.cpp
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
 * \date       05-05-2011
*/

#include <iostream>
#include "ICP.h"
#include <Eigen/Geometry>
using namespace Eigen;


void ICP_PointPlane_2D( Mat3<double>::type& targetP, Mat3<double>::type& n, Mat3<double>::type& templateP,
                       TransformMatrix& R, double& errSqr, bool& isSingular )
{
    Mat3<double>::type& p = templateP;
    Mat3<double>::type& q = targetP;

    // make the 'covariance' matrix C
    Matrix<double, Dynamic, 3> c(p.rows(), 3);
    Matrix4d C = Matrix4d::Zero();
    for( int i=0 ; i<c.rows() ; i++ )
    {
        c.row(i) = p.row(i).cross( n.row(i) );
        C.block(1,1,3,3) += n.row(i).transpose() * n.row(i);
    }
    C(0,0) = c.col(2).array().square().sum();
    C.block(0,1,1,3) = (c.array() * n.array()).colwise().sum();
    C.block(1,0,3,1) = C.block(0,1,1,3).transpose();

    //the 'b' term in Cx = b
    Vector4d b = Vector4d::Zero();
    ArrayXd relDotN = ((p-q).array() * n.array()).rowwise().sum();
    b(0) = -( c.col(2).array() * relDotN ).sum();
    RowVector3d b_temp = ( n.array() * relDotN.replicate(1,3) ).colwise().sum();
    b.tail(3) = -b_temp.transpose();

    //solve for x
    Vector4d x = Vector4d::Zero();
    //if( abs(C.determinant()) < 0.001 )
    //{
    //    std::cout << "det(C)=" << C.determinant() << std::endl;
    //    isSingular = true;
    //}
    //else
    //{
    isSingular = false;
    //solve Cx = b
    //x = (C.adjoint() * C).llt().solve( C.adjoint()*b).transpose();
    x = (C).llt().solve(b).transpose();

    //x: theta, x, y, z
    //convert x into a rotation matrix.
    R = Matrix4d::Identity();
    R.block(0,0,3,3) = AngleAxisd( x(0), Vector3d::UnitZ() ).matrix();
    R.block(0,3,3,1) = x.tail(3);

    Matrix<double, Dynamic, 3> templateTrans = ( R.block(0,0,3,3) * templateP.transpose() + R.block(0,3,3,1).replicate( 1, templateP.rows() ) ).transpose();
    errSqr = ( (templateTrans - targetP).array() * n.array() ).rowwise().sum().square().sum();
}




void ICP_PointPlane_3D( Mat3<double>::type& targetP, Mat3<double>::type& n, Mat3<double>::type& templateP,
                       TransformMatrix& R, double& errSqr, bool& isSingular )
{
    Mat3<double>::type& p = templateP;
    Mat3<double>::type& q = targetP;

    // make the 'covariance' matrix C
    Matrix<double,6,6> C;
    C.setZero();
    //the 'b' term in Cx = b
    Matrix<double,6,1> b;
    b.setZero();
    for( int i=0 ; i<p.rows() ; i++ )
    {
        Vector3d ni = n.row(i).transpose();
        Vector3d ci = p.row(i).transpose().cross( ni );
        C.block(0,0,3,3) += ci * ci.transpose();
        C.block(3,0,3,3) += ni * ci.transpose();
        C.block(0,3,3,3) += ci * ni.transpose();
        C.block(3,3,3,3) += ni * ni.transpose();

        Vector3d rel = p.row(i) - q.row(i);
        double relDotN = rel.dot(ni);
        b.block(0,0,3,1) -= ci*relDotN;
        b.block(3,0,3,1) -= ni*relDotN;
    }

    //solve for x
    Matrix<double,6,1> x;
    isSingular = false;
    //solve Cx = b
    x = C.llt().solve(b).transpose();

    //x: theta, x, y, z
    //convert x into a rotation matrix.
    //NOTE- this 'angles to matrix' conversion is the same as in python's alastair.icp.icp_fns.rotxyz(),
    // not the same as pyception.rotations.xyzrpy_to_matrix()
    R = Matrix4d::Identity();
    Quaterniond rot = AngleAxisd( x(0), Vector3d::UnitX() ) * AngleAxisd( x(1), Vector3d::UnitY() ) * AngleAxisd( x(2), Vector3d::UnitZ() );
    R.block(0,0,3,3) = rot.matrix();
    R.block(0,3,3,1) = x.tail(3);

    Matrix<double, Dynamic, 3> templateTrans = ( R.block(0,0,3,3) * templateP.transpose() + R.block(0,3,3,1).replicate( 1, templateP.rows() ) ).transpose();
    errSqr = ( (templateTrans - targetP).array() * n.array() ).rowwise().sum().square().sum();
}
