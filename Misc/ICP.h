/*! ICP.h
 * Iterative Closest Points
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

#ifndef ICP_HEADER_GUARD
#define ICP_HEADER_GUARD

#define EIGEN_DEFAULT_TO_ROW_MAJOR 1
#include <Eigen/Dense>
#include "Common/ArrayTypes.h"


typedef Eigen::Matrix<double,4,4,Eigen::RowMajor> TransformMatrix;

//@{
/*!
Solves point-to-plance ICP by linearizing the rotations and solving a least squares linear equation.
Presumes target and template points are already mean-centred, outputs the rotation matrix R.
Based on www.cs.princeton.edu/~smr/papers/icpstability.pdf

<ICP_PointPlane_2D>"()" optimizes xyz translation, rotation about z only.
<ICP_PointPlane_3D>"()" optimizes the full 6dof pose.

\param targetP - target points
\param n - surface normals of target points
\param templateP - template points (these are moved to align with the target points)
\param R - (output) rotation matrix
\param errSqr - (output) resulting ICP error
\param isSingular - (output) if this is true, the rotation matrix was not found. Currently not used- need to detect singular matrices somehow.
*/
void ICP_PointPlane_2D( Mat3<double>::type& targetP, Mat3<double>::type& n, Mat3<double>::type& templateP,
                       TransformMatrix& R, double& errSqr, bool& isSingular );

void ICP_PointPlane_3D( Mat3<double>::type& targetP, Mat3<double>::type& n, Mat3<double>::type& templateP,
                       TransformMatrix& R, double& errSqr, bool& isSingular );
//@}

#endif //ICP_HEADER_GUARD
