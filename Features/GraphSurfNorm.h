/*! GraphSurfNorm.h 
 * Calculate and smooth surface normals using a bearing graph.
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
 * \date       19-05-2011
*/

#ifndef GRAPH_SURF_NORM_HEADER_GUARD
#define GRAPH_SURF_NORM_HEADER_GUARD

#include "Common/ArrayTypes.h"

/*!
Calculate surface normals.

First generate a bearing graph (eg. DataStore/BearingGraph.h).
Use it to select 4 neighbours about each point. This produces 'neighs', where each row
contains the index numbers of 4 neighbouring points (left,right,up,down in the range image).
For each point, the neighbours form relative vectors.
Sets of 2 vectors are cross-producted and then averaged to get a surface normal.
\param neighs - neighbours, (n,4) matrix, see the output of <Get4Neighs>"()"
\param P - 3d points, (n,3) matrix
\param sn - (output) surface normals, (n,3) matrix
\param valid - (output) whether each point has a valid surface normal, (n,) matrix
*/
void CalcSurfNorms( Graph& neighs, Mat3<double>::type& P, Mat3<double>::type& sn, Vect<bool>::type& valid );


/*!
Blur surface normals.

Normals are blurred based on their neighbours 'neighs'.
Neighbouring surface normals are gaussian weighted by the distance
from the centre point, with the specified 'sd'.

\param neighs - neighbours, (n,4) matrix, see the output of <Get4Neighs>"()"
\param P - 3d points, (n,3) matrix
\param sn - surface normals, (n,3) matrix
\param valid - whether each point has a valid surface normal, (n,) matrix
\param sn_blurred - (output) blurred surface normals, (n,3) matrix
\param sd - (optional) standard deviation of gaussian, larger = far neighbours get weighted higher.
*/
void BlurSurfNorms( Graph& neighs, Mat3<double>::type& P, Mat3<double>::type& sn,
                   Vect<bool>::type& valid, Mat3<double>::type& sn_blurred, double sd=1.0 );
#endif //GRAPH_SURF_NORM_HEADER_GUARD
