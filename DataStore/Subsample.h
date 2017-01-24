/*! Subsample.h
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
 * \author  Alastair Quadros
 * \date    31-05-2011
*/

#ifndef SUB_SAMPLE_HEADER_GUARD
#define SUB_SAMPLE_HEADER_GUARD

#include "export.h"
#include <vector>
#include "Common/ArrayTypes.h"
#include "Selector.h"


/*! Evenly subsample by region selection.
The selector is used to select points about a centre point. These neighbours are
removed, the centre point added to the output sample, and the next non-removed centre point is selected.
\param sel- This selects neighbours, the radius / other criteria was defined in here.
\param nTotal- total number of points that was given to 'sel'.
\param sample- (output) subsampled points.
*/
LASERLIB_DATASTORE_EXPORT void SubSampleEven( Selector& sel, int nTotal, std::vector<int>& sample );



/*! Evenly subsample keypoints by region selection.
As in SubSampleEven, but 'keys' are iterated over as centre points.
*/
LASERLIB_DATASTORE_EXPORT void SubSampleKeysEvenly( Selector& sel, int nTotal, Vect<int>::type& keys, std::vector<int>& sample );



/*! Select locally maximum-valued points.

Find points that have a maximum value within radius, giving how much they are greater on average.
maxBy is filled with how much the resulting maximum was larger than its surrounds. it must be a
full sized array (ie, the number of points in the selector = nTotal = val.size()).
it will be 0 where there is no max.

\param items- keypoints to iterate through
\param val- the value to maximise, size (nTotal,)
\param maxBy- (output) how much each maximum was larger than its surrounds.
This is 0 where there is no maximum. Size (nTotal,)
*/
LASERLIB_DATASTORE_EXPORT void LocalMax( Selector& sel, int nTotal, Vect<int>::type& items, Vect<double>::type& val, Vect<double>::type maxBy );



/*! Remove points near each other with similar surface normals.
\param items- keypoints to be subsampled.
\param sn- surface normals, (nTotal,3) array.
\param thresh- if the dot product of two surface normals is above this, a point is removed (cos(thresh) is min angle difference).
\param sample- (output) subsampled points.
*/
LASERLIB_DATASTORE_EXPORT void SubsampleBySurfNorm( Selector& sel, int nTotal, Vect<int>::type& items, Mat3<double>::type& sn,
                         double thresh, std::vector<int>& sample );


#endif //SUB_SAMPLE_HEADER_GUARD
