/*! ClusterKMeans.h
 *
 * Copyright (C) 2012 Alastair Quadros.
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
 * \date       18-01-2012
*/
#ifndef CLUSTER_K_MEANS_HEADER_GUARD
#define CLUSTER_K_MEANS_HEADER_GUARD

#include "export.h"
#include <Eigen/Core>
#include "Common/ArrayTypes.h"

LASERLIB_MISC_EXPORT void ClusterKMeans( MapMatXf& data, int nClusters, Vect<int>::type& ids, int nIters );

#endif //CLUSTER_K_MEANS_HEADER_GUARD
