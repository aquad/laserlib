/*! Histogram.h
 *
 * Copyright (C) 2010 Alastair Quadros.
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
 * \date       01-12-2010
*/

#ifndef HISTOGRAM_HEADER_GUARD
#define HISTOGRAM_HEADER_GUARD

//Eigen
#define EIGEN_DEFAULT_TO_ROW_MAJOR 1
#include <Eigen/Core>

#include <algorithm>

template <typename Derived, typename OtherDerived>
float hist_intersection_kernel( const Eigen::MatrixBase<Derived>& h1, const Eigen::MatrixBase<OtherDerived>& h2 )
{
    float sum = 0.0;
    for( int i=0 ; i<h1.size() ; i++ )
    {
        sum += std::min(h1.coeff(i), h2.coeff(i));
    }
    return sum;
}


#endif //HISTOGRAM_HEADER_GUARD
