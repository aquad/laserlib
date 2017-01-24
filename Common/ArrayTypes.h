/* ArrayTypes.h
 *
 * Some typedefs for commonly used eigen matrices.
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
 * \date       08-03-2011
*/

#ifndef ARRAY_TYPES
#define ARRAY_TYPES

#include "export.h"

//Python defaults to row major, but eigen (and PCL) defaults to column major.
//This line must be before any eigen includes, so that eigen typedefs like MatrixXf are row-major.
//Although, if you only use the eigen typedefs here, this line may not be necessary.
#define EIGEN_DEFAULT_TO_ROW_MAJOR 1

#include <Eigen/Core>


//some templated eigen types.
template <typename T>
struct Vect {
    typedef Eigen::Map< Eigen::Matrix< T, Eigen::Dynamic, 1 > > type;
};

LASERLIB_COMMON_EXTERN template struct LASERLIB_COMMON_IMPORT Vect<double>;
LASERLIB_COMMON_EXTERN template struct LASERLIB_COMMON_IMPORT Vect<float>;

template <typename T>
struct Mat2 {
    typedef Eigen::Map< Eigen::Matrix< T, Eigen::Dynamic, 2, Eigen::RowMajor > > type;
};

LASERLIB_COMMON_EXTERN template struct LASERLIB_COMMON_IMPORT Mat2<double>;
LASERLIB_COMMON_EXTERN template struct LASERLIB_COMMON_IMPORT Mat2<float>;

template <typename T>
struct Mat3 {
    typedef Eigen::Map< Eigen::Matrix< T, Eigen::Dynamic, 3, Eigen::RowMajor > > type;
};

LASERLIB_COMMON_EXTERN template struct LASERLIB_COMMON_IMPORT Mat3<double>;
LASERLIB_COMMON_EXTERN template struct LASERLIB_COMMON_IMPORT Mat3<float>;

typedef Eigen::Map< Eigen::Matrix< int, Eigen::Dynamic, 4, Eigen::RowMajor > > Graph;

typedef Eigen::Matrix<double, Eigen::Dynamic, 3, Eigen::RowMajor> MatX3d;
typedef Eigen::Matrix<float, Eigen::Dynamic, 3, Eigen::RowMajor> MatX3f;

typedef Eigen::Map< Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> > MapMatXf;
typedef Eigen::Map< Eigen::Matrix<int, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> > MapMatXi;
typedef Eigen::Map< Eigen::Matrix<unsigned char, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> > MapMatXuc;

typedef Eigen::Map< Eigen::VectorXf > MapVecXf;
typedef Eigen::Map< Eigen::VectorXd > MapVecXd;
typedef Eigen::Map< Eigen::VectorXi > MapVecXi;
typedef Eigen::Map< Eigen::Vector3f > MapVec3f;
typedef Eigen::Map< Eigen::Vector3d > MapVec3d;
typedef Eigen::Map< Eigen::Vector3i > MapVec3i;
typedef Eigen::Map< Eigen::Matrix<long long int, Eigen::Dynamic, 1> > MapVecXll;
typedef Eigen::Map< Eigen::Matrix<unsigned char, Eigen::Dynamic, 1> > MapVecXuc;

#endif //ARRAY_TYPES
