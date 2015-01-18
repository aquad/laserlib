/*! PCAGrid.h
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
 * \date       28-01-2012
*/

#ifndef PCA_GRID_HEADER_GUARD
#define PCA_GRID_HEADER_GUARD


#define EIGEN_DEFAULT_TO_ROW_MAJOR 1

#include <Eigen/Core>
#include <flann/flann.hpp>
#include "Common/ArrayTypes.h"
#include "PCA.h"


// store a single PCAGrid result (a grid of points, evects etc)
struct PCAGridElement
{
    PCAGridElement( Mat3<float>::type P_, Mat3<float>::type evals_,
                    MapMat33Xf evects_, Vect<unsigned char>::type valid_ )
        : P(P_), evals(evals_), evects(evects_), valid(valid_)
    {}

    Mat3<float>::type P;
    Mat3<float>::type evals;
    MapMat33Xf evects;
    Vect<unsigned char>::type valid;
};



// compute a feature based on a grid of PCA results
class PCAGrid
{
public:
    PCAGrid( Mat3<float>::type meanP, Mat3<float>::type evals, MapMat33Xf evects,
             float gridLength, int cellSide );
    void compute( Eigen::Map<Eigen::Matrix3f>& R, Eigen::Vector3f& point,
                  PCAGridElement& data );

    int nCells;

private:
    Mat3<float>::type meanP_;
    Mat3<float>::type evals_;
    MapMat33Xf evects_;
    float gridLength_;
    int cellSide_;
    float cellLength, halfCell;

    Eigen::MatrixXf localP; //local cell centres

    //flann stuff
    flann::SearchParams searchParams;
    flann::Index< flann::L2<float> > ind;
    Eigen::VectorXi indices; //actual storage here
    Eigen::VectorXf dists;
    flann::Matrix<int> indices_flann; //these wrap the above
    flann::Matrix<float> dists_flann;
};



#endif //PCA_GRID_HEADER_GUARD
