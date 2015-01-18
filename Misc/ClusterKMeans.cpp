/*! ClusterKMeans.cpp
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

// this is ok here, not in the header. the definition is indepedent of this.
// this allows eigen matrices to be used with flann matrices.
#define EIGEN_DEFAULT_TO_ROW_MAJOR 1

#include <stdlib.h>
#include <vector>
#include <flann/flann.hpp>
#include "ClusterKMeans.h"


using namespace Eigen;

void ClusterKMeans( MapMatXf& data, int nClusters, Vect<int>::type& ids, int nIters )
{
    int nPoints = data.rows();
    int nDims = data.cols();
    flann::SearchParams searchParams( flann::FLANN_CHECKS_UNLIMITED, 0, true );
    flann::Matrix<float> queries( data.data(), nPoints, nDims );

    VectorXf dists( nPoints );
    flann::Matrix<int> indices_mat(ids.data(), nPoints, 1);
    flann::Matrix<float> dists_mat(dists.data(), nPoints, 1);

    MatrixXf centres( nClusters, nDims );
    VectorXf nPerCluster( nClusters );
    std::vector<int> centreIds;
    centreIds.resize(nClusters);
    for( int i=0 ; i<nClusters ; i++ )
    {
        centreIds[i] = rand() % nPoints;
        centres.row(i) = data.row( centreIds[i] );
    }

    for( int i=0 ; i<nIters ; i++ )
    {
        //find nearest cluster centre for each point
        flann::Matrix<float> centres_mat( centres.data(), centres.rows(), centres.cols() );
        flann::Index< flann::L2<float> > ind( centres_mat, flann::KDTreeSingleIndexParams(10,true) );
        ind.buildIndex();
        int val = ind.knnSearch( queries, indices_mat, dists_mat, 1, searchParams );

        //for each cluster find the mean (store sum in centres)
        nPerCluster.setZero();
        centres.setZero();
        for( int k=0 ; k<nPoints ; k++ )
        {
            int clusterId = ids[k];
            nPerCluster[ clusterId ] += 1;
            centres.row( clusterId ) += data.row(k);
        }
        for( int cl=0 ; cl<nClusters ; cl++ )
        {
            centres.row(cl) /= nPerCluster[cl];
        }
    }
}

