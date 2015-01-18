/*! PCAGrid.cpp
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

#define EIGEN_DEFAULT_TO_ROW_MAJOR 1
#include <Eigen/Core>
#include "PCAGrid.h"
#include <iostream>


using namespace Eigen;


PCAGrid::PCAGrid( Mat3<float>::type meanP, Mat3<float>::type evals, MapMat33Xf evects,
                  float gridLength, int cellSide )
    : meanP_(meanP), evals_(evals), evects_(evects),
      gridLength_(gridLength), cellSide_(cellSide),
      nCells(cellSide*cellSide),
      localP(nCells,3),
      ind( flann::Matrix<float>(meanP.data(), meanP.rows(), meanP.cols()), flann::KDTreeSingleIndexParams(10,true) ),
      searchParams( flann::FLANN_CHECKS_UNLIMITED, 0, true ),
      indices(nCells), dists(nCells),
      indices_flann(indices.data(), nCells, 1),
      dists_flann(dists.data(), nCells, 1),
      cellLength(gridLength_ / cellSide_),
      halfCell(cellLength/2)
{
    ind.buildIndex();
    //make local grid
    for( int i=0 ; i<cellSide_ ; i++ )
    {
        for( int j=0 ; j<cellSide_ ; j++ )
        {
            localP.row(cellSide_*i+j) = Vector3f( cellLength*j, cellLength*i, 0 );
        }
    }
    localP.rowwise() += RowVector3f(halfCell, halfCell, 0); //cell centres
    localP.rowwise() -= RowVector3f(gridLength_/2, gridLength_/2, 0);
}



//make sure result arrays are the right size, as we won't check them in here (for speed!)
void PCAGrid::compute( Map<Matrix3f>& R, Vector3f& point, PCAGridElement& data )
{
    Matrix3f Rt = R.transpose();
    //sensor-frame cells
    MatrixXf cellP = Rt * localP.transpose();
    cellP.colwise() += point;
    cellP.transposeInPlace();

    //find nearest meanP
    flann::Matrix<float> queries(cellP.data(), nCells, 3);
    int val = ind.knnSearch(queries, indices_flann, dists_flann, 1, searchParams );
    data.valid.setZero();
    //transform corresponding meanP points and evects to local frame
    for( int i=0 ; i<nCells ; i++ )
    {
        int id = indices[i];
        //transform point
        Vector3f p = R * (meanP_.row(id).transpose() - point);
        //check point is within cell
        Vector3f rel = localP.row(i).transpose() - p;
        if( fabs(rel[0]) > halfCell || fabs(rel[1]) > halfCell || fabs(rel[2]) > halfCell )
            continue;

        data.P.row(i) = p;
        data.evals.row(i) = evals_.row(id);
        data.valid[i] = 1;

        //transform eigenvectors
        Map<Matrix3f> thisEvect( evects_.row(id).data() ); //one vector per column
        Matrix3f localEvect = Rt * thisEvect; //so no need to transpose
        //flatten again
        Map< Matrix<float,9,1> > localEvect_flat( localEvect.data() );
        data.evects.row(i) = localEvect_flat;
    }
}


