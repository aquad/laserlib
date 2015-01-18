/*! PCA.cpp
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
 * \date       24-11-2010
*/

#include "PCA.h"
#include <iostream>


using namespace std;
using namespace Eigen;


PCA::PCA(Mat3<double>::type& P)
    :   P_(P)
{}


void PCA::compute(std::vector<int>& neigh, Map<Vector3f>& evals, MapMat3f& evects, Map<Vector3d>& meanP )
{
    //set matrices used with += to zero
    RowVector3d mean;
    mean.setZero();
    Matrix3f cov;
    cov.setZero();

    //get mean point:
    //Xbar = 1/N sum(Xi)
    for(int c=0 ; c<neigh.size() ; c++)
    {
        int n = neigh[c];
        mean += P_.row(n);
    }
    mean /= neigh.size();
    meanP = mean;

    //symmetric covariance matrix
    //cov = 1/N sum( (Xi - Xbar) (Xi - Xbar)' )
    Vector3d Xim;
    for(int c=0 ; c<neigh.size() ; c++)
    {
        int n = neigh[c];
        Xim = P_.row(n) - mean;
        cov += (Xim * Xim.transpose()).cast<float>();
    }
    cov /= neigh.size();

    SelfAdjointEigenSolver<Matrix3f> eigensolver(cov);
    evals = eigensolver.eigenvalues();
    evects = eigensolver.eigenvectors();
}





void minRadiusSelection( Graph& graph, Mat3<double>::type& P, Vect<float>::type& rad, Vect<bool>::type& valid )
{
    for(int i=0 ; i<P.rows() ; i++)
    {
        RowVector3d centreP = P.row(i);
        //of the vertical neighbours that exist, get the closest distance.
        double upDist = 0;
        bool upDistExists = false;
        double downDist = 0;
        bool downDistExists = false;
        double vertDist = 0;
        bool vertDistExists = false;
        int upId = graph(i,2);
        if( upId!=-1 )
        {
            RowVector3d rel = P.row(upId) - centreP;
            upDist = rel.norm();
            upDistExists = true;
        }
        int downId = graph(i,3);
        if( downId!=-1 )
        {
            RowVector3d rel = P.row(downId) - centreP;
            downDist = rel.norm();
            downDistExists = true;
        }

        if( upDistExists && downDistExists )
        {
            vertDist = std::min(upDist, downDist);
            vertDistExists = true;
        }
        else if( upDistExists )
        {
            vertDist = upDist;
            vertDistExists = true;
        }
        else if( downDistExists )
        {
            vertDist = downDist;
            vertDistExists = true;
        }

        //of the horizontal neighbours that exist, get the closest distance.
        double leftDist = 0;
        bool leftDistExists = false;
        double rightDist = 0;
        bool rightDistExists = false;
        double horizDist = 0;
        bool horizDistExists = false;
        int leftId = graph(i,0);
        if( leftId!=-1 )
        {
            RowVector3d rel = P.row(leftId) - centreP;
            leftDist = rel.norm();
            leftDistExists = true;
        }
        int rightId = graph(i,1);
        if( rightId!=-1 )
        {
            RowVector3d rel = P.row(rightId) - centreP;
            rightDist = rel.norm();
            rightDistExists = true;
        }

        if( leftDistExists && rightDistExists )
        {
            horizDist = std::min(leftDist, rightDist);
            horizDistExists = true;
        }
        else if( leftDistExists )
        {
            horizDist = leftDist;
            horizDistExists = true;
        }
        else if( rightDistExists )
        {
            horizDist = rightDist;
            horizDistExists = true;
        }

        //maximum of horizontal & vertical distances (if they both exist etc)
        if( vertDistExists && horizDistExists )
        {
            rad(i) = std::max(vertDist, horizDist);
            valid(i) = 1;
        }
        else if( vertDistExists )
        {
            rad(i) = vertDist;
            valid(i) = 1;
        }
    }
}



