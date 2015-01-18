/*! PCAFrame.cpp
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
 * \date       25-01-2012
*/

#define EIGEN_DEFAULT_TO_ROW_MAJOR 1

#include <boost/math/constants/constants.hpp>
const double pi = boost::math::constants::pi<double>();

#include "PCAFrame.h"
#include "DataStore/FlannKDTree.h"
#include "DataStore/Subsample.h"

using namespace Eigen;

void ComputePCAFrames(
        Mat3<double>::type meanP, Mat3<float>::type evals, MapMat33Xf evects,
        PCAFrames& frames, float surfThresh, float linThresh, float ssRad)
{
    //determine what pca results are flat or linear
    std::vector<int> flatPoints, linearPoints;
    int nResults = evals.rows();
    for( int i=0 ; i<nResults ; i++ )
    {
        float evalsNorm = evals.row(i).norm();
        float surfness = (evals(i,1) - evals(i,0)) / evalsNorm;
        if( surfness > surfThresh )
        {
            flatPoints.push_back(i);
        }
        else
        {
            float linear = (evals(i,2) - evals(i,1)) / evalsNorm;
            if( linear > linThresh )
            {
                linearPoints.push_back(i);
            }
        }
    }

    //subsample
    std::vector<int> flatSamples, linearSamples;
    FlannKDTree<double> sel( meanP, ssRad );
    if( flatPoints.size() > 0 )
    {
        Vect<int>::type flatPoints_vect( &flatPoints[0], flatPoints.size() );
        SubSampleKeysEvenly( sel, nResults, flatPoints_vect, flatSamples );
    }
    if( linearPoints.size() > 0 )
    {
        Vect<int>::type linearPoint_vect( &linearPoints[0], linearPoints.size() );
        SubSampleKeysEvenly( sel, nResults, linearPoint_vect, linearSamples );
    }

    //create the result storage
    int nFrames = flatSamples.size() + linearSamples.size();
    //boost::shared_ptr<PCAFrames> result( new PCAFrames(nFrames) );
    frames.resize(nFrames);

    //for each flat sample, get the surf norm
    Matrix3f thisR;
    Map< Matrix<float,1,9> > thisR_flattened( thisR.data() );
    for( int i=0 ; i<flatSamples.size() ; i++ )
    {
        int id = flatSamples[i];
        Map< Matrix3f > evect( evects.row(id).data() );
        Vector3f sn = evect.col(0); //smallest eigenvector
        Vector3f p = meanP.row(id).cast<float>();
        //face it the right way
        float dotP = -sn.dot( p );
        float normP = p.norm();
        float angle = acos( dotP/normP ); //may have to check for (-1,1)
        if( angle > pi/2 ){ sn *= -1; }
        thisR.row(2) = sn;
        if( fabs(sn.coeff(2)) > 0.99 ) //facing up/down, just use largest evect
        {
            thisR.row(0) = evect.col(2).transpose();
        }
        else //downwards (but perp to sn- project (0,0,1) onto other evects)
        {
            Vector3f downProject = evect.coeff(2,1) * evect.col(1) + evect.coeff(2,2) * evect.col(2);
            thisR.row(0) = downProject.normalized();
        }
        thisR.row(1) = sn.cross( thisR.row(0) ).normalized();

        frames.R.row(i) = thisR_flattened;
        frames.frameType(i,0) = FRAME_SURFACE;
        frames.P.row(i) = p;
        frames.pcaId[i] = id;
        frames.alignVect.row(i) = sn;
    }

    //same for linear samples
    for( int i=0 ; i<linearSamples.size() ; i++ )
    {
        int id = linearSamples[i];
        Map< Matrix3f > evect( evects.row(id).data() );
        Vector3f line = evect.col(2); //largest eigenvector
        Vector3f p = meanP.row(id).cast<float>();

        // z (eg line image line direction) is facing towards sensor
        thisR.row(2) = p.dot(line)*line - p;
        thisR.row(2).normalize();

        // if the line is mostly downwards, set it as x
        if( fabs(line(2)) > 0.70710678118654757 ) // cos(45deg)
        {
            // face it downwards (if up), note z+ is downwards
            if( line.coeff(2) < 0 ){ line *= -1; }
            thisR.row(0) = line;
            thisR.row(1) = thisR.row(2).cross( line ).normalized();
        }
        else // mostly sidewards, y
        {
            // face x downwards, as a similarly shaped just-too-flat region
            // would do this. (think 45 deg-facing linear/flat region,
            // want jumps between 45-deg boundary and flat/linear to be smooth.
            // watch the order of cross products!
            Vector3f xAxis = line.cross(thisR.row(2)).normalized();
            if( xAxis.coeff(2) < 0)
            {
                xAxis *= -1;
                line *= -1; // to keep cross product result as right-hand axes
            }
            thisR.row(0) = xAxis;
            thisR.row(1) = line;
        }
        int rid = flatSamples.size() + i;
        frames.R.row(rid) = thisR_flattened;
        frames.frameType(rid,0) = FRAME_LINEAR;
        frames.P.row(rid) = p;
        frames.pcaId[rid] = id;
        frames.alignVect.row(rid) = line;
    }
}

