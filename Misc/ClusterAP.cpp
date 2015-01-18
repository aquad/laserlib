/*! ClusterAP.cpp
 *
 * Copyright (C) 2013 Alastair Quadros.
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
 * \date       10-07-2013
*/

#include <iostream>
#include <limits>
#include "ClusterAP.h"
#include "LaserLibConfig.h" //cmake option for openmp


using namespace Eigen;


ClusterAP::ClusterAP( MapMatXf& sim, bool showProgress, float damping,
        int minIter, int convergeIter, int maxIter,
        int nThreads )
    : sim_(sim.data(), sim.rows(), sim.cols()),
      showProgress_(showProgress),
      damping_(damping),
      minIter_(minIter),
      convergeIter_(convergeIter),
      maxIter_(maxIter),
      avail_( Eigen::MatrixXf::Zero(sim.rows(), sim.cols()) ),
      respon_( Eigen::MatrixXf::Zero(sim.rows(), sim.cols()) ),
      assign_( Eigen::VectorXi::Zero(sim.rows()) ),
      assignScores_( Eigen::VectorXf::Zero(sim.rows()) ),
      assignDegenerate_(true),
      sumR_( Eigen::VectorXf::Zero(sim.rows()) )
{
    #ifdef LaserLib_USE_OPENMP
    if( nThreads > 0 )
        omp_set_num_threads(nThreads);
    #endif
}


int ClusterAP::run()
{
    int convergeCount = 0; // number of iterations with same exemplars
    int nIters = 0;

    for(int i=0; i<maxIter_; i++, nIters++)
    {
        updateResponsibilities();
        updateAvailabilities();
        bool changed = computeAssignments();

        if( showProgress_ )
        {
            float netSim = assignScores_.sum();
            std::cout << "iter: " << nIters << ", netSim: " << netSim << "          \r" << std::flush;
        }

        // Check for convergence using exemplar changes
        if( changed )
            convergeCount = 0;
        else
            convergeCount++;

        if(convergeCount >= convergeIter_ && nIters >= minIter_ && !assignDegenerate_)
            break;
    }

    return nIters;
}



bool ClusterAP::runIters( int nIters )
{
    for(int i=0; i<nIters; ++i)
    {
        updateResponsibilities();
        updateAvailabilities();
    }
    bool changed = computeAssignments();
    return changed;
}



MapVecXi ClusterAP::getAssignments()
{
    return MapVecXi( assign_.data(), assign_.rows() );
}


MapVecXf ClusterAP::getAssignmentScores()
{
    return MapVecXf( assignScores_.data(), assignScores_.rows() );
}


void ClusterAP::updateResponsibilities()
{
    int n = sim_.rows();
    int i;
    #pragma omp parallel for
    for(i=0 ; i<n ; i++) //rows
    {
        // find the 2 largest (avail_ + sim_) values in this row, for the upcoming max statement.
        // (2, so we have a max for the k'=k case). Just do 2 passes.
        float maxVal = -std::numeric_limits<float>::max();
        int maxValIndex = -1;
        for(int j=0 ; j<n ; j++) //cols
        {
            float val = avail_.coeff(i,j) + sim_.coeff(i,j);
            if( val > maxVal )
            {
                maxVal = val;
                maxValIndex = j;
            }
        }

        //second largest
        float maxVal2 = -std::numeric_limits<float>::max();
        for(int j=0 ; j<n ; j++) //cols
        {
            float val = avail_.coeff(i,j) + sim_.coeff(i,j);
            if( val > maxVal2 && j != maxValIndex )
            {
                maxVal2 = val;
            }
        }

        for(int k=0 ; k<n ; k++) //cols (exemplars)
        {
            float r = 0; // new responsibility
            // max_{k', k' != k}{ a(i,k') + s(i,k') }
            if( maxValIndex != k )
            {
                r = sim_.coeff(i,k) - maxVal;
            }
            else // k' = k
            {
                r = sim_.coeff(i,k) - maxVal2;
            }

            //update with damping
            respon_.coeffRef(i,k) = damping_ * respon_.coeff(i,k) + (1-damping_) * r;
        }
    }
}




void ClusterAP::updateAvailabilities()
{
    int n = sim_.rows();

    //sum over columns for respon_, excluding diagonal and negatives
    sumR_.setZero(n);
    for(int i=0 ; i<n ; i++) //rows
    {
        for(int j=0 ; j<n ; j++) //cols
        {
            float r = respon_.coeff(i,j);
            if( i != j && r > 0 )
                sumR_.coeffRef(j) += r;
        }
    }

    int i;
    #pragma omp parallel for
    for(i=0 ; i<n ; i++) //rows
    {
        for(int k=0 ; k<n ; k++) //cols (exemplars)
        {
            float a = 0; // new availability
            if( i!=k )
            {
                // sum_{i', i' != i,k}{ max(0,r(i',k)) }
                a = sumR_.coeff(k);
                float rik = respon_.coeff(i,k);
                if( rik > 0 )
                    a -= rik;

                // min(0, r(k,k) + above)
                a += respon_.coeff(k,k);
                a = a > 0 ? 0 : a;
            }
            else // a(k,k)
            {
                // sum_{i', i' != k}{ max(0,r(i',k)) }
                a = sumR_.coeff(k);
            }

            //update with damping
            avail_.coeffRef(i,k) = damping_ * avail_.coeff(i,k) + (1-damping_) * a;
        }
    }
}





bool ClusterAP::computeAssignments()
{
    bool changed = false;
    int n = sim_.rows();
    int i;
    // bit messy- assignDegenerate should be a returned value
    bool degenerate = true;
    #pragma omp parallel for
    for(i=0 ; i<n ; i++) //rows
    {
        // find k that max( a(i,k) + r(i,k) )
        int newAssign = 0;
        float maxVal = -std::numeric_limits<float>::max();
        for(int k=0 ; k<n ; k++) //cols (exemplars)
        {
            float val = avail_.coeff(i,k) + respon_.coeff(i,k);
            if( val > maxVal )
            {
                maxVal = val;
                newAssign = k;
            }
        }

        // Changed if this is an exemplar and wasn't before,
        // or is not an exemplar, but was before.
        bool isExemplar = newAssign == i;
        bool wasExemplar = assign_.coeff(i) == i;
        if(     isExemplar && !wasExemplar ||
                !isExemplar && wasExemplar )
        {
            #pragma omp critical
            changed = true;
        }
        assign_.coeffRef(i) = newAssign;
        assignScores_.coeffRef(i) = maxVal;

        if(!isExemplar)
            degenerate = false;
    }
    assignDegenerate_ = degenerate;
    return changed;
}

