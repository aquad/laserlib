/*! KnnClassifier.cpp
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
 * \date       06-12-2010
*/

#include <omp.h>
#include <vector>
#include <algorithm>
#include "KnnClassifier.h"
#include "Common/ProgressIndicator.h"
#include "Common/argsort.h"
#include "LaserLibConfig.h"

using namespace Eigen;


void KnnClassifier::Classify( int nTest, int nTraining, MapMatXi& matches, MapMatXf& values )
{
    unsigned int knn = matches.cols();
    //temp storage for all comparison values of current point
    std::vector<float> allVals;
    std::vector<int> argSorted;
    allVals.resize(nTraining);
    argSorted.resize(nTraining);

    ProgressIndicator prog(nTest, 5);
    if( showProgress_ ) { prog.start(); }

    #ifdef LaserLib_USE_OPENMP
    if( nThreads_ > 0 )
        omp_set_num_threads(nThreads_);
    #endif

    int i;
    #pragma omp parallel for firstprivate(allVals, argSorted)
    for( i=0 ; i<nTest ; i++)
    {
        //compare this point to all training points
        for( int j=0 ; j<nTraining ; j++ )
        {
            allVals[j] = Distance(i, j);
        }

        //get top k
        argsort( allVals.begin(), allVals.end(), argSorted.begin(), argSorted.end() );
        for( int k=0 ; k < std::min(int(knn),nTraining) ; k++ )
        {
            matches(i,k) = argSorted[k];
            values(i,k) = allVals[ argSorted[k] ];
        }
        if( showProgress_ ) { prog+=1; }
    }
    if( showProgress_ ) { prog.stop(); }
}


//matches, values: shape (testIds.size(), knn). match ids refer to full length ids.
void KnnClassifier::ClassifyKeys( Vect<int>::type& testIds, Vect<int>::type& trainIds,
                         MapMatXi& matches, MapMatXf& values )
{
    unsigned int knn = matches.cols();
    //temp storage for all comparison values of current point
    std::vector<float> allVals;
    std::vector<int> argSorted;
    allVals.resize(trainIds.size());
    argSorted.resize(trainIds.size());

    ProgressIndicator prog(testIds.size(), 5);
    if( showProgress_ ) { prog.start(); }

    #ifdef LaserLib_USE_OPENMP
    if( nThreads_ > 0 )
        omp_set_num_threads(nThreads_);
    #endif

    int i;
    #pragma omp parallel for firstprivate(allVals, argSorted)
    for( i=0 ; i<testIds.size() ; i++)
    {
        //compare this point to all training points
        int testI = testIds[i];
        for( int j=0 ; j<trainIds.size() ; j++ )
        {
            allVals[j] = Distance(testI, trainIds[j]);
        }

        //get top k
        argsort( allVals.begin(), allVals.end(), argSorted.begin(), argSorted.end() );
        int nTrain = trainIds.size();
        for( int k=0 ; k < std::min(int(knn),nTrain) ; k++ )
        {
            matches(i,k) = trainIds[ argSorted[k] ];
            values(i,k) = allVals[ argSorted[k] ];
        }
        if( showProgress_ ) { prog+=1; }
    }
    if( showProgress_ ) { prog.stop(); }
}



void ObjectKnnClassifier::Classify( MapMatXi& objectMatches,
                                    MapMatXi& pointMatches, MapMatXf& values )
{
    boost::posix_time::ptime startTime = boost::posix_time::microsec_clock::universal_time();
    int knn = objectMatches.cols();

    //will be accumulating all training points together in one array-
    // trainIdx informs the index each object starts at
    std::vector<int> trainIdx(nTrainObjs_ + 1);
    int accum = 0;
    trainIdx[0] = 0;
    for( int i=1 ; i<trainIdx.size() ; i++ )
    {
        accum += nTrainPts_[i-1];
        trainIdx[i] = accum;
    }

    int nSorted = std::min(nTrainTotal_, int(knn));

    ProgressIndicator prog(nTestPts_, 5);
    if( showProgress_ ) { prog.start(); }
    int i;
    //#pragma omp parallel for firstprivate(allVals_, argSorted_)
    //for each test point
    for( i=0 ; i<nTestPts_ ; i++)
    {
        int trainId = 0;
        //for each training object
        for( int o=0 ; o<nTrainObjs_ ; o++ )
        {
            //for each training point
            for( int j=0 ; j<nTrainPts_[o] ; j++ )
            {
                allVals_[trainId++] = Distance(i, o, j);
            }
        }

        //get top k
        // using partial_sort- middle iterator is upper boundary of sorted range
        std::vector<int>::iterator midArgIt = argSorted_.begin() + nSorted;
        partial_argsort( allVals_.begin(), allVals_.end(), argSorted_.begin(), midArgIt, argSorted_.end() );
        for( int k=0 ; k < nSorted ; k++ )
        {
            trainId = argSorted_[k];
            int objNo = (std::upper_bound( trainIdx.begin(), trainIdx.end(), trainId ) - 1) - trainIdx.begin();
            int pointId = trainId - trainIdx[objNo];
            objectMatches(i,k) = objNo;
            pointMatches(i,k) = pointId;
            values(i,k) = allVals_[ trainId ];
        }
        if( showProgress_ ) { prog+=1; }
    }
    if( showProgress_ ) { prog.stop(); }
    computeTime = boost::posix_time::microsec_clock::universal_time() - startTime;
}

