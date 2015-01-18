/*! fpfh_interface.cpp
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
 *  \author     Alastair Quadros
 *  \date       25-11-2010
*/
#include "fpfh_interface.h"

#include <omp.h>

#include "pcl/pcl_base.h"
#include "pcl/features/fpfh.h"
#include "pcl/features/impl/fpfh.hpp"
#include "pcl/point_cloud.h"
#include "pcl/point_types.h"

//#include "Common/ProgressIndicator.h"
#include "DataStore/Selector.h"
#include "LaserLibConfig.h" //cmake option for openmp

using namespace Eigen;


//fpfh_histogram: float vector, length: sum(noBins)
//noBins: try 11 (default)
void fpfh( Mat3<double>::type& P, Mat3<double>::type& sn,
           Selector& sel, Vect<int>::type& keys,
           Eigen::Vector3i& noBins, MapMatXf& hist, int nThreads )
{
    // setup histogram matrices. they are aligned to the full point/sn data (not keypoints).

    // NOTE: from PCL 1.4, it uses row-major eigen matrices
    MatrixXf hist_f1(P.rows(), noBins(0));
    MatrixXf hist_f2(P.rows(), noBins(1));
    MatrixXf hist_f3(P.rows(), noBins(2));
    hist_f1.setZero();
    hist_f2.setZero();
    hist_f3.setZero();

    VectorXf fpfh_histogram( noBins.sum() );
    fpfh_histogram.setZero();

    //convert point and normals to PointCloud type
    pcl::PointCloud<pcl::Normal> normals;
    pcl::PointCloud<pcl::PointXYZ> points;
    points.points.resize(P.rows());
    normals.points.resize(sn.rows());

    //make a bool array indicating if surface normals are valid.
    //can filter out invalid ones after region selection.
    std::vector<bool> isSnValid(sn.rows(), true);
    for( int i=0 ; i<sn.rows() ; i++ )
    {
        points.points[i].x = (float)P(i,0);
        points.points[i].y = (float)P(i,1);
        points.points[i].z = (float)P(i,2);
        normals.points[i].normal[0] = (float)sn(i,0);
        normals.points[i].normal[1] = (float)sn(i,1);
        normals.points[i].normal[2] = (float)sn(i,2);

        if( sn.row(i).norm() == 0 )
            isSnValid[i] = false;
    }

    //cache the region selection, so we don't have to do it twice
    std::vector< std::vector<int> > allNeighs(keys.size());

    pcl::FPFHEstimation<pcl::PointXYZ, pcl::Normal, pcl::FPFHSignature33> histCalc;

    // Estimate the FPFH signature at each patch
    #ifdef LaserLib_USE_OPENMP
    if( nThreads > 0 )
        omp_set_num_threads(nThreads);
    #endif

    std::vector< boost::shared_ptr<Selector> > sels;
    int i;
    int threadId=0;
    int nThreadsUsed=1;
    #pragma omp parallel private(threadId, histCalc) shared(sels)
    {
        //copy the selector, one for each thread
        #pragma omp master
        {
            #ifdef LaserLib_USE_OPENMP
            nThreadsUsed = omp_get_num_threads();
            #endif
            for( int j=0 ; j<nThreadsUsed ; j++ )
                { sels.push_back( sel.clone() ); }
        }
        #pragma omp barrier //need to wait for master to finish
        #pragma omp for
        for( i=0 ; i<keys.size() ; i++ )
        {
            int key=keys(i);
            //get neighbours (using the selector for this thread)
            #ifdef LaserLib_USE_OPENMP
            threadId = omp_get_thread_num();
            #endif
            std::vector<int>& neighsRaw = sels[threadId]->SelectRegion(key);
            std::vector<int>& neighs = allNeighs[i];
            //only add points with valid surface normals
            //neighs.clear();
            for( unsigned int j=0 ; j<neighsRaw.size() ; j++ )
            {
                if( isSnValid[neighsRaw[j]] )
                    neighs.push_back(neighsRaw[j]);
            }
            if( allNeighs[i].size() <= 1 )
                continue;

            histCalc.computePointSPFHSignature(points, normals, key, key, neighs,
                                       hist_f1, hist_f2, hist_f3);
        }
    }

    //allocate distance vector for the next loop
    std::vector<float> dists;
    dists.resize(3000);
    RowVector3d rel;

    // Compute weighted signature
    #pragma omp parallel for private(histCalc) firstprivate(dists, fpfh_histogram)
    for( i=0 ; i<keys.size() ; i++ )
    {
        int key=keys(i);
        //get neighbours
        std::vector<int>& neighs = allNeighs[i];
        dists.clear();
        for( unsigned int j=0 ; j<neighs.size() ; j++ )
        {
            rel = P.row(key)-P.row(neighs[j]);
            dists.push_back(rel.norm());
        }
        if( neighs.size() <= 1 )
            continue;

        histCalc.weightPointSPFHSignature(hist_f1, hist_f2, hist_f3, neighs, dists, fpfh_histogram);

        //store this point's results in the full array
        for( int j=0 ; j<fpfh_histogram.size() ; j++ )
        {
            hist(i,j) = fpfh_histogram(j);
        }
        fpfh_histogram.setZero();
    }
}
