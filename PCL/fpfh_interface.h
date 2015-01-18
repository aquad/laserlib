/*! fpfh_interface.h
 * Fast point feature histogram (PCL interface).
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
 * \date       25-11-2010
*/
#ifndef FPFH_HEADER_GUARD
#define FPFH_HEADER_GUARD

#define EIGEN_DEFAULT_TO_ROW_MAJOR 1
#include <Eigen/Core>
#include "Common/ArrayTypes.h"
#include "DataStore/Selector.h"
#include "Features/KnnClassifier.h"
#include "Features/Histogram.h"


void fpfh( Mat3<double>::type& P, Mat3<double>::type& sn,
           Selector& sel, Vect<int>::type& keys,
           Eigen::Vector3i& noBins, MapMatXf& hist, int nThreads=1 );


//! Classify test points using the FPFH feature, KNN and a training set
class fpfh_knn_classifier : public KnnClassifier
{
public:
    fpfh_knn_classifier( MapMatXf& test, MapMatXf& train )
        : test_(test), train_(train)
    {}

    float Distance( int i, int j )
    {
        return -hist_intersection_kernel( test_.row(i), train_.row(j) );
    }

private:
    MapMatXf test_, train_;
};




//! Classify points in an object using a dataset, KNN and the FPFH feature
class FPFHObjectKnn : public ObjectKnnClassifier
{
public:
    FPFHObjectKnn( std::vector<MapMatXf>& trainObjData, std::vector<int>& nTrainPts,
                   bool showProgress=false )
        :   ObjectKnnClassifier( trainObjData.size(), nTrainPts, showProgress ),
            trainObjData_(trainObjData)
    {}

    void SetTestObject( MapMatXf& testData )
    {
        testData_.reset( new MapMatXf(testData) );
        SetNTest( testData.rows() );
    }

    //! test point i, train object o, train point j
    inline float Distance(int i, int o, int j)
    {
        MapMatXf& train = trainObjData_[o];
        MapMatXf& test = *testData_.get();
        return -hist_intersection_kernel( test.row(i), train.row(j) );
    }

    std::vector<MapMatXf> trainObjData_;
    boost::shared_ptr<MapMatXf> testData_;
};



#endif //FPFH_HEADER_GUARD
