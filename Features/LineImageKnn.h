/*! LineImageKnn.h
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
 * \date       26-05-2012
*/

#ifndef LINE_IMAGE_KNN_HEADER_GUARD
#define LINE_IMAGE_KNN_HEADER_GUARD

#include "export.h"

//Eigen
#define EIGEN_DEFAULT_TO_ROW_MAJOR 1
#include <Eigen/Core>

#include "LineImage.h"
#include "KnnClassifier.h"
#include "LineImageMatcher.h"
#include <iostream>
#include <vector>
#include <boost/shared_ptr.hpp>



/*! Classify points in an object using KNN.
*/
class LASERLIB_FEATURES_EXPORT LineImageKnn : public ObjectKnnClassifier
{
public:
    LineImageKnn( LineImageParams& params, int metricNo,
                 std::vector<ObjLineImages>& trainObjData,
                 std::vector<int>& nTrainPts,
                 float rmseThresh, float knownThresh,
                 float knownWeight, bool showProgress=false )
        :   ObjectKnnClassifier( trainObjData.size(), nTrainPts, showProgress ),
            trainObjData_(trainObjData),
            rmseThresh_(rmseThresh), knownThresh_(knownThresh),
            knownWeight_(knownWeight),
            matcher_( MakeLIMatcher( params, metricNo ) )
    {}

    virtual ~LineImageKnn() {}

    void SetTestObject( ObjLineImages& testData )
    {
        testData_.reset( new ObjLineImages(testData) );
        SetNTest( testData.nPoints );
    }

    //! test point i, train object o, train point j
    inline float Distance(int i, int o, int j)
    {
        ObjLineImages& train = trainObjData_[o];
        ObjLineImages& test = *testData_.get();

        float rmse = 0;
        float known = 0;
        float dist = 0;
        matcher_->match_rmse( test.values.row(i).data(), test.status.row(i).data(),
                            train.values.row(j).data(), train.status.row(j).data(), rmse, known );
        if( rmse > rmseThresh_ || known < knownThresh_ || rmse != rmse || rmse < -10000 )
            dist = std::numeric_limits<float>::max();
        else
            dist = rmse + known * knownWeight_;

        return dist;
    }

    std::vector<ObjLineImages> trainObjData_;
    boost::shared_ptr<ObjLineImages> testData_;
    float rmseThresh_, knownThresh_, knownWeight_;
    boost::shared_ptr<LineImageMatcher> matcher_;
};






/*! Classify points in an object using KNN (with alignment).

Only match points if the aligning vectors have a similar z component (ie, objects
rotate about z axis only).
*/
class LASERLIB_FEATURES_EXPORT LineImageKnnAligned : public ObjectKnnClassifier
{
public:
    LineImageKnnAligned( LineImageParams& params, int metricNo,
                         std::vector<ObjLineImagesAligned>& trainObjData,
                         std::vector<int>& nTrainPts,
                         float alignThresh, float rmseThresh, float knownThresh,
                         float knownWeight, bool showProgress=false )
        :   ObjectKnnClassifier( trainObjData.size(), nTrainPts, showProgress ),
            trainObjData_(trainObjData),
            alignThresh_(alignThresh), rmseThresh_(rmseThresh), knownThresh_(knownThresh),
            knownWeight_(knownWeight),
            matcher_( MakeLIMatcher( params, metricNo ) )
    {}

    virtual ~LineImageKnnAligned() {}

    void SetTestObject( ObjLineImagesAligned& testData )
    {
        testData_.reset( new ObjLineImagesAligned(testData) );
        SetNTest( testData.nPoints );
    }

    //! test point i, train object o, train point j
    inline float Distance(int i, int o, int j)
    {
        ObjLineImagesAligned& train = trainObjData_[o];
        ObjLineImagesAligned& test = *testData_.get();

        if( test.alignType(i) != train.alignType(j) )
            return std::numeric_limits<float>::max();

        //z component of surface normal / linear direction must be within threshold
        if( fabs(test.alignVectors(i,2) - train.alignVectors(j,2)) < alignThresh_ )
        {
            float rmse = 0;
            float known = 0;
            float dist = 0;
            matcher_->match_rmse( test.values.row(i).data(), test.status.row(i).data(),
                                train.values.row(j).data(), train.status.row(j).data(), rmse, known );
            if( rmse > rmseThresh_ || known < knownThresh_ || rmse != rmse || rmse < -10000 )
                dist = std::numeric_limits<float>::max();
            else
                dist = rmse + known * knownWeight_;

            return dist;
        }
        else
            return std::numeric_limits<float>::max();
    }

    std::vector<ObjLineImagesAligned> trainObjData_;
    boost::shared_ptr<ObjLineImagesAligned> testData_;
    float alignThresh_, rmseThresh_, knownThresh_, knownWeight_;
    boost::shared_ptr<LineImageMatcher> matcher_;
};





/*!
For comparing an unknown test object to a set of labelled training objects.
Each object has a set of features. For a given training object:
- For each test feature:
  - Compare it to the training features from the training object.
  - Pick the lowest distance, and add that distance to a histogram.

This results in a histogram of feature distances for each training object.
A good matching object will have lots of low-distance matches.
*/
class LASERLIB_FEATURES_EXPORT ObjectMatchHistogram
{
public:
    ObjectMatchHistogram( LineImageParams& params, int metricNo,
                          std::vector<ObjLineImagesAligned>& trainFData,
                          float alignThresh, float rmseThresh, float knownThresh,
                          float knownWeight, float binWidth, int nBins )
        : params_(params), metricNo_(metricNo),
          trainFData_(trainFData),
          matcher_( MakeLIMatcher( params, metricNo ) ),
          alignThresh_(alignThresh), rmseThresh_(rmseThresh),
          knownThresh_(knownThresh), knownWeight_(knownWeight),
          binWidth_(binWidth), nBins_(nBins)
    {}

    void MatchObject( ObjLineImagesAligned& testData, MapMatXi& hist );

    LineImageParams params_;
    int metricNo_;
    std::vector<ObjLineImagesAligned> trainFData_;
    boost::shared_ptr<LineImageMatcher> matcher_;
    float alignThresh_, rmseThresh_, knownThresh_, knownWeight_;
    float binWidth_;
    int nBins_;
};


#endif //LINE_IMAGE_KNN_HEADER_GUARD
