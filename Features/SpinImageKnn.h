/*! SpinImageKnn.h
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
 * \date       12-01-2013
*/

#ifndef SPIN_IMAGE_KNN_HEADER_GUARD
#define SPIN_IMAGE_KNN_HEADER_GUARD


//Eigen
#define EIGEN_DEFAULT_TO_ROW_MAJOR 1
#include <Eigen/Core>

#include "SpinImage.h"
#include "KnnClassifier.h"
#include <vector>
#include <boost/shared_ptr.hpp>



// These operate on a single large array of concatenated spin images, with no notion of objects.

class SpinSetKnn : public KnnClassifier
{
public:
    SpinSetKnn( MapMatXf& test, MapMatXf& train, float lamb, SpinMetric metric, bool showProgress=false )
        :   KnnClassifier(showProgress),
            test_(test), train_(train),
            lamb_(lamb),
            metric_(metric),
            nCells_(test.cols())
    {}

    inline float Distance(int i, int j)
    {
        return SpinDistance( test_.row(i).data(), train_.row(j).data(), nCells_, lamb_, metric_ );
    }

protected:
    MapMatXf test_;
    MapMatXf train_;
    float lamb_;
    SpinMetric metric_;
    int nCells_;
};




class SpinSetKnnAligned : public KnnClassifier
{
public:
    SpinSetKnnAligned( MapMatXf& test, Vect<float>::type& test_align,
                       MapMatXf& train, Vect<float>::type& train_align,
                       float thresh, SpinMetric metric, bool showProgress=false )
        :   KnnClassifier(showProgress),
            test_(test), test_align_(test_align),
            train_(train), train_align_(train_align),
            thresh_(thresh), metric_(metric),
            nCells_(test.cols())
    {}

    inline float Distance(int i, int j)
    {
        if( fabs(test_align_(i) - test_align_(j)) < thresh_ )
        {
            return SpinDistance( test_.row(i).data(), train_.row(j).data(), nCells_, lamb_, metric_ );
        }
        else
            return std::numeric_limits<float>::max();
    }

protected:
    MapMatXf test_;
    Vect<float>::type test_align_;
    MapMatXf train_;
    Vect<float>::type train_align_;
    float lamb_;
    SpinMetric metric_;
    float thresh_;
    int nCells_;
};




//------------------------------------------

// These operate on objects, each with their own array of spin images



/*! Classify points in an object using KNN.
*/
class SpinImageKnn : public ObjectKnnClassifier
{
public:
    SpinImageKnn( std::vector<MapMatXf>& trainObjData,
                  std::vector<int>& nTrainPts, SpinMetric metric,
                  float lamb = 0, bool showProgress = false )
        :   ObjectKnnClassifier( trainObjData.size(), nTrainPts, showProgress ),
            trainObjData_(trainObjData),
            metric_( metric ),
            lamb_( lamb ),
            nCells_( trainObjData[0].cols() )
    {}

    virtual ~SpinImageKnn() {}

    void SetTestObject( MapMatXf& testData )
    {
        testData_.reset( new MapMatXf(testData) );
        SetNTest( testData_->rows() );
    }

    //! test point i, train object o, train point j
    inline float Distance(int i, int o, int j)
    {
        MapMatXf& train = trainObjData_[o];
        MapMatXf& test = *testData_.get();
        return SpinDistance( test.row(i).data(), train.row(j).data(), nCells_, lamb_, metric_ );
    }

protected:
    std::vector<MapMatXf> trainObjData_;
    boost::shared_ptr<MapMatXf> testData_;
    SpinMetric metric_;
    float lamb_;
    int nCells_;
};




//! Spin images and aligning vectors in an object
class ObjSpinImagesAligned
{
public:
    ObjSpinImagesAligned( Mat3<float>::type& alignVectors_,
                          Vect<unsigned char>::type& alignType_,
                          MapMatXf& images_, int nPoints_ )
        : alignVectors(alignVectors_),
          alignType(alignType_),
          images(images_),
          nPoints(nPoints_)
    {}

    Mat3<float>::type alignVectors; //!< Surface normal or linear vector
    Vect<unsigned char>::type alignType; //!< Whether alignVectors is a surface normal (0) or a linear vector (1)
    MapMatXf images; //!< Spin images, one (flattened image) per row
    int nPoints;
};




/*! Classify points in an object using KNN (with alignment).

Only match points if the aligning vectors have a similar z component (ie, objects
rotate about z axis only).
*/
class SpinImageKnnAligned : public ObjectKnnClassifier
{
public:
    SpinImageKnnAligned( std::vector<ObjSpinImagesAligned>& trainObjData,
                         std::vector<int>& nTrainPts, SpinMetric metric,
                         float alignThresh, float lamb = 0,
                         bool showProgress = false )
        :   ObjectKnnClassifier( trainObjData.size(), nTrainPts, showProgress ),
            trainObjData_(trainObjData),
            metric_( metric ),
            alignThresh_( alignThresh ),
            lamb_( lamb ),
            nCells_( trainObjData[0].images.cols() )
    {}

    virtual ~SpinImageKnnAligned() {}

    void SetTestObject( ObjSpinImagesAligned& testData )
    {
        testData_.reset( new ObjSpinImagesAligned(testData) );
        SetNTest( testData.nPoints );
    }

    //! test point i, train object o, train point j
    inline float Distance(int i, int o, int j)
    {
        ObjSpinImagesAligned& train = trainObjData_[o];
        ObjSpinImagesAligned& test = *testData_.get();
        float dist;

        if( test.alignType(i) != train.alignType(j) )
            dist = std::numeric_limits<float>::max();

        // z component of surface normal / linear direction must be within threshold
        else if( fabs(test.alignVectors(i,2) - train.alignVectors(j,2)) < alignThresh_ )
        {
            return SpinDistance( test.images.row(i).data(), train.images.row(j).data(), nCells_, lamb_, metric_ );
        }
        else
            dist = std::numeric_limits<float>::max();

        return dist;
    }

protected:
    std::vector<ObjSpinImagesAligned> trainObjData_;
    boost::shared_ptr<ObjSpinImagesAligned> testData_;
    SpinMetric metric_;
    float alignThresh_;
    float lamb_;
    int nCells_;
};



#endif //SPIN_IMAGE_KNN_HEADER_GUARD
