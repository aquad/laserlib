/*! KnnClassifier.h
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
 * \detail
 * Abstract class for KNN classification. Derivated classes provide the
 * appropriate distance function.  'test' and 'training' have a feature vector
 * per row. 'matches' has the same number of rows as test, with k columns,
 * storing the indices of the k nearest neighbours. 'values' has the associated
 * distance/similarity values.
 *
 * \author     Alastair Quadros
 * \date       06-12-2010
*/

#ifndef KNN_CLASSIFIER_HEADER_GUARD
#define KNN_CLASSIFIER_HEADER_GUARD

#include "export.h"
//Eigen
#define EIGEN_DEFAULT_TO_ROW_MAJOR 1
#include <Eigen/Core>
#include <vector>
#include "Common/ArrayTypes.h"
#include <boost/date_time/posix_time/posix_time.hpp>


//! Base class, provide your own compare function.
class LASERLIB_FEATURES_EXPORT KnnClassifier
{
public:
    KnnClassifier( bool showProgress=false, int nThreads=1 )
        :   showProgress_(showProgress),
            nThreads_(nThreads)
    {}

    virtual ~KnnClassifier() {}

    //! Provide the top k matches for each test point. matches/values: shape (testpoints, k)
    virtual void Classify( int nTest, int nTraining, MapMatXi& matches, MapMatXf& values );

    //! matches, values: shape (testIds.size(), knn). match ids refer to full length ids.
    virtual void ClassifyKeys( Vect<int>::type& testIds, Vect<int>::type& trainIds,
        MapMatXi& matches, MapMatXf& values );

    //! compare two feature vectors (test i, train j)
    virtual float Distance(int i, int j) = 0;

private:
    bool showProgress_;
    int nThreads_;
};



/* Due to multiple inheritance issues, first define this interface class.
 */
class LASERLIB_FEATURES_EXPORT ObjectKnnClassifierInterface
{
public:
    ObjectKnnClassifierInterface() {}
    virtual ~ObjectKnnClassifierInterface() {}

    //! For a given object, provide the top k matches for each point. matches/values: shape (testpoints, k)
    virtual void Classify( MapMatXi& objectMatches, MapMatXi& pointMatches, MapMatXf& values ) = 0;

    //! compare test point i with train object o, point j
    virtual float Distance(int i, int o, int j) = 0;

    //! The number of test points must be set prior to calling Classify()
    virtual void SetNTest(int nTest) = 0;
    virtual int GetNTest() = 0;

    //! Progress indicator
    virtual bool GetShowProgress() = 0;
    virtual void SetShowProgress(bool show) = 0;
    boost::posix_time::time_duration computeTime;
};



/*! Rather than concatenate all training objects into one big array, provide the feature data
for each object in a vector of arrays.
This is still a virtual base class- define your own distance measure, and add
functions/members which store the relevant training/testing data.
*/
class LASERLIB_FEATURES_EXPORT ObjectKnnClassifier : virtual public ObjectKnnClassifierInterface
{
public:
    ObjectKnnClassifier( int nTrainObjs, std::vector<int>& nTrainPts, bool showProgress=false )
        : nTrainObjs_(nTrainObjs),
          nTrainPts_(nTrainPts),
          nTestPts_(0),
          showProgress_(showProgress)
    {
        nTrainTotal_ = 0;
        for( int i=0 ; i<nTrainPts.size() ; i++ )
            { nTrainTotal_ +=nTrainPts[i]; }
        //temp allocation
        allVals_.resize( nTrainTotal_ );
        argSorted_.resize( nTrainTotal_ );
    }

    virtual ~ObjectKnnClassifier() {}

    //! For a given object, provide the top k matches for each point. matches/values: shape (testpoints, k)
    void Classify( MapMatXi& objectMatches, MapMatXi& pointMatches, MapMatXf& values );

    //! The number of test points must be set prior to calling Classify()
    void SetNTest(int nTest) { nTestPts_ = nTest; }
    int GetNTest() { return nTestPts_; }

    bool GetShowProgress() { return showProgress_; }
    void SetShowProgress(bool show) { showProgress_ = show; }

private:
    bool showProgress_;
    int nTrainObjs_;
    std::vector<int> nTrainPts_;
    int nTestPts_; //!< This must be set prior to calling Classify()
    int nTrainTotal_;

    //temp storage for all comparison values of current point
    std::vector<float> allVals_;
    std::vector<int> argSorted_;
};



#endif //KNN_CLASSIFIER_HEADER_GUARD
