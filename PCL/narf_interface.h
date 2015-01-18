/*! narf_interface.h
 * Computes pcl range images, NARF keypoints and features. Based on online
 * examples by Bastian Steder.
 *
 * Copyright (C) 2011 Alastair Quadros.
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
 * \date       28-06-2011
*/

#ifndef NARF_INTERFACE
#define NARF_INTERFACE

#include "Common/ArrayTypes.h"
#include <boost/shared_ptr.hpp>

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/range_image/range_image.h>
#include <pcl/keypoints/narf_keypoint.h>

#include "Features/KnnClassifier.h"


boost::shared_ptr<pcl::RangeImage> make_pcl_rangeimage( Mat3<float>::type& P,
            float angular_resolution, bool setUnseenToMaxRange );

void make_narf_keypoints( boost::shared_ptr<pcl::RangeImage> range_image_ptr,
                          pcl::NarfKeypoint::Parameters& params, std::vector<int>& keys );

boost::shared_ptr< pcl::PointCloud< pcl::Narf36 > > make_narf_features( boost::shared_ptr<pcl::RangeImage> range_image_ptr,
            std::vector<int>& keys, float support_size, bool rotation_invariant);




class NarfKnnAligned : public KnnClassifier
{
public:
    NarfKnnAligned( MapMatXf& _test, MapVecXf& _testAlign, MapMatXf& _train, MapVecXf& _trainAlign, float _alignThresh, bool showProgress=false )
        :   KnnClassifier(showProgress),
            test(_test),
            testAlign(_testAlign),
            trainAlign(_trainAlign),
            train(_train),
            alignThresh(_alignThresh),
            length( _test.cols() )
    {}

    //! manhattan distance (test i, train j)
    inline float Distance(int i, int j)
    {
        if( fabs(testAlign(i) - trainAlign(j)) > alignThresh )
            return 1.0;

        float sum = 0.0;
        for( int a=0 ; a<length ; a++ )
        {
            sum += fabs(test(i,a) - train(j,a));
        }
        sum /= length;
        return sum;
    }

    MapMatXf test, train;
    MapVecXf testAlign, trainAlign;

private:
    float alignThresh;
    int length;
};




#endif //NARF_INTERFACE
