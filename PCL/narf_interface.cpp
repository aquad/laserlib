/*! narf_interface.cpp
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

#define EIGEN_DEFAULT_TO_ROW_MAJOR 1

#include "pcl/features/range_image_border_extractor.h"
#include "pcl/keypoints/narf_keypoint.h"
#include "pcl/features/narf_descriptor.h"

#include "narf_interface.h"

using namespace pcl;
typedef PointXYZ PointType;


boost::shared_ptr<pcl::RangeImage> make_pcl_rangeimage( Mat3<float>::type& P,
            float angular_resolution, bool setUnseenToMaxRange )
{
    RangeImage::CoordinateFrame coordinate_frame = RangeImage::LASER_FRAME;

    //input data
    pcl::PointCloud<PointType>::Ptr point_cloud_ptr (new pcl::PointCloud<PointType>);
    pcl::PointCloud<PointType>& point_cloud = *point_cloud_ptr;
    PointCloud<PointWithViewpoint> far_ranges;
    Eigen::Affine3f scene_sensor_pose (Eigen::Affine3f::Identity ());

    point_cloud.points.resize( P.rows() );
    for(int i=0; i<P.rows() ; i++ )
    {
        point_cloud.points[i].x = P(i,0);
        point_cloud.points[i].y = P(i,1);
        point_cloud.points[i].z = P(i,2);
    }
    point_cloud.width = point_cloud.points.size();
    point_cloud.height = 1;

    // -----Create RangeImage from the PointCloud-----
    float noise_level = 0.0;
    float min_range = 0.0f;
    int border_size = 1;
    boost::shared_ptr<RangeImage> range_image_ptr (new RangeImage);
    RangeImage& range_image = *range_image_ptr;
    range_image.createFromPointCloud (point_cloud, angular_resolution, deg2rad (360.0f), deg2rad (180.0f),
                                      scene_sensor_pose, coordinate_frame, noise_level, min_range, border_size);
    range_image.integrateFarRanges (far_ranges);
    if (setUnseenToMaxRange)
        range_image.setUnseenToMaxRange ();
    return range_image_ptr;
}



void make_narf_keypoints( boost::shared_ptr<pcl::RangeImage> range_image_ptr,
                          pcl::NarfKeypoint::Parameters& params, std::vector<int>& keys )
{
    RangeImage& range_image = *range_image_ptr;
    RangeImageBorderExtractor range_image_border_extractor;
    NarfKeypoint narf_keypoint_detector (&range_image_border_extractor);
    narf_keypoint_detector.setRangeImage (&range_image);
    //narf_keypoint_detector.getParameters ().support_size = support_size;
    narf_keypoint_detector.getParameters() = params;

    PointCloud<int> keypoint_indices;
    narf_keypoint_detector.compute(keypoint_indices);
    keys.assign(keypoint_indices.points.begin(), keypoint_indices.points.end());
}



boost::shared_ptr< PointCloud<Narf36> > make_narf_features( boost::shared_ptr<pcl::RangeImage> range_image_ptr,
            std::vector<int>& keys, float support_size, bool rotation_invariant)
{
    RangeImage& range_image = *range_image_ptr;
    NarfDescriptor narf_descriptor (&range_image, &keys);
    narf_descriptor.getParameters ().support_size = support_size;
    narf_descriptor.getParameters ().rotation_invariant = rotation_invariant;
    PointCloud<Narf36>* narf_descriptors_ptr = new PointCloud<Narf36>();
    narf_descriptor.compute( *narf_descriptors_ptr );
    return boost::shared_ptr< PointCloud<Narf36> >(narf_descriptors_ptr);
}


