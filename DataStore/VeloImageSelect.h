/*! VeloImageSelect.h
 *
 * Functions that use the range image for selecting regions.
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
 * \date       20-05-2011
*/

#ifndef VELO_IMAGE_SELECT
#define VELO_IMAGE_SELECT

#include <vector>
#include "Selector.h"
#include "VeloRangeImage.h"
#include "Common/ArrayTypes.h"
#include "VelodyneDb.h"
#include <boost/shared_ptr.hpp>

/*!
Select all points within a rectange of the range image.

The dimensions determined by the specified radius and the
centre point's depth. An extra pixel on each side is added.
This can assist in selecting a 3d region plus it's
surrounding points (for occupancy / interpolation).
*/
class ImagePlusSelector : public virtual Selector
{
public:
    /*! \param w - an array of azimuth angles, rad, dtype=float64
        \param id - an array of laser ids [0-63], dtype=uint8
        \param D - an array of depth, m, dtype=float64
        \param rad - radius of selection, m, double
    */
    ImagePlusSelector( VeloRangeImage& image, VelodyneDb& db, double* w, unsigned char* id, double* D, double rad );
    std::vector<int>& SelectRegion( unsigned int centre );

    boost::shared_ptr<Selector> clone()
    {
        return boost::shared_ptr<Selector>( new ImagePlusSelector(*this) );
    }

    std::vector<int>& SelectRegionPoint( Eigen::Vector3d& point );
    void SetRadius( double _rad ) { rad = _rad; }

private:
    inline void SelectPointsIndexRange( int indexStart, int indexEnd );

    VeloRangeImage& image;
    VelodyneDb db;
    double* w;
    unsigned char* id;
    double* D;
    double rad;
    std::vector<int> neigh;
};



/*!
Select all points within a 3d sphere, using the range image.

This method is (probably) faster than a KDTree. First a rectangular region in the range image is defined,
based on the specified radius and the depth of the centre point. All points in this region are
checked for 3d distance.
*/
class ImageSphereSelector : public virtual Selector
{
public:
    /*! \param w - an array of azimuth angles, rad, dtype=float64
        \param id - an array of laser ids [0-63], dtype=uint8
        \param D - an array of depth, m, dtype=float64
        \param P - (n,3) array of points, dtype=float64
        \param rad - radius of selection, m, double
    */
    ImageSphereSelector( VeloRangeImage& image, VelodyneDb& db, double* w, unsigned char* id, double* D, Mat3<double>::type& P, double rad );
    std::vector<int>& SelectRegion( unsigned int centre );

    boost::shared_ptr<Selector> clone()
    {
        return boost::shared_ptr<Selector>( new ImageSphereSelector(*this) );
    }

    void SetRadius( double _rad )
    {
        rad = _rad;
        radSqr = _rad*_rad;
    }

private:
    void SelectPointsIndexRange( int indexStart, int indexEnd, Eigen::RowVector3d& Pcentre );

    VeloRangeImage& image;
    VelodyneDb db;
    double* w;
    unsigned char* id;
    double* D;
    MatX3d PImage; // internally stored points, aligned to image (not original)
    Mat3<double>::type P; // externally stored points, original order
    double rad, radSqr;
    std::vector<int> neigh;
};


//helper function
inline void ImageSphereSelector::SelectPointsIndexRange( int indexStart, int indexEnd, Eigen::RowVector3d& Pcentre )
{
    //iterate along row section
    for( int i = indexStart ; i <= indexEnd ; i++ )
    {
        int pid = image.image[i];
        if( pid!=-1 )
        {
            Eigen::RowVector3d rel = PImage.row(i) - Pcentre;
            if( rel.squaredNorm() < radSqr )
                neigh.push_back(pid);
        }
    }
}





/*! Find the point ids in the range image that match a subset 'source'.
  Could just use a KDTree... hopefully this is faster.
*/
void RangeImageMatch( VeloRangeImage& image, MapVecXll& t, Mat3<double>::type P,
        MapVecXuc& srcLid, MapVecXd& srcW, MapVecXll& srcT, Mat3<double>::type srcP,
        MapVecXi& matches );


#endif //VELO_IMAGE_SELECT
