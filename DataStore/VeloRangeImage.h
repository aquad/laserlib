/*! VeloRangeImage.h
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
 *
 * \details
 * An image (2d grid) where each cell refers to a part of the range image
 * (elevation and azimuth coordinates).  Each cell contains a set of point
 * indexes, referring to a 3d point array / other aligned data arrays.
 *
 * \author  Alastair Quadros
 * \date    10-03-2011
*/

#ifndef VELO_RANGE_IMAGE
#define VELO_RANGE_IMAGE

#include <vector>
#include "VelodyneDb.h"

/*!
Store the index values in an image. Each pixel contains several points.
The image goes vertically from db.vc.min() to db.vc.max(), horizontally from 0 to 2*pi.

Common parameters:
\param az - azimuth (rad)
\param el - elevation (rad)
\param result - pointer to an array of indices, referring to 3D points
*/
class VeloRangeImage
{
public:
    VeloRangeImage(int xRes, int maxPointsPerPixel, VelodyneDb& db);
    VeloRangeImage( const VeloRangeImage& );
    virtual ~VeloRangeImage();

    /*! \param w - an array of azimuth angles, degrees * 500, direct from the velodyne (post-calibration-offset)
        \param id - an array of laser ids [0-63]
    */
    void AddPoints(unsigned short const* w, unsigned char const* id, int nPoints);

    /*! \param w - an array of azimuth angles (rad)
        \param id - an array of laser ids [0-63]
    */
    void AddPoints(double const* w, unsigned char const* id, int nPoints);

    /*! \param w - an array of azimuth angles (rad)
        \param id - an array of elevation angles (rad)
    */
    void AddPoints(double const* w, double const* el, int nPoints);

    void Clear();

    void GetNearestPixel(double az, double el, int& x, int& y);

    //! Get points in the nearest pixel.
    void GetNearestPixelsPoints(double az, double el, int*& result, int& nPoints);

    //! Get points in the specified pixel.
    void GetPoints(int x, int y, int*& result, int& nPoints);

    //! Find the integer y coordinate corresponding to the elevation angle (rad).
    int FindElevationPixel(double el);

    //! Find the integer x coordinate corresponding to the azimuth angle (rad).
    int FindAzimuthPixel(double az);

    //@{
    /*! Convert a 3D point to 2D azimuth, elevation coordinates
        \param p3d - a 3D point (input)
        \param p2d - a 2D point (output)
    */
    void Point3dToRangeImage( double* p3d, double* p2d );
    void Point3dToRangeImage( float* p3d, float* p2d );
    //@}

    //! Adjust angle to be [0,2*pi)
    double BoundAnglePositive(double angle);

//private:
    //! Get the index of x,y coordinates (pointer offset to image array)
    int GetIndex(int x, int y);

    VelodyneDb db;
    int xRes;
    int maxPointsPerPixel;
    int* image; //!< All data is stored here, size xRes * 64 * maxPointsPerPixel
    int* nPointsInPixel; //!< The number of points stored in each pixel, size xRes * 64
    int idToLineNo[64]; //!< Maps laser id to line number (sorted top to bottom).
    std::vector<double> sortedVc; //!< Elevation of the 64 lasers, sorted.
    std::vector<double> elBounds; //!< Elevation boundaries between each laser.
};


#endif //VELO_RANGE_IMAGE
