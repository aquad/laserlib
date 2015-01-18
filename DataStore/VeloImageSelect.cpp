/*! VeloImageSelect.cpp
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

#include "VeloImageSelect.h"
#include <iostream>
#include <cmath>

using namespace Eigen;

ImagePlusSelector::ImagePlusSelector( VeloRangeImage& _image, VelodyneDb& _db,
                                     double* _w, unsigned char* _id, double* _D, double _rad )
    :   image(_image), db(_db), w(_w), id(_id), D(_D), rad(_rad)
{
    neigh.reserve(300);
}


std::vector<int>& ImagePlusSelector::SelectRegion( unsigned int centre )
{
    //calc image borders
    double az1, el1, az2, el2;
    double elCentre = db.vc[id[centre]];
    double angle = atan2( rad, D[centre] );
    az1 = w[centre] - angle;
    az2 = w[centre] + angle;
    el1 = elCentre - angle;
    el2 = elCentre + angle;

    int x1,y1,x2,y2;
    image.GetNearestPixel(az1, el1, x1, y1);
    image.GetNearestPixel(az2, el2, x2, y2);

    //add one pixel on each side, allowing us to interpolate within the specified radius etc.
    x1 -= 1;
    y1 -= 1;
    x2 += 1;
    y2 += 1;
    if( x1 < 0 )
        x1 = image.xRes-1;
    if( y1 < 0 )
        y1 = 0;
    if( x2 >= image.xRes )
        x2 -= image.xRes;
    if( y2 > 63 )
        y2 = 63;

    neigh.clear();

    if( x2 < x1 ) //wraparound needed
    {
        for( int y=y1 ; y<=y2 ; y++ )
        {
            int indexStart = y * image.xRes * image.maxPointsPerPixel + x1 * image.maxPointsPerPixel;
            int indexEnd = y * image.xRes * image.maxPointsPerPixel + (image.xRes-1) * image.maxPointsPerPixel;
            SelectPointsIndexRange( indexStart, indexEnd );

            indexStart = y * image.xRes * image.maxPointsPerPixel;
            indexEnd = y * image.xRes * image.maxPointsPerPixel + x2 * image.maxPointsPerPixel;
            SelectPointsIndexRange( indexStart, indexEnd );
        }
    }
    else
    {
        for( int y=y1 ; y<=y2 ; y++ )
        {
            int indexStart = y * image.xRes * image.maxPointsPerPixel + x1 * image.maxPointsPerPixel;
            int indexEnd = y * image.xRes * image.maxPointsPerPixel + x2 * image.maxPointsPerPixel;
            SelectPointsIndexRange( indexStart, indexEnd );
        }
    }
    return neigh;
}



std::vector<int>& ImagePlusSelector::SelectRegionPoint( Eigen::Vector3d& point )
{
    Eigen::Vector2d p2d;
    image.Point3dToRangeImage( point.data(), p2d.data() );
    double azimuth = p2d[0];
    double elevation = p2d[1];
    double distance = point.norm(); //approx- don't know vc

    //calc image borders
    double az1, el1, az2, el2;
    double angle = atan2( rad, distance );
    az1 = azimuth - angle;
    az2 = azimuth + angle;
    el1 = elevation - angle;
    el2 = elevation + angle;

    int x1,y1,x2,y2;
    image.GetNearestPixel(az1, el1, x1, y1);
    image.GetNearestPixel(az2, el2, x2, y2);

    //add one pixel on each side, allowing us to interpolate within the specified radius etc.
    x1 -= 1;
    y1 -= 1;
    x2 += 1;
    y2 += 1;
    if( x1 < 0 )
        x1 = image.xRes-1;
    if( y1 < 0 )
        y1 = 0;
    if( x2 >= image.xRes )
        x2 -= image.xRes;
    if( y2 > 63 )
        y2 = 63;

    neigh.clear();

    if( x2 < x1 ) //wraparound needed
    {
        for( int y=y1 ; y<=y2 ; y++ )
        {
            int indexStart = y * image.xRes * image.maxPointsPerPixel + x1 * image.maxPointsPerPixel;
            int indexEnd = y * image.xRes * image.maxPointsPerPixel + (image.xRes-1) * image.maxPointsPerPixel;
            SelectPointsIndexRange( indexStart, indexEnd );

            indexStart = y * image.xRes * image.maxPointsPerPixel;
            indexEnd = y * image.xRes * image.maxPointsPerPixel + x2 * image.maxPointsPerPixel;
            SelectPointsIndexRange( indexStart, indexEnd );
        }
    }
    else
    {
        for( int y=y1 ; y<=y2 ; y++ )
        {
            int indexStart = y * image.xRes * image.maxPointsPerPixel + x1 * image.maxPointsPerPixel;
            int indexEnd = y * image.xRes * image.maxPointsPerPixel + x2 * image.maxPointsPerPixel;
            SelectPointsIndexRange( indexStart, indexEnd );
        }
    }
    return neigh;
}




//helper function
inline void ImagePlusSelector::SelectPointsIndexRange( int indexStart, int indexEnd )
{
    //iterate along row section
    for( int* imgPos = image.image + indexStart ; imgPos <= image.image + indexEnd ; imgPos++ )
    {
        if( *imgPos!=-1 )
            neigh.push_back(*imgPos);
    }
}




//-----------------------------


ImageSphereSelector::ImageSphereSelector( VeloRangeImage& _image, VelodyneDb& _db,
                                     double* _w, unsigned char* _id, double* _D, Mat3<double>::type& _P, double _rad )
    :   image(_image), db(_db), w(_w), id(_id), D(_D), P(_P), rad(_rad), radSqr(_rad*_rad),
        PImage( _image.xRes * 64 * _image.maxPointsPerPixel, 3 )
{
    neigh.reserve(300);
    for( int i = 0 ; i < image.xRes * 64 * image.maxPointsPerPixel; i++ )
    {
        if( image.image[i]!=-1 )
            PImage.row(i) = P.row( image.image[i] );
    }
}


std::vector<int>& ImageSphereSelector::SelectRegion( unsigned int centre )
{
    //calc image borders
    double az1, el1, az2, el2;
    double elCentre = db.vc[id[centre]];
    double angle = asin( rad / D[centre] );
    az1 = w[centre] - angle;
    az2 = w[centre] + angle;
    el1 = elCentre - angle;
    el2 = elCentre + angle;
    RowVector3d Pcentre = P.row(centre);

    int x1,y1,x2,y2;
    image.GetNearestPixel(az1, el1, x1, y1);
    image.GetNearestPixel(az2, el2, x2, y2);

    neigh.clear();

    if( x2 < x1 ) //wraparound needed
    {
        for( int y=y1 ; y<=y2 ; y++ )
        {
            int indexStart = y * image.xRes * image.maxPointsPerPixel + x1 * image.maxPointsPerPixel;
            int indexEnd = y * image.xRes * image.maxPointsPerPixel + (image.xRes-1) * image.maxPointsPerPixel;
            SelectPointsIndexRange( indexStart, indexEnd, Pcentre );

            indexStart = y * image.xRes * image.maxPointsPerPixel;
            indexEnd = y * image.xRes * image.maxPointsPerPixel + x2 * image.maxPointsPerPixel;
            SelectPointsIndexRange( indexStart, indexEnd, Pcentre );
        }
    }
    else
    {
        for( int y=y1 ; y<=y2 ; y++ )
        {
            int indexStart = y * image.xRes * image.maxPointsPerPixel + x1 * image.maxPointsPerPixel;
            int indexEnd = y * image.xRes * image.maxPointsPerPixel + x2 * image.maxPointsPerPixel;
            SelectPointsIndexRange( indexStart, indexEnd, Pcentre );
        }
    }
    return neigh;
}





//-----------------------------

void RangeImageMatch( VeloRangeImage& image, MapVecXll& t, Mat3<double>::type P,
        MapVecXuc& srcLid, MapVecXd& srcW, MapVecXll& srcT, Mat3<double>::type srcP,
        MapVecXi& matches )
{
    for( int i=0 ; i<srcLid.size() ; i++ )
    {
        int match = -1;
        // defines max distance in time/distance we accept matching.
        long long int bestTime = 10;
        double bestDist = 0.2;

        int xi = image.FindAzimuthPixel( srcW[i] );
        int yi = image.idToLineNo[ srcLid[i] ];
        //checking neighbouring pixels too, due to floating point errors
        //int startx = (xi-1)%image.xRes;
        //int endx = (xi+1)%image.xRes;
        for( int xOffset=-1 ; xOffset <= 1 ; xOffset++ )
        {
            int x = (xi + xOffset)%image.xRes;
            int* points;
            int nPoints;
            image.GetPoints( x, yi, points, nPoints);

            for( int j=0 ; j<nPoints ; j++ )
            {
                int pid = points[j];
                // Because of stupid inconsistent timestamps, can't just get
                // the equal ones. Find closest in time, if there are
                // duplicates, find closest in 3D.
                long long int tDiff = abs(t[pid] - srcT[i]);
                if( tDiff < bestTime )
                {
                    // time match, trumps distance matching.
                    match = pid;
                    bestTime = tDiff;
                    // store distance, in case there are duplicates later.
                    // (check for max??)
                    RowVector3d rel = P.row(pid) - srcP.row(i);
                    bestDist = rel.squaredNorm();
                }
                else if( tDiff == bestTime && match != -1 )
                {
                    // duplicate laser id + timestamp- resolve with 3D distance.
                    RowVector3d rel = P.row(pid) - srcP.row(i);
                    double dist = rel.squaredNorm();
                    if( dist < bestDist )
                    {
                        match = pid;
                        bestDist = dist;
                        //bestTime is the same
                    }
                }
            }
        }
        matches[i] = match;
    }
}
