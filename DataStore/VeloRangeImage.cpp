/*! VeloRangeImage.cpp
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
 * \date       10-03-2011
*/

#include "VeloRangeImage.h"
#include <utility>
#include <vector>
#include <map>
#include <math.h>
#include <algorithm>
#include <iostream>

#include "boost/math/constants/constants.hpp"
const double pi = boost::math::constants::pi<double>();


VeloRangeImage::VeloRangeImage(int _xRes, int _maxPointsPerPixel, VelodyneDb& _db)
    :   db(_db),
        xRes(_xRes),
        maxPointsPerPixel(_maxPointsPerPixel),
        image( new int[64*xRes*maxPointsPerPixel] ),
        nPointsInPixel( new int[64*xRes] )
{
    memset(nPointsInPixel, 0, 64*xRes * sizeof(int));
    memset(image, -1, 64*xRes*maxPointsPerPixel * sizeof(int));

    //operations on db to create convenient vectors etc.
    std::map<double,int> vcToIdMap; //use a map to argsort by vc
    for( int i=0 ; i<64 ; i++ )
        vcToIdMap.insert( std::pair<double,int>(db.vc[i], i) );

    sortedVc.resize(64);
    std::map<double,int>::iterator it = vcToIdMap.begin();
    for( int i=0 ; i<64 ; i++ )
    {
        sortedVc[i] = (*it).first;
        idToLineNo[(*it).second] = i;
        ++it;
    }

    //determine image cell boundaries. elBounds[0] is the elevation of the top boundary of the image.
    elBounds.resize(65);
    elBounds[0] = sortedVc[0] - 0.01;
    elBounds[64] = sortedVc[63] + 0.01;
    for( int i=1 ; i<64 ; i++ )
    {
        elBounds[i] = sortedVc[i-1] + (sortedVc[i] - sortedVc[i-1])/2;
    }
}


VeloRangeImage::VeloRangeImage( const VeloRangeImage& other )
    : db(other.db),
      xRes(other.xRes),
      maxPointsPerPixel(other.maxPointsPerPixel),
      image( new int[64*xRes*maxPointsPerPixel] ),
      nPointsInPixel( new int[64*xRes] ),
      sortedVc( other.sortedVc ),
      elBounds( other.elBounds )
{
    memcpy( nPointsInPixel, other.nPointsInPixel, 64*xRes * sizeof(int) );
    memcpy( image, other.image, 64*xRes*maxPointsPerPixel * sizeof(int) );
    memcpy( idToLineNo, other.idToLineNo, 64*sizeof(int) );
}


VeloRangeImage::~VeloRangeImage()
{
    delete[] image;
    delete[] nPointsInPixel;
}


//w 0-35999, id 0-63
void VeloRangeImage::AddPoints( unsigned short const* w, unsigned char const* id, int nPoints )
{
    for( int i=0 ; i<nPoints ; i++ )
    {
        int x = w[i] * xRes / 36000;
        if( x>=xRes )
        {
            std::cerr << "w element " << i << " out of range" << std::endl;
            continue;
        }
        int y = idToLineNo[id[i]];
        int index = GetIndex(x,y);
        int* nPointsHere = nPointsInPixel + y * xRes + x;
        if( *nPointsHere >= maxPointsPerPixel )
            continue;
        int* pos = image + index + *nPointsHere;
        *pos = i;
        (*nPointsHere)++;
    }
}

//w 0-2pi, id 0-63
void VeloRangeImage::AddPoints( double const* w, unsigned char const* id, int nPoints )
{
    for( int i=0 ; i<nPoints ; i++ )
    {
        int x = (int)(w[i] * xRes / (2*pi));
        if( x>=xRes )
        {
            std::cerr << "w element " << i << " out of range" << std::endl;
            continue;
        }
        int y = idToLineNo[id[i]];
        int index = GetIndex(x,y);
        int* nPointsHere = nPointsInPixel + y * xRes + x;
        if( *nPointsHere >= maxPointsPerPixel )
            continue;
        int* pos = image + index + *nPointsHere;
        *pos = i;
        (*nPointsHere)++;
    }
}


//w 0-2pi, el -pi-pi
void VeloRangeImage::AddPoints( double const* w, double const* el, int nPoints )
{
    for( int i=0 ; i<nPoints ; i++ )
    {
        int x = (int)(w[i] * xRes / (2*pi));
        if( x>=xRes )
        {
            std::cerr << "w element " << i << " out of range" << std::endl;
            continue;
        }
        if( el[i] >= pi || el[i] <= -pi )
        {
            std::cerr << "el element " << i << " out of range" << std::endl;
            continue;
        }
        int y = FindElevationPixel(el[i]);
        int index = GetIndex(x,y);
        int* nPointsHere = nPointsInPixel + y * xRes + x;
        if( *nPointsHere >= maxPointsPerPixel )
            continue;
        int* pos = image + index + *nPointsHere;
        *pos = i;
        (*nPointsHere)++;
    }
}


void VeloRangeImage::Clear()
{
    memset(nPointsInPixel, 0, 64*xRes*sizeof(int));
    memset(image, -1, 64*xRes*maxPointsPerPixel * sizeof(int));
}


/*
void VeloRangeImage::GetNearestPixel(double az, double el, int& x, int& y)
{
    //search for closest elevation
    std::vector<double>::iterator it = std::lower_bound(sortedVc.begin(), sortedVc.end(), el);
    if( it==sortedVc.begin() )
        y=0;
    else if( it==sortedVc.end() )
        y=63;
    else
    {
        std::vector<double>::iterator beforeIt = it;
        beforeIt--;
        if( abs((*it) - el) < abs((*beforeIt--) - el) )
        {
            y = (int)(it - sortedVc.begin());
        }
        else
        {
            y = (int)(beforeIt - sortedVc.begin());
        }
    }
    x = BoundAnglePositive(az) / (2*pi) * xRes;
}
*/


void VeloRangeImage::GetNearestPixel(double az, double el, int& x, int& y)
{
    //search for closest elevation
    std::vector<double>::iterator it = std::lower_bound(elBounds.begin(), elBounds.end(), el);
    if( it==elBounds.begin() )
        y=0;
    else if( it==elBounds.end() )
        y=63;
    else
        y = (int)(it - elBounds.begin()) - 1;

    x = BoundAnglePositive(az) / (2*pi) * xRes;
}



void VeloRangeImage::GetNearestPixelsPoints(double az, double el, int*& result, int& nPoints)
{
    int x,y;
    GetNearestPixel(az, el, x, y);
    GetPoints(x, y, result, nPoints);
}


void VeloRangeImage::GetPoints(int x, int y, int*& result, int& nPoints)
{
    if( x<0 || x>xRes || y<0 || y>63 ) //allow out of bounds calls
    {
        nPoints = 0;
        return;
    }
    int index = GetIndex(x,y);
    result = image + index;
    nPoints = nPointsInPixel[y*xRes + x];
}


inline int VeloRangeImage::GetIndex(int x, int y)
{
    return y * xRes * maxPointsPerPixel + x * maxPointsPerPixel;
}



int VeloRangeImage::FindElevationPixel(double el)
{
    int pixel;
    std::vector<double>::iterator it = std::lower_bound(elBounds.begin(), elBounds.end(), el);
    if( it==elBounds.begin() )
        pixel = -1;
    else if( it==elBounds.end() )
        pixel = 64;
    else
    {
        it--;
        pixel = (int)(it - elBounds.begin());
    }
    return pixel;
}


int VeloRangeImage::FindAzimuthPixel(double az)
{
    return BoundAnglePositive(az) / (2*pi) * xRes;
}



void VeloRangeImage::Point3dToRangeImage( double* p3d, double* p2d )
{
    //vc (elevation angle)
    p2d[1] = atan2( p3d[2] - db.voffc[32], sqrt(p3d[0]*p3d[0] + p3d[1]*p3d[1]) );
    if( p2d[1] < elBounds[32] )
        p2d[1] = atan2( p3d[2] - db.voffc[0], sqrt(p3d[0]*p3d[0] + p3d[1]*p3d[1]) );
    //w (azimuth angle)
    p2d[0] = BoundAnglePositive( atan2(p3d[1], p3d[0]) );
}

void VeloRangeImage::Point3dToRangeImage( float* p3d, float* p2d )
{
    //vc (elevation angle)
    p2d[1] = atan2( p3d[2] - db.voffc[32], sqrt(p3d[0]*p3d[0] + p3d[1]*p3d[1]) );
    if( p2d[1] < elBounds[32] )
        p2d[1] = atan2( p3d[2] - db.voffc[0], sqrt(p3d[0]*p3d[0] + p3d[1]*p3d[1]) );
    //w (azimuth angle)
    p2d[0] = BoundAnglePositive( atan2(p3d[1], p3d[0]) );
}



inline double VeloRangeImage::BoundAnglePositive(double angle)
{
    if( angle < 0 )
        angle += 2*pi;
    else if( angle > 2*pi )
        angle -= 2*pi;
    return angle;
}

