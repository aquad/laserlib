/*! LineImage.cpp
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
 * \date       08-02-2011
*/


#include "LineImage.h"
#include <boost/math/constants/constants.hpp>
const double pi = boost::math::constants::pi<double>();

using namespace Eigen;

ComputeLineImage::ComputeLineImage( LineImageParams& _params,
        Mat3<double>::type& _P, Vect<unsigned char>::type& _id, Vect<double>::type& _D,
        Vect<double>::type& _w, PCAResults& _pcaResults, VelodyneDb& _db,
        VeloRangeImage& _image )

    :   nPoints(_P.rows()),
        params(_params),
        localLinesEnd(params.nLines, 3),
        localLinesStart(params.nLines, 3),
        linesStart(params.nLines, 3),
        linesEnd(params.nLines, 3),
        occLine(params, _P, _id, _D, _w, _pcaResults, _db, _image)
{
    //set up local line coordinates
    unsigned int count = 0;
    for( unsigned int i=0 ; i<params.nRadSections ; i++ )
    {
        double radius = params.diskRad / params.nRadSections * (i+1);
        for( unsigned int j=0 ; j<params.angularSections[i] ; j++ )
        {
            double angle = 2*pi / params.angularSections[i] * j;
            localLinesEnd(count,0) = radius * cos(angle);
            localLinesEnd(count,1) = radius * sin(angle);
            double height = sqrt( params.regionRad*params.regionRad - radius*radius );
            localLinesEnd(count,2) = height;
            count +=1;
        }
    }
    localLinesStart = localLinesEnd;
    localLinesStart.col(2) *= -1;
}



ComputeLineImage::~ComputeLineImage()
{}


void ComputeLineImage::setObjectMask( std::vector<bool>& mask )
{
    occLine.setObjectMask(mask);
}


void ComputeLineImage::compute( const Map<Matrix3f>& R, const Vector3d& point, float* values, unsigned char* status )
{
    //lines start behind the surface
    RowVector3f pointT = point.cast<float>().transpose();
    Matrix3f Rt = R.transpose();
    linesStart = (Rt * localLinesStart.transpose()).transpose();
    linesStart += pointT.replicate(params.nLines,1);

    linesEnd = (Rt * localLinesEnd.transpose()).transpose();
    linesEnd += pointT.replicate(params.nLines,1);

    //process each line
    for( int i=0 ; i<params.nLines ; i++ )
    {
        occLine.compute(linesStart.row(i), linesEnd.row(i), values[i], status[i] );
    }
}


