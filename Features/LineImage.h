/*! LineImage.h
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

#ifndef LINE_IMAGE_HEADER_GUARD
#define LINE_IMAGE_HEADER_GUARD

#define EIGEN_DEFAULT_TO_ROW_MAJOR 1
#include <Eigen/Dense>

#include <vector>

#include "Common/ArrayTypes.h"
#include "DataStore/VelodyneDb.h"
#include "DataStore/VeloRangeImage.h"
#include "OccLine.h"


struct LineImageParams : OccLineParams
{
    std::vector<int> angularSections;
    int nLines;
    int nRadSections;
    double diskRad; //outer lines are at this radius.
    double regionRad; //radius of region to consider (should be slightly larger than diskRad).
};


class ComputeLineImage
{
public:
    ComputeLineImage( LineImageParams& params, Mat3<double>::type& P,
            Vect<unsigned char>::type& id, Vect<double>::type& D, Vect<double>::type& w,
            PCAResults& pcaResults, VelodyneDb& db, VeloRangeImage& image );
    virtual ~ComputeLineImage();

    //set this if you know the background and don't want it in the feature
    void setObjectMask( std::vector<bool>& mask );

    void compute( const Eigen::Map<Eigen::Matrix3f>& R, const Eigen::Vector3d& point, float* values, unsigned char* status );

    const int nPoints;

protected:
    LineImageParams params;

    Eigen::Matrix<float, Eigen::Dynamic, 3> localLinesEnd, localLinesStart, linesStart, linesEnd;
    OccLine occLine;
};



#endif //LINE_IMAGE_HEADER_GUARD
