/*! LineImageKnn.cpp
 *
 * Copyright (C) 2012 Alastair Quadros.
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
 * \date       26-05-2012
*/

#include <iostream>
#include "LineImageKnn.h"

using namespace Eigen;



void ObjectMatchHistogram::MatchObject( ObjLineImagesAligned& testData, MapMatXi& hist )
{
    if( hist.rows() != trainFData_.size() || hist.cols() != nBins_ )
    {
        std::cerr << "ObjectMatchHistogram::MatchObject - provided hist wrong size" << std::endl;
        return;
    }

    //horrible... but it seems maps and matrix are not interchangeable even using templated eigen types.
    VectorXi objectMatches( testData.nPoints );
    MapMatXi objectMatches_map( objectMatches.data(), testData.nPoints, 1 );
    VectorXi pointMatches( testData.nPoints );
    MapMatXi pointMatches_map( pointMatches.data(), testData.nPoints, 1 );
    VectorXf values( testData.nPoints );
    MapMatXf values_map( values.data(), testData.nPoints, 1 );

    //for each training object
    int o;
    #pragma omp parallel for schedule(guided, 5)
    for( o=0 ; o < trainFData_.size() ; o++ )
    {
        int nTrainPts = trainFData_[o].nPoints;
        std::vector<ObjLineImagesAligned> trainObj(1, trainFData_[o]);
        std::vector<int> nTrainPtsVect(1, nTrainPts);
        LineImageKnnAligned knnAligned( params_, metricNo_, trainObj, nTrainPtsVect,
                                        alignThresh_, rmseThresh_, knownThresh_,
                                        knownWeight_, false );
        knnAligned.SetTestObject( testData );
        knnAligned.Classify( objectMatches_map, pointMatches_map, values_map );

        //now have the best match of each test point. put the match values in a histogram.
        for( int i=0 ; i<testData.nPoints ; i++ )
        {
            int bin = values.coeff(i) / binWidth_;
            if( bin < nBins_ && bin >= 0 )
                hist(o, bin) += 1;
        }
    }
}



