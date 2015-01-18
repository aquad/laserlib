/*! LineImageMatcher.cpp
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

#include <string>
#include <iostream>
#include <cmath>
#include "boost/math/constants/constants.hpp"
const double pi = boost::math::constants::pi<double>();
const double loge2 = log(2.0);

#include "LineImageMatcher.h"
#include "LineImageMatcherDerived.h"

using namespace Eigen;


LineImageMatcher::LineImageMatcher(LineImageParams& _params)
    :   params(_params),
        maxNAngularSections( *std::max_element( params.angularSections.begin(), params.angularSections.end() ) ),
        rotValues( new float[params.nLines] ),
        rotStatus( new unsigned char[params.nLines] )
{
    ringPositions.push_back(0);
    for( int i=0 ; i<params.nRadSections ; i++ )
    {
        ringPositions.push_back( ringPositions.back() + params.angularSections[i] );
    }

    lineLengths.resize(params.nLines);
    sumLineLength=0;
    int lineNo = 0;
    for( int i=0 ; i<params.nRadSections ; i++ )
    {
        double radius = params.diskRad / params.nRadSections * (i+1);
        double height = sqrt( params.regionRad*params.regionRad - radius*radius );
        for( int j=0 ; j<params.angularSections[i] ; j++ )
        {
            lineLengths[lineNo] = height*2;
            sumLineLength += lineLengths[lineNo];
            lineNo++;
        }
    }
}



LineImageMatcher::LineImageMatcher( const LineImageMatcher& other)
    :   params(other.params),
        maxNAngularSections(other.maxNAngularSections),
        rotValues( new float[params.nLines] ),
        rotStatus( new unsigned char[params.nLines] ),
        lineLengths(other.lineLengths),
        sumLineLength(other.sumLineLength),
        ringPositions(other.ringPositions)
{
    memcpy(rotValues, other.rotValues, params.nLines*sizeof(float));
    memcpy(rotStatus, other.rotStatus, params.nLines*sizeof(unsigned char));
}



LineImageMatcher::~LineImageMatcher()
{
    delete[] rotValues;
    delete[] rotStatus;
}



void LineImageMatcher::match_rmse_spin( float* values1, unsigned char* status1,
                              float* values2, unsigned char* status2,
                              float& rmse, float& percKnown, int& matchAngle )
{
    //store line counts & squared errors per rotary offset (same index arrangment as values etc)
    std::vector<int> lineCounts;
    std::vector<double> sqrErr;
    std::vector<double> lengthKnown;
    lineCounts.assign(params.nLines, 0);
    sqrErr.assign(params.nLines, 0);
    lengthKnown.assign(params.nLines, 0);

    for( int i=0 ; i<params.nRadSections ; i++ ) //each radial section
    {
        //rotate to each angular position
        for( int j=0 ; j<params.angularSections[i] ; j++ )
        {
            int resultIndex = j + ringPositions[i];
            //compare all values for this section & rotation
            for( int k=0 ; k<params.angularSections[i] ; k++ )
            {
                int i1 = k + ringPositions[i];
                int i2 = (k+j)%params.angularSections[i] + ringPositions[i];

                if( status1[i1]==UNKNOWN && status2[i2]==VALUE )
                {
                    if( values2[i2] > values1[i1] ) //point in empty region
                    {
                        sqrErr[resultIndex] += std::pow(values1[i1] - values2[i2],2);
                        lineCounts[resultIndex]++;
                    }
                    lengthKnown[resultIndex] += lineLengths[i]/2 - values1[i1];
                }
                else if( status2[i2]==UNKNOWN && status1[i1]==VALUE )
                {
                    if( values1[i1] > values2[i2] ) //point in empty region
                    {
                        sqrErr[resultIndex] += std::pow(values1[i1] - values2[i2],2);
                        lineCounts[resultIndex]++;
                    }
                    lengthKnown[resultIndex] += lineLengths[i]/2 - values2[i2];
                }
                else if( (status2[i2]==UNKNOWN && status1[i1]==UNKNOWN) ||
                         (status2[i2]==EMPTY && status1[i1]==UNKNOWN) ||
                         (status2[i2]==UNKNOWN && status1[i1]==EMPTY) )
                { //add the overlapping 'known empty' bit
                    lengthKnown[resultIndex] += lineLengths[i]/2 - std::max(values1[i1], values2[i2]);
                }
                else if( status1[i1]==EMPTY && status2[i2]==EMPTY )
                {
                    lengthKnown[resultIndex] += lineLengths[i];
                    lineCounts[resultIndex]++; //added 0 to sqrErr
                }
                else //both have a value
                {
                    sqrErr[resultIndex] += std::pow(values1[i1] - values2[i2], 2);
                    lineCounts[resultIndex]++;
                    lengthKnown[resultIndex] += lineLengths[i];
                }
            }
        }
    }

    float bestRmse = 1000;
    float known = 0;
    int angle = 0;
    //add results from each radial section together
    for( int i=0 ; i<maxNAngularSections ; i++ )
    {
        double rotFraction = (double)i / maxNAngularSections;
        float thisRmse = 0;
        int thisLineCount = 0;
        float thisKnown = 0;
        //summing each radial section occurring at this angle
        for( int j=0 ; j<params.nRadSections ; j++ )
        {
            int index = (int)(rotFraction * params.angularSections[j]) + ringPositions[j];
            thisRmse += sqrErr[index];
            thisLineCount += lineCounts[index];
            thisKnown += lengthKnown[index];
        }
        thisRmse = sqrt(thisRmse/thisLineCount);
        thisKnown /= sumLineLength;

        if(thisRmse < bestRmse)
        {
            bestRmse = thisRmse;
            known = thisKnown;
            angle = i;
        }
    }

    rmse = bestRmse;
    percKnown = known;
    matchAngle = angle;
}




void LineImageMatcher::match_rmse_one_many( float* values, unsigned char* status,
            float* valuesSet, unsigned char* statusSet, int n,
            float* rmse, float* known )
{
    for( int i=0 ; i<n ; i++ )
    {
        int index = i*params.nLines;
        float* values2 = valuesSet + index;
        unsigned char* status2 = statusSet + index;
        match_rmse(values, status, values2, status2, rmse[i], known[i]);
    }
}



void LineImageMatcher::match_rmse_one_one( float* values1, unsigned char* status1,
            float* values2, unsigned char* status2, int n,
            float* rmse, float* known )
{
    for( int i=0 ; i<n ; i++ )
    {
        int index = i*params.nLines;
        match_rmse(values1 + index, status1 + index,
                   values2 + index, status2 + index,
                   rmse[i], known[i]);
    }
}



void LineImageMatcher::match_rmse_one_many_keys( float* values, unsigned char* status,
            float* valuesSet, unsigned char* statusSet, std::vector<int>& keys,
            float* rmse, float* known )
{
    for( int k=0 ; k<keys.size() ; k++ )
    {
        int i = keys[k];
        int index = i*params.nLines;
        float* values2 = valuesSet + index;
        unsigned char* status2 = statusSet + index;
        match_rmse(values, status, values2, status2, rmse[i], known[i]);
    }
}



boost::shared_ptr<LineImageMatcher> MakeLIMatcher( LineImageParams& params, int metric )
{
    boost::shared_ptr<LineImageMatcher> matcher;
    switch(metric)
    {
        case 1:
            matcher.reset( new LineImageMatcher1(params) );
            break;
        case 2:
            matcher.reset( new LineImageMatcher2(params) );
            break;
        case 3:
            matcher.reset( new LineImageMatcher3(params) );
            break;
        case 4:
            matcher.reset( new LineImageMatcher4(params) );
            break;
        case 5:
            matcher.reset( new LineImageMatcher5(params) );
            break;
        case 6:
            matcher.reset( new LineImageMatcher6(params) );
            break;
        case 7:
            matcher.reset( new LineImageMatcher7(params) );
            break;
        default:
            std::cerr << "invalid metric: using original" << std::endl;
            matcher.reset( new LineImageMatcher1(params) );
    }
    return matcher;
}




