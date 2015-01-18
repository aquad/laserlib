/*! LineImageMatcherDerived.cpp
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

#include <cmath>
#include "LineImageMatcherDerived.h"


void LineImageMatcher1::match_rmse( float* values1, unsigned char* status1,
                              float* values2, unsigned char* status2,
                              float& rmse, float& percKnown )
{
    int lineCount = 0;
    float sqrErr = 0;
    int nPointsLost = 0;
    int nPointsTotal = 0;

    //compare all lines
    for( int i=0 ; i<params.nLines ; i++ )
    {
        float lineLength = lineLengths[i];

        if( status1[i]==UNKNOWN && status2[i]==VALUE )
        {
            nPointsTotal++;
            if( values2[i] > values1[i] ) //point in empty region
            {
                sqrErr += std::pow(values1[i] - values2[i],2);
                lineCount++;
            }
            else
                nPointsLost++;
        }
        else if( status2[i]==UNKNOWN && status1[i]==VALUE )
        {
            nPointsTotal++;
            if( values1[i] > values2[i] ) //point in empty region
            {
                sqrErr += std::pow(values1[i] - values2[i],2);
                lineCount++;
            }
            else
                nPointsLost++;
        }
        else if( status2[i]==EMPTY && status1[i]==VALUE || //empty+value
                 status2[i]==VALUE && status1[i]==EMPTY )
        {
            sqrErr += lineLength*lineLength;
            lineCount++;
            nPointsTotal++;
        }
        else if( status1[i]==EMPTY && status2[i]==EMPTY )
        {
            lineCount++; //added 0 to sqrErr
        }
        else if( status1[i]==VALUE && status2[i]==VALUE )
        {
            sqrErr += std::pow(values1[i] - values2[i], 2);
            lineCount++;
            nPointsTotal+=2;
        }
        /*
        //implied remaining cases- do nothing
        else if( ( status1[i]==UNKNOWN || status1[i]==EMPTY ) &&
                 ( status2[i]==UNKNOWN || status2[i]==EMPTY ) )
        {
        }
        */
    }

    //prevent infs
    if( nPointsTotal==0 )
        percKnown = 0.0;
    else
        percKnown = (float)(nPointsTotal - nPointsLost) / (float)nPointsTotal;

    if( lineCount==0 )
        rmse = 100.0;
    else
        rmse = sqrt(sqrErr/lineCount);
}






void LineImageMatcher2::match_rmse( float* values1, unsigned char* status1,
                              float* values2, unsigned char* status2,
                              float& rmse, float& percKnown )
{
    int lineCount = 0;
    float sqrErr = 0;
    int nPoints = 0;
    int nPointsTotal = 0;

    //compare all lines
    for( int i=0 ; i<params.nLines ; i++ )
    {
        float lineLength = lineLengths[i];

        if( status1[i]==VALUE )
            nPointsTotal++;
        if( status2[i]==VALUE )
            nPointsTotal++;

        if( status1[i]==UNKNOWN && status2[i]==VALUE )
        {
            if( values2[i] > values1[i] ) //point in empty region
            {
                sqrErr += lineLength*lineLength;
                lineCount++;
                nPoints++;
            }
        }
        else if( status2[i]==UNKNOWN && status1[i]==VALUE )
        {
            if( values1[i] > values2[i] ) //point in empty region
            {
                sqrErr += lineLength*lineLength;
                lineCount++;
                nPoints++;
            }
        }
        else if( status2[i]==EMPTY && status1[i]==VALUE ||
                 status2[i]==VALUE && status1[i]==EMPTY )
        {
            sqrErr += lineLength*lineLength;
            lineCount++;
            nPoints++;
        }
        else if( status1[i]==EMPTY && status2[i]==EMPTY )
        {
            lineCount++; //added 0 to sqrErr
        }
        else if( status1[i]==VALUE && status2[i]==VALUE )
        {
            sqrErr += std::pow(values1[i] - values2[i], 2);
            lineCount++;
            nPoints+=2;
        }
    }

    //prevent infs
    if(nPointsTotal==0)
        percKnown = 0.0;
    else
        percKnown = (float)nPoints / (float)nPointsTotal;

    if( lineCount==0 )
        rmse = 100.0;
    else
        rmse = sqrt(sqrErr/lineCount);
}





void LineImageMatcher3::match_rmse( float* values1, unsigned char* status1,
                              float* values2, unsigned char* status2,
                              float& rmse, float& percKnown )
{
    float sqrErr = 0;
    float lengthKnown = 0.0;
    int nPoints = 0;
    int nPointsTotal = 0;

    //compare all lines
    for( int i=0 ; i<params.nLines ; i++ )
    {
        float lineLength = lineLengths[i];

        if( status1[i]==VALUE )
            nPointsTotal++;
        if( status2[i]==VALUE )
            nPointsTotal++;

        if( status1[i]==UNKNOWN && status2[i]==VALUE )
        {
            if( values2[i] > values1[i] ) //point in empty region
            {
                sqrErr += lineLength*lineLength;
                nPoints++;
            }
            lengthKnown += lineLength/2 - values1[i];
        }
        else if( status2[i]==UNKNOWN && status1[i]==VALUE )
        {
            if( values1[i] > values2[i] ) //point in empty region
            {
                sqrErr += lineLength*lineLength;
                nPoints++;
            }
            lengthKnown += lineLength/2 - values2[i];
        }
        else if( status2[i]==EMPTY && status1[i]==VALUE ||
                 status2[i]==VALUE && status1[i]==EMPTY )
        {
            sqrErr += lineLength*lineLength;
            nPoints++;
            lengthKnown += lineLength;
        }
        else if( (status2[i]==UNKNOWN && status1[i]==UNKNOWN) ||
                 (status2[i]==EMPTY && status1[i]==UNKNOWN) ||
                 (status2[i]==UNKNOWN && status1[i]==EMPTY) )
        { //add the overlapping 'known empty' bit
            lengthKnown += lineLength/2 - std::max(values1[i], values2[i]);
        }
        else if( status1[i]==EMPTY && status2[i]==EMPTY )
        {
            lengthKnown += lineLength;
        }
        else if( status1[i]==VALUE && status2[i]==VALUE )
        {
            sqrErr += std::pow(values1[i] - values2[i], 2);
            nPoints+=2;
            lengthKnown += lineLength;
        }
    }

    //prevent infs
    if(nPointsTotal==0)
        percKnown = 0.0;
    else
        percKnown = (float)nPoints / (float)nPointsTotal;

    if( lengthKnown==0.0 )
        rmse = 100.0;
    else
        rmse = sqrt(sqrErr/lengthKnown);
}





void LineImageMatcher4::match_rmse( float* values1, unsigned char* status1,
                              float* values2, unsigned char* status2,
                              float& rmse, float& percKnown )
{
    int lineCount = 0;
    float PQsum = 0.0;
    float Psum = 0.0;
    float Qsum = 0.0;
    float PsqrSum = 0.0;
    float QsqrSum = 0.0;

    //compare all lines
    for( int i=0 ; i<params.nLines ; i++ )
    {
        float Pi = values1[i] + lineLengths[i]/2;
        float Qi = values2[i] + lineLengths[i]/2;

        if( status1[i]==UNKNOWN || status2[i]==UNKNOWN )
            continue;
        else
        {
            PQsum += Pi*Qi;
            Psum += Pi;
            Qsum += Qi;
            PsqrSum += Pi*Pi;
            QsqrSum += Qi*Qi;
            lineCount++;
        }
    }

    percKnown = (float)lineCount / (float)params.nLines;
    float R = ( lineCount * PQsum - Psum*Qsum ) /
               sqrt( ( lineCount * PsqrSum - Psum*Psum ) *
                     ( lineCount * QsqrSum - Qsum*Qsum ) );
    float atanhR = 0.5*log((1+R)/(1-R));
    float corr = atanhR*atanhR;
    rmse = -corr;
}




void LineImageMatcher5::match_rmse( float* values1, unsigned char* status1,
                              float* values2, unsigned char* status2,
                              float& rmse, float& percKnown )
{
    int lineCount = 0;
    float sqrErr = 0;
    int nPointsLost = 0;
    int nPointsTotal = 0;

    //compare all lines
    for( int i=0 ; i<params.nLines ; i++ )
    {
        float lineLength = lineLengths[i];

        if( status1[i]==UNKNOWN && status2[i]==VALUE )
        {
            nPointsTotal++;
            if( values2[i] > values1[i] ) //point in empty region
            {
                sqrErr += lineLength/4;
                lineCount++;
            }
            else
                nPointsLost++;
        }
        else if( status2[i]==UNKNOWN && status1[i]==VALUE )
        {
            nPointsTotal++;
            if( values1[i] > values2[i] ) //point in empty region
            {
                sqrErr += lineLength/4;
                lineCount++;
            }
            else
                nPointsLost++;
        }
        else if( status2[i]==EMPTY && status1[i]==VALUE || //empty+value
                 status2[i]==VALUE && status1[i]==EMPTY )
        {
            sqrErr += lineLength/4;
            lineCount++;
            nPointsTotal++;
        }
        else if( status1[i]==EMPTY && status2[i]==EMPTY )
        {
            lineCount++; //added 0 to sqrErr
        }
        else if( status1[i]==VALUE && status2[i]==VALUE )
        {
            sqrErr += std::pow(values1[i] - values2[i], 2);
            lineCount++;
            nPointsTotal+=2;
        }
        /*
        //implied remaining cases- do nothing
        else if( ( status1[i]==UNKNOWN || status1[i]==EMPTY ) &&
                 ( status2[i]==UNKNOWN || status2[i]==EMPTY ) )
        {
        }
        */
    }

    //prevent infs
    if( nPointsTotal==0 )
        percKnown = 0.0;
    else
        percKnown = (float)(nPointsTotal - nPointsLost) / (float)nPointsTotal;

    if( lineCount==0 )
        rmse = 100.0;
    else
        rmse = sqrt(sqrErr/lineCount);
}




/* Hausdorff distance.
Two sets: X, Y
Each is a line image, where some dimensions are unknown (and bounded).

For each possible x, find the closest possible y. Of all these, select the
maximum distance.
Do the same for y to x, and take the max of the two.

The non-occluded distance measure is the L2 norm (empty = 0)

For the closest possible distance between two line images, consider each line
pair with occlusion.
*/
void LineImageMatcher6::match_rmse( float* values1, unsigned char* status1,
                              float* values2, unsigned char* status2,
                              float& rmse, float& percKnown )
{
    float sqrErr = 0;
    for( int i=0 ; i<params.nLines ; i++ )
    {
        float halfLine = lineLengths[i]/2;
        float xy = match_xy_faster( status1[i], values1[i], status2[i], values2[i], halfLine );
        float yx = match_xy_faster( status2[i], values2[i], status1[i], values1[i], halfLine );
        sqrErr += std::max(xy, yx);
    }
    rmse = sqrErr / params.nLines;
}


/*
(X - Y)
- unknown - intercept/empty (both conflicting or no-conflict):
  - the largest possible (x + L/2).
- unknown - unknown:
  - inf: 0 where they overlap. if Y has more empty, pick closest possible.
  - sup: if Y has more empty (y < x), use x-y. else 0.
- intercept/empty - intercept/empty: standard distance.
- intercept/empty - unknown (no conflict): 0
- intercept - unknown (conflict): intercept -> occlusion depth

speed up: non-commutative function to combine X,Y status (eg bit-combine).
Then map numbers to functions. only 16. use function pointers in array??
NOT POSSIBLE- you can't inline, no real benefit.
*/
inline float LineImageMatcher6::match_xy( unsigned char sx, float x, unsigned char sy, float y, float halfLine )
{
    if( sx==UNKNOWN && ( sy==VALUE || sy==EMPTY ) )
    {
        return x + halfLine;
    }

    else if( sx==UNKNOWN && sy==UNKNOWN )
    {
        if( y < x )
            return std::pow(x-y, 2);
        else
            return 0;
    }

    else if( (sx==EMPTY || sx==VALUE) && (sy==EMPTY || sy==VALUE) )
        return std::pow(x-y, 2);

    else if( sx==EMPTY && sy==UNKNOWN )
        return 0;

    else if( sx==VALUE && sy==UNKNOWN )
    {
        if( x>y ) //conflict
            return std::pow(x-y, 2);
        else
            return 0;
    }
}


// as match_xy, but using a switch statement. should be faster...
inline float LineImageMatcher6::match_xy_faster( unsigned char sx, float x, unsigned char sy, float y, float halfLine )
{
    //s: x, y
    unsigned char s = sx;
    s = s << 4;
    s = s | sy;

    //{UNKNOWN=0, EMPTY=1, VALUE=2};
    switch(s)
    {
        case 0x00: //unknown-unknown
        {
            if( y < x )
                return std::pow(x-y, 2);
            else
                return 0;
            break;
        }
        case 0x01: //unknown-empty
        {
            return x + halfLine;
            break;
        }
        case 0x02: //unknown-value
        {
            return x + halfLine;
            break;
        }
        case 0x10: //empty-unknown
        {
            return 0;
            break;
        }
        case 0x11: //empty-emtpy
        {
            return std::pow(x-y, 2);
            break;
        }
        case 0x12: //empty-value
        {
            return std::pow(x-y, 2);
            break;
        }
        case 0x20: //value-unknown
        {
            if( x>y ) //conflict
                return std::pow(x-y, 2);
            else
                return 0;
            break;
        }
        case 0x21: //value-empty
        {
            return std::pow(x-y, 2);
            break;
        }
        case 0x22: //value-value
        {
            return std::pow(x-y, 2);
            break;
        }
        default: //should throw an error or something
        {
            return 0;
            break;
        }
    }
}





void LineImageMatcher7::match_rmse( float* values1, unsigned char* status1,
                              float* values2, unsigned char* status2,
                              float& rmse, float& percKnown )
{
    int lineCount = 0;
    float sqrErr = 0;
    int nPoints = 0;
    int nPointsTotal = 0; //only for status1

    //compare all lines
    for( int i=0 ; i<params.nLines ; i++ )
    {
        float lineLength = lineLengths[i];

        if( status1[i]==VALUE )
            nPointsTotal++;

        //s: 1, 2
        unsigned char s = status1[i];
        s = s << 4;
        s = s | status2[i];
        float x = values1[i];
        float y = values2[i];

        //{UNKNOWN=0, EMPTY=1, VALUE=2};
        switch(s)
        {
            case 0x00: //unknown-unknown
            {
                break;
            }
            case 0x01: //unknown-empty
            {
                break;
            }
            case 0x02: //unknown-value
            {
                if( y>x ) //conflict
                {
                    sqrErr += std::pow(lineLength, 2);
                    lineCount++;
                }
                break;
            }
            case 0x10: //empty-unknown
            {
                break;
            }
            case 0x11: //empty-empty
            {
                lineCount++;
                break;
            }
            case 0x12: //empty-value
            {
                sqrErr += std::pow(lineLength, 2);
                lineCount++;
                break;
            }
            case 0x20: //value-unknown
            {
                if( x>y ) //conflict
                {
                    sqrErr += std::pow(lineLength, 2);
                    lineCount++;
                    nPoints++;
                }
                break;
            }
            case 0x21: //value-empty
            {
                sqrErr += std::pow(lineLength, 2);
                lineCount++;
                nPoints++;
                break;
            }
            case 0x22: //value-value
            {
                sqrErr += std::pow(x-y, 2);
                lineCount++;
                nPoints++;
                break;
            }
            default:
            {
                break;
            }
        }
    }

    //prevent infs
    if(nPointsTotal==0)
        percKnown = 0.0;
    else
        percKnown = (float)nPoints / (float)nPointsTotal;

    if( lineCount==0 )
        rmse = 100.0;
    else
        rmse = sqrt(sqrErr/lineCount);
}






