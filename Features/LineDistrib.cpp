/*! LineDistrib.cpp
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
 * \date       01-05-2012
*/

#include "LineDistrib.h"
#include <algorithm>
#include <math.h>
#include <limits>
#include <iostream>


LineDistrib::LineDistrib( LineImageParams& params, float binLength, float zeroValue, float emptyValue )
    :   params_(params),
        binLength_(binLength),
        zeroValue_(zeroValue),
        emptyValue_(emptyValue),
        logZeroValue_( log(zeroValue_) ),
        logOddsPrior_( log(1.0/(1-0.5)) )
{
    // assign vector sizes
    logOddsIntercept_.assign(params.nLines, logOddsPrior_);
    height_.assign(params.nLines, 0);
    startBinValue_.assign(params.nLines, 0);
    bins_.resize(params.nLines);

    //find number of bins in each line.
    float midBinLen = binLength/2;
    int n=0; //indexes line number
    for( int i=0 ; i<params.nRadSections ; i++ )
    {
        double radius = params.diskRad / params.nRadSections * (i+1);
        double height = sqrt( params.regionRad*params.regionRad - radius*radius );
        //the number of bins above the centre, 0-depth bin.
        int nBinsAbove = (int)( (height - midBinLen) / binLength_ );
        int nBinsPerLine = 1 + nBinsAbove * 2; //centre bin + above + below
        if(nBinsPerLine < 0){ nBinsPerLine = 1; } //could be a very short line

        //calculate the depth (0=middle of line image) of the starting bin.
        //infront = +ve, behind = -ve. starting from the back
        float startBinVal = -binLength/2 - nBinsAbove * binLength_;

        //line height, nBins same for all in current ring
        for( unsigned int j=0 ; j<params.angularSections[i] ; j++ )
        {
            bins_[n].assign(nBinsPerLine, 0);
            height_[n] = height;
            startBinValue_[n] = startBinVal;
            n++;
        }
    }
}


LineDistrib::LineDistrib( const LineDistrib& rhs )
    :   params_(rhs.params_),
        binLength_(rhs.binLength_),
        zeroValue_(rhs.zeroValue_),
        emptyValue_(rhs.emptyValue_),
        logZeroValue_( rhs.logZeroValue_ ),
        logOddsPrior_( rhs.logOddsPrior_ ),
        logOddsIntercept_( rhs.logOddsIntercept_ ),
        bins_( rhs.bins_ ),
        height_( rhs.height_ ),
        startBinValue_( rhs.startBinValue_ )
{}



void LineDistrib::Merge( const LineDistrib& other )
{
    //assumes other is the same size
    for( int i=0 ; i<params_.nLines ; i++ )
    {
        // multiply, normalise
        double sum = 0.0;
        ConstBinIter otherIt = other.begin(i),
             otherEnd = other.end(i);
        BinIter thisIt = begin(i),
                thisEnd = end(i);
        for( ; thisIt!=thisEnd ; ++thisIt, ++otherIt )
        {
            float otherVal = *otherIt;
            // cap the min value possible. prevents sum from getting too small
            // exp(*thisIt) will reach min numeric limit.
            // need to ensure bins can get out of the minimum value if merged
            // with larger values
            if( *thisIt < logZeroValue_ )
                *thisIt = logZeroValue_;
            if( otherVal < logZeroValue_ )
                otherVal = logZeroValue_;
            *thisIt += otherVal;
            sum += exp((double)*thisIt);
            logOddsIntercept_[i] += other.logOddsIntercept_[i] - logOddsPrior_;
        }
        // normalise
        float logSum = (float)log(sum);
        for( thisIt = begin(i) ; thisIt != thisEnd ; ++thisIt )
        {
            (*thisIt) -= logSum;
        }
    }
}






//does not presume bins are cleared
void LineDistrib::Set( float* values, unsigned char* status )
{
    for( int i=0 ; i<params_.nLines ; i++ )
    {
        switch(status[i])
        {
            case UNKNOWN:
            {
                //all bins before value are empty
                //all after are the same, sum to 1.
                // this bin is where the transition to unknown space occurs:
                int binNo = ValueToLineBin(values[i], i);
                //std::cout << "line " << i << ", binNo: " << binNo;
                // therefore, there are binNo+1 bins with an 'unknown' state.
                int nBinsUnknown = binNo + 1;

                // Bins must sum to 1, calculate value for each 'unknown' bin
                // (derivation in lab book).
                int nBins = bins_[i].size();
                float p = (1 - (nBins-nBinsUnknown) * emptyValue_) / nBinsUnknown;
                //std::cout << ", p: " << p << std::endl;
                std::fill_n( bins_[i].begin(), nBinsUnknown, log(p) );
                // the rest are emptyValue.
                if( nBinsUnknown < nBins )
                    std::fill( bins_[i].begin() + nBinsUnknown, bins_[i].end(), log(emptyValue_) );
                break;
            }

            case EMPTY:
            {
                // bins all equal, sum to 1
                float p = 1.0/bins_[i].size();
                std::fill( bins_[i].begin(), bins_[i].end(), log(p) );
                // use probability of intercept 0.01
                logOddsIntercept_[i] = log(0.01/(1-0.01));
                break;
            }

            case VALUE:
            {
                // assign other bins the min value
                std::fill( bins_[i].begin(), bins_[i].end(), log(emptyValue_) );
                // the remaining prob mass goes into the one bin.
                int nBins = bins_[i].size();
                float p = 1 - (nBins-1) * emptyValue_;
                int binNo = ValueToLineBin(values[i], i);
                bins_[i][binNo] = log(p);
                // use probability of intercept 0.99
                logOddsIntercept_[i] = log(0.99/(1-0.99));
                break;
            }
        }
    }
}




