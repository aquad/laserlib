/*! LineDistrib.h
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

#ifndef LINE_DISTRIBUTION_HEADER_GUARD
#define LINE_DISTRIBUTION_HEADER_GUARD

#include "export.h"
#include <vector>
#include "LineImage.h"


/* Convert a line image to a 'Line Distribution', where each line is chopped up
 * into bins of a given length. This produces a distribution for each line,
 * stating the probability of a surface existing in each bin.
 * There is also a separate probability of each line containing an intercept at
 * all (ie. empty lines influence this).
 *
 * The centre bin (at value 0) is centralised, so flat surfaces don't jump
 * between bins. Although there will be smoothing between bins anyway.
 *
 * This class will then be used in a bayes-filter style clustering algorithm,
 * which will allow unknown regions to be revealed as matching shapes are
 * integrated.
 * */
class LASERLIB_FEATURES_EXPORT LineDistrib
{
public:
    typedef std::vector<float>::iterator BinIter;
    typedef std::vector<float>::const_iterator ConstBinIter;

    LineDistrib( LineImageParams& params, float binLength, float zeroValue=0.0001, float emptyValue=0.01 );
    LineDistrib( const LineDistrib& );
    virtual ~LineDistrib(){}

    //! set bins from a line image
    void Set( float* values, unsigned char* status );

    //! the bin number of the specified line with the specified value
    int ValueToLineBin( float val, int line ) const
        { return (int)( (val - startBinValue_[line]) / binLength_ ); }

    void Merge( const LineDistrib& other );

    //!{ Access bins
    ConstBinIter begin( int line ) const
        { return bins_[line].begin(); }

    ConstBinIter end(int line ) const
        { return bins_[line].end(); }

    BinIter begin( int line )
        { return bins_[line].begin(); }

    BinIter end(int line )
        { return bins_[line].end(); }

    float operator()( int line, int bin ) const
        { return bins_[line][bin]; }
    //!}

    // Hard to get around providing the raw data containers...
    /*! the bins in each line, from the back to the front.
     * (log probability of intercept at each bin).
     */
    std::vector< std::vector<float> > bins_;

     //! log odds of each line having a surface intercept
    std::vector<float> logOddsIntercept_;

private:
    LineImageParams params_;
    float binLength_;

    //! Can't have zero-valued probability in bins, use a small number instead
    const float zeroValue_;
    const float logZeroValue_;

    /*! When converting from a line image, set the empty space to this
     * probability of intercept (more than zeroValue_, so multiple empty-space
     * observations can stack).
     */
    const float emptyValue_;

    //! uninformative prior for intercept/empty state
    const float logOddsPrior_;

    std::vector<float> height_; //!< height of each line

    /*! due to the awkward bin alignment, it helps to know the
     * depth of the starting bin in each line.
     */
    std::vector<float> startBinValue_;
};



#endif //LINE_DISTRIBUTION_HEADER_GUARD

