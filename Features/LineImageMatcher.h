/*! LineImageMatcher.h
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

#ifndef LINE_IMAGE_MATCHER_HEADER_GUARD
#define LINE_IMAGE_MATCHER_HEADER_GUARD

#include "export.h"
//Eigen
#define EIGEN_DEFAULT_TO_ROW_MAJOR 1
#include <Eigen/Core>

#include "Common/ArrayTypes.h"
#include "LineImage.h"
#include "KnnClassifier.h"
#include <vector>
#include <boost/shared_ptr.hpp>


//! Compute distance metrics for line images
class LASERLIB_FEATURES_EXPORT LineImageMatcher
{
public:
    LineImageMatcher(LineImageParams& params);
    LineImageMatcher(const LineImageMatcher& other);
    virtual ~LineImageMatcher();

    /*! Match two line images, returning an rmse and %known value.
    The rmse is only for violations of 'known empty' regions.
    The percent known is the % length of line not in occlusion.
    */
    virtual void match_rmse( float* values1, unsigned char* status1,
            float* values2, unsigned char* status2,
            float& rmse, float& known ) = 0;

    /*! This allows someone to copy the derived object via the base class.
    Necessary for OMP parallel for loops
    */
    virtual LineImageMatcher* clone() = 0;

    //! match one against a set
    void match_rmse_one_many( float* values, unsigned char* status,
            float* valuesSet, unsigned char* statusSet, int n,
            float* rmse, float* known );

    /*! match one against a set, but only the items in 'keys'.
        Note- (rmse, known) are the full size of valuesSet.
    */
    void match_rmse_one_many_keys( float* values, unsigned char* status,
                float* valuesSet, unsigned char* statusSet,
                std::vector<int>& keys, float* rmse, float* known );

    //! match two equal-sized sets, one-to-one
    void match_rmse_one_one( float* values1, unsigned char* status1,
                float* values2, unsigned char* status2, int n,
                float* rmse, float* known );

    /*! Without a known spin alignment, try them all, returning the best.
    Currently uses it's own (old) match metric- need to update to use
    the derived match_rmse function
    */
    virtual void match_rmse_spin( float* values1, unsigned char* status1,
            float* values2, unsigned char* status2,
            float& rmse, float& known, int& matchAngle );



protected:
    LineImageParams params;
    int maxNAngularSections;
    float* rotValues;
    unsigned char* rotStatus;
    std::vector<double> lineLengths; //for each radial section
    double sumLineLength;
    std::vector<int> ringPositions;
};




//! Make a line image matcher of the given metric number.
LASERLIB_FEATURES_EXPORT boost::shared_ptr<LineImageMatcher> MakeLIMatcher( LineImageParams& params, int metric );



//! Line images in an object
class LASERLIB_FEATURES_EXPORT ObjLineImages
{
public:
    ObjLineImages( MapMatXf& values_, MapMatXuc& status_,
                   int nPoints_ )
        : values(values_),
          status(status_),
          nPoints(nPoints_)
    {}

    MapMatXf values; //!< line image values
    MapMatXuc status; //!< line image status
    int nPoints;
};



//! Line images and aligning vectors in an object
class LASERLIB_FEATURES_EXPORT ObjLineImagesAligned : public ObjLineImages
{
public:
    ObjLineImagesAligned( Mat3<float>::type& alignVectors_,
                          Vect<unsigned char>::type& alignType_,
                          MapMatXf& values_,
                          MapMatXuc& status_,
                          int nPoints_ )
        : ObjLineImages( values_, status_, nPoints_ ),
          alignVectors(alignVectors_),
          alignType(alignType_)
    {}

    Mat3<float>::type alignVectors; //!< Surface normal or linear vector
    Vect<unsigned char>::type alignType; //!< Whether alignVectors is a surface normal (0) or a linear vector (1)
};





#endif //LINE_IMAGE_MATCHER_HEADER_GUARD
