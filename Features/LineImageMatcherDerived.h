/*! LineImageMatcherDerived.h
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

#ifndef LINE_IMAGE_MATCHER_DERIVED_HEADER_GUARD
#define LINE_IMAGE_MATCHER_DERIVED_HEADER_GUARD

#include "LineImageMatcher.h"


class LineImageMatcher1 : public LineImageMatcher
{
public:
    LineImageMatcher1(LineImageParams& params) : LineImageMatcher(params) {}
    LineImageMatcher1(const LineImageMatcher1& other) : LineImageMatcher(other) {}

    /*! Lines are penalised as:
      point-point: distance -> rmse.
      point-occlusion: if point is in empty space, point-occlusion distance -> rmse.
        else, point-occlusion distance -> known.
      occlusion-occlusion: max occlusion -> known.
      empty-empty: rmse goes down.
    */
    void match_rmse( float* values1, unsigned char* status1,
            float* values2, unsigned char* status2,
            float& rmse, float& known );

    LineImageMatcher* clone()
        { return new LineImageMatcher1(params); }
};



class LineImageMatcher2 : public LineImageMatcher
{
public:
    LineImageMatcher2(LineImageParams& params) : LineImageMatcher(params) {}
    LineImageMatcher2(const LineImageMatcher2& other) : LineImageMatcher(other) {}

    /*! When a point is in an empty space, give it a large error (full line error).
    Percent known is now the number of mutually shared points / total points from both
    (need to penalise the info loss of points going in occluded regions).
    */
    void match_rmse( float* values1, unsigned char* status1,
            float* values2, unsigned char* status2,
            float& rmse, float& known );

    LineImageMatcher* clone()
        { return new LineImageMatcher2(params); }
};



class LineImageMatcher3 : public LineImageMatcher
{
public:
    LineImageMatcher3(LineImageParams& params) : LineImageMatcher(params) {}
    LineImageMatcher3(const LineImageMatcher3& other) : LineImageMatcher(other) {}

    /*! Also need to penalise lots of empty space going into occluded space...
    each line has an 'occlusion' value- when a point has to go in occluded space,
    that value is maxed- give it the entire length of the line. when a section of
    empty line must match with occluded line, give it that length of occluded empty.
    The total length of 'visible' line is the length of total empty plus the full length
    of lines that have intercepts. This allows us to penalise an entire empty line that
    matches to an entire occluded line at the same value of a point becoming occluded.
    Likewise, the preexisting occlusion of each line image does not affect the occlusion
    value- it's just when the occlusion is used to 'explain' data that this value goes up.

    reminder- 'value' is the distance from the centre to the point of intercept/occlusion,
    in the direction along the surface normal (away from the surface, towards the sensor).
    */
    void match_rmse( float* values1, unsigned char* status1,
            float* values2, unsigned char* status2,
            float& rmse, float& known );

    LineImageMatcher* clone()
        { return new LineImageMatcher3(params); }
};



class LineImageMatcher4 : public LineImageMatcher
{
public:
    LineImageMatcher4(LineImageParams& params) : LineImageMatcher(params) {}
    LineImageMatcher4(const LineImageMatcher4& other) : LineImageMatcher(other) {}

    /*! Try spin image linear correlation coefficient for non-occluded bits.
    */
    void match_rmse( float* values1, unsigned char* status1,
            float* values2, unsigned char* status2,
            float& rmse, float& known );

    LineImageMatcher* clone()
        { return new LineImageMatcher4(params); }
};


class LineImageMatcher5 : public LineImageMatcher
{
public:
    LineImageMatcher5(LineImageParams& params) : LineImageMatcher(params) {}
    LineImageMatcher5(const LineImageMatcher5& other) : LineImageMatcher(other) {}

    /*! Reduce penalty for points in empty space
    */
    void match_rmse( float* values1, unsigned char* status1,
            float* values2, unsigned char* status2,
            float& rmse, float& known );

    LineImageMatcher* clone()
        { return new LineImageMatcher5(params); }
};




class LineImageMatcher6 : public LineImageMatcher
{
public:
    LineImageMatcher6(LineImageParams& params) : LineImageMatcher(params) {}
    LineImageMatcher6(const LineImageMatcher6& other) : LineImageMatcher(other) {}

    /*! Hausdorff distance
    */
    void match_rmse( float* values1, unsigned char* status1,
            float* values2, unsigned char* status2,
            float& rmse, float& known );

    LineImageMatcher* clone()
        { return new LineImageMatcher6(params); }

private:
    float match_xy( unsigned char sx, float x, unsigned char sy, float y, float halfLine );
    float match_xy_faster( unsigned char sx, float x, unsigned char sy, float y, float halfLine );
};



class LineImageMatcher7 : public LineImageMatcher
{
public:
    LineImageMatcher7(LineImageParams& params) : LineImageMatcher(params) {}
    LineImageMatcher7(const LineImageMatcher7& other) : LineImageMatcher(other) {}

    /*! Non-symmetric 'known':
     *  known(a,b):
     *  % of a's points missing in b
    */
    void match_rmse( float* values1, unsigned char* status1,
            float* values2, unsigned char* status2,
            float& rmse, float& known );

    LineImageMatcher* clone()
        { return new LineImageMatcher7(params); }
};





#endif //LINE_IMAGE_MATCHER_DERIVED_HEADER_GUARD
