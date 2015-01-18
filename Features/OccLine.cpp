/*! OccLine.cpp
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
 * \date       05-12-2012
*/


#include "OccLine.h"

#include <iostream>
#include <list>
#include <algorithm>
#include <math.h>
#include <cmath>
#include <string>
#include <boost/math/constants/constants.hpp>
const double pi = boost::math::constants::pi<double>();
const double pow_2pi_neg_1p5 = std::pow( 2*boost::math::constants::pi<double>(), -1.5 );

#include "DataStore/VeloRangeImage.h"
#include "DataStore/VeloImageSelect.h"
#include "Common/argsort.h"

using namespace Eigen;


LineCoords::LineCoords( const Vector3f& _start, const Vector3f& _end, VeloRangeImage& image )
    : start(_start), end(_end)
{
    length = (start-end).norm();
    //line equation x = n*t + start
    n = end - start;
    n /= length;

    //convert to range image coords (2d)
    image.Point3dToRangeImage( start.data(), startBear.data() );
    image.Point3dToRangeImage( end.data(), endBear.data() );
    //2d line params: x = n*t + start
    nBear = endBear - startBear;
    nBearNorm = nBear.norm();
}



OccLine::OccLine( OccLineParams& _params, Mat3<double>::type& _P,
        Vect<unsigned char>::type& _id, Vect<double>::type& _D, Vect<double>::type& _w,
        PCAResults& _pcaResults, VelodyneDb& _db, VeloRangeImage& _image )

    :   params(_params),
        db(_db),
        P(_P), id(_id), D(_D), w(_w),
        pcaResults(_pcaResults),
        image(_image),
        isInObject(P.rows(), true)
{
    //allocate space for working variables
    pixelsOnLine.reserve(100);
    pointsOnLine.reserve(200);

    pointIds_unordered.reserve(200);
    lineTs_unordered.reserve(200);
    lineTs_argsort.reserve(200);
    pointIds.reserve(200);
    lineTs.reserve(200);
}


OccLine::~OccLine()
{}


void OccLine::setObjectMask( std::vector<bool>& mask )
{
    isInObject = mask;
}


//states of algorithm in compute() that moves along a given line, searching for an intercept and appropriate status.
enum LineComputeState {INFRONT, INTERCEPTED_AND_NEAR, INTERCEPTED, OCCLUDED};


// Line starts below the surface
void OccLine::compute( const Vector3f& start, const Vector3f& end, float& value, unsigned char& status )
{
    // Init status to empty- will be set to unknown if theres a break or it goes
    // behind something, set to value if it intercepts the surface.
    status = EMPTY;

    //------STEP 1: define the current line
    LineCoords line(start, end, image);

    //-------STEP 2: get points along line from the range image
    float line2dBounds[4] = {line.startBear[0], line.startBear[1], line.endBear[0], line.endBear[1]};
    pointsOnLine.clear();
    getPointsOnLine( line2dBounds, pointsOnLine );

    //-------STEP 2.5: order these points along the line (and remove far away ones)
    pointIds.clear();
    lineTs.clear();
    orderPointsOnLine(line, pointsOnLine, pointIds, lineTs);

    //-------STEP 3: go through these points in order (traverse the line)
    //initialise 'value'- this specifics up to where the line is empty. start from front of line.
    value = line.length/2;

    //first check if there are no points
    if( pointIds.size() == 0 )
    {
        status = UNKNOWN;
        value = line.length/2;
        return;
    }

    // Now go along the line, inspecting each point in order, looking for where
    // the line intercepts the surface, or goes behind it.
    int closestPid;
    traverseLineForIntercept( pointIds, lineTs, line, closestPid, value, status );
}




void OccLine::orderPointsOnLine( LineCoords& line,
        std::vector<int>& pointsOnLine, std::vector<int>& pointIds_ordered,
        std::vector<float>& lineTs_ordered )
{
    //clear working variables
    pointIds_unordered.clear();
    lineTs_unordered.clear();

    float startD = line.start.norm();
    float endD = line.end.norm();

    //compute t values
    Vector2f p, rel;

    for( std::vector<int>::iterator it = pointsOnLine.begin() ;
            it != pointsOnLine.end() ; ++it )
    {
        //get bearing coords of points
        int pid = *it;
        p.coeffRef(0) = (float) w[pid];
        p.coeffRef(1) = (float) db.vc[ id[ pid ] ];
        //project points onto line, find t values.
        rel = p - line.startBear;
        float proj = line.nBear.dot(rel) / line.nBearNorm;
        float xSqr = rel.squaredNorm() - proj*proj; //distance from line
        //if x is too far, don't include
        if( xSqr > std::pow(params.wNearThresh,2) )
            continue;

        //convert 2d 't'(tBear) to 3d 't' (lineT)
        float tBear = proj / line.nBearNorm;
        float lineT = tBear * line.length;

        // Remove points outside the 2D bounds (beyond wNearThres).
        // This section has been modified a lot- must account for the condition
        // when the line is parallel to sensed rays (startBear is very close to
        // endBear, all t value are outside of bounds).
        float pointD = D[pid];
        if( lineT < 0 )
        {
            // if further than wNearThresh of start point, don't include
            if( rel.squaredNorm() > std::pow(params.wNearThresh,2) )
                continue;
            else
                lineT = 0;

            //// if within depth bounds of line (start is behind the surface, so
            //// is of greatest depth)
            //if( endD < pointD && pointD < startD )
            //    lineT = 0;
            //else
            //    continue;
        }
        else if( lineT > line.length ) //same for other side of line
        {
            rel = p - line.endBear;
            if( rel.squaredNorm() > std::pow(params.wNearThresh,2) )
                continue;
            else
                lineT = line.length;

            //if( endD < pointD && pointD < startD )
            //    lineT = line.length;
            //else
            //    continue;
        }

        pointIds_unordered.push_back(pid);
        lineTs_unordered.push_back(lineT);
    }

    //sort by t
    int nPoints = lineTs_unordered.size();
    lineTs_argsort.resize(nPoints);
    lineTs_ordered.resize(nPoints);
    pointIds_ordered.resize(nPoints);

    argsort( lineTs_unordered.begin(), lineTs_unordered.end(), lineTs_argsort.begin(), lineTs_argsort.end() );
    for( int i=0 ; i<pointIds_unordered.size() ; i++ )
    {
        lineTs_ordered[i] = lineTs_unordered[ lineTs_argsort[i] ];
        pointIds_ordered[i] = pointIds_unordered[ lineTs_argsort[i] ];
    }
}



//check if a point is a point is near a line, and if it goes through the surface.
// recent change: pointLineDist is actually the gaussian max value along the line.
// we want to maximise this, not minimize...
inline void OccLine::checkLineProximity(int pid, float pointT,
        LineCoords& line, float& depthDiff, bool& isIntercepted,
        float& interceptT, float& pointLineDist)
{
    isIntercepted = false;
    //p1: associated position along the line (as determined via 2d proximity).
    p1 = line.start + pointT * line.n;

    //get d value (warning: overwriting p1 as a working variable)
    p1.z() += db.voffc[ id[pid] ];
    float lineD = p1.norm();
    depthDiff = D[pid] - lineD; //negative when line is behind surface

    //filter out non-object points here- depth check after will set it to unknown if it's another object's point
    int pcaId = pcaResults.pidToResult[pid];
    if( !isInObject[pid] || pcaId == -1 )
        return;

    //check for surface plane intersection:
    //use PCA results
    eval = pcaResults.evals.row(pcaId);
    evalSqrt = eval.cwiseSqrt();
    Map<Matrix3f> evec( pcaResults.evects.row(pcaId).data(), 3, 3 );
    globalToLocalR = evec.transpose();
    m << 1/evalSqrt.coeff(0), 0, 0,
         0, 1/evalSqrt.coeff(1), 0,
         0, 0, 1/evalSqrt.coeff(2);
    minv << evalSqrt.coeff(0), 0, 0,
            0, evalSqrt.coeff(1), 0,
            0, 0, evalSqrt.coeff(2);
    covinv << 1/eval.coeff(0), 0, 0,
              0, 1/eval.coeff(1), 0,
              0, 0, 1/eval.coeff(2); //is inverse of m

    //transform line to frame defined by this point's pca results.
    thisMeanP = pcaResults.meanP.row(pcaId).transpose().cast<float>();
    nLocal = globalToLocalR * line.n;
    dLocal = globalToLocalR * (line.start - thisMeanP);
    //transform line so that ellipse is a unit sphere
    nDash = m * nLocal;
    dDash = m * dLocal;
    //solve intercept
    float t = -dDash.dot(nDash) / nDash.squaredNorm();
    xDash = nDash * t + dDash;
    x = minv * xDash; //in mean-centred, evector aligned frame.
    //evaluate pdf at x:
    float exponent = x.transpose() * covinv * x;
    if( exponent > 20 )
        return;
    double val = pow_2pi_neg_1p5 * std::pow(eval.squaredNorm(), -0.25) * std::exp( -0.5 * exponent );
    if( val > params.pcaInterceptThresh && (t <= line.length) && (t >= 0) )
    {
        pointLineDist = val;
        isIntercepted = true;
        interceptT = t;
    }
}




void OccLine::lineTrace( float* bounds, std::vector< std::pair<int,int> >& pixels )
{
    //the line could pass the wrap around point at 0. add 2pi if needed, mod when adding pixel.
    float boundsWrap[4];
    memcpy(boundsWrap, bounds, 4*sizeof(float));
    if( fabs(bounds[0] - bounds[2]) > pi )
    {
        if( bounds[0] < bounds[2] )
            boundsWrap[0] += 2*pi;
        else
            boundsWrap[2] += 2*pi;
    }
    //find pixels of start/end
    int pixBounds[4];
    pixBounds[0] = (int)(boundsWrap[0]/(2*pi) * image.xRes);
    pixBounds[2] = (int)(boundsWrap[2]/(2*pi) * image.xRes);
    pixBounds[1] = image.FindElevationPixel(boundsWrap[1]);
    pixBounds[3] = image.FindElevationPixel(boundsWrap[3]);

    pixels.push_back( std::pair<int,int>(pixBounds[0], pixBounds[1]) );
    pixels.push_back( std::pair<int,int>(pixBounds[2], pixBounds[3]) );

    //line vector equation x = n*t + d
    float nAz = boundsWrap[2]-boundsWrap[0];
    float nEl = boundsWrap[3]-boundsWrap[1];
    //interating from lower to higher
    int startAz = (nAz < 0 ? pixBounds[2] : pixBounds[0]);
    int endAz = (nAz < 0 ? pixBounds[0] : pixBounds[2]);
    int startEl = (nEl < 0 ? pixBounds[3] : pixBounds[1]);
    int endEl = (nEl < 0 ? pixBounds[1] : pixBounds[3]);

    //iterate over az pixel boundaries (at every 2pi/2000)
    for( int x = startAz+1 ; x<=endAz ; x++ )
    {
        float az = x * (2*pi) / image.xRes;
        float t = ( az - boundsWrap[0] ) / nAz;
        float el = nEl*t + boundsWrap[1];
        int y = image.FindElevationPixel(el);
        pixels.push_back( std::pair<int,int>(x%image.xRes, y) );
    }

    //iterate over el pixel boundaries
    for( int y = startEl+1 ; y<=endEl ; y++ )
    {
        float el = image.elBounds[y]; //the bottom of the current pixel
        float t = ( el - boundsWrap[1] ) / nEl;
        float az = nAz*t + boundsWrap[0];
        int x = (int)( az/(2*pi) * image.xRes);
        pixels.push_back( std::pair<int,int>(x%image.xRes, y) );
    }
}




//iterate over longer dimension, selecting additional pixels (eg above/below for a mostly horizontal line).
void OccLine::lineTraceFat( float* bounds, std::vector< std::pair<int,int> >& pixels )
{
    //the line could pass the wrap around point at 0. add 2pi if needed, mod when adding pixel.
    float az1 = bounds[0];
    float el1 = bounds[1];
    float az2 = bounds[2];
    float el2 = bounds[3];
    if( fabs(az1 - az2) > pi )
    {
        if( az1 < az2 )
            az1 += 2*pi;
        else
            az2 += 2*pi;
    }

    //find pixels of start/end.
    int x1 = (int)(az1/(2*pi) * image.xRes);
    int x2 = (int)(az2/(2*pi) * image.xRes);

    //clip elevation pixels to be within image.
    //these get used if we iterate over elevation pixels.
    int y1 = image.FindElevationPixel(el1);
    if( y1 < 0 )
        y1 = 0;
    else if( y1 > 63 )
        y1 = 63;

    int y2 = image.FindElevationPixel(el2);
    if( y2 < 0 )
        y2 = 0;
    else if( y2 > 63 )
        y2 = 63;

    //line vector equation x = n*t + d
    float nAz = az2 - az1;
    float nEl = el2 - el1;
    //interating from lower (az/el) to higher
    int startX = (nAz < 0 ? x2 : x1);
    int endX = (nAz < 0 ? x1 : x2);
    int startY = (nEl < 0 ? y2 : y1);
    int endY = (nEl < 0 ? y1 : y2);

    if( endX-startX >= endY-startY )
    {
        //iterate over az pixel boundaries (at every 2pi/2000)
        for( int x = startX ; x<=endX ; x++ )
        {
            //note that start/end of line is defined by the equation (az = nAz*t + az1).
            //startAz & endAz just specify the iteration order over azimuth
            float az = x * (2*pi) / image.xRes;
            float t = ( az - az1 ) / nAz;
            float el = 0.0;
            if(t <= 0.0)
            {
                el = el1;
            }
            else if(t >= 1.0)
            {
                el = el2;
            }
            else
            {
                el = nEl*t + el1;
            }
            //note: cannot use y1/y2 for the el1/el2 cases above, as y1/y2 have been clipped to within 0-63
            int y = image.FindElevationPixel(el);
            if(y<0 || y>63)
                continue;
            int modx = x%image.xRes;
            pixels.push_back( std::pair<int,int>(modx,y) );
            //add above/below
            if(y>0)
                pixels.push_back( std::pair<int,int>(modx,y-1) );
            if(y<63)
                pixels.push_back( std::pair<int,int>(modx,y+1) );
        }
    }
    else
    {
        //iterate over el pixel boundaries
        for( int y = startY ; y<=endY ; y++ )
        {
            float el = image.elBounds[y]; //the bottom of the current pixel
            float t = ( el - el1 ) / nEl;
            float az = 0.0;
            int x = 0;
            if(t <= 0.0)
            {
                az = az1;
                x = x1;
            }
            else if(t >= 1.0)
            {
                az = az2;
                x = x2;
            }
            else
            {
                az = nAz*t + az1;
                x = ((int)( az/(2*pi) * image.xRes)) % image.xRes;
            }
            pixels.push_back( std::pair<int,int>(x,y) );
            //add left/right
            int xLeft = (x-1) % image.xRes;
            int xRight = (x+1) % image.xRes;
            pixels.push_back( std::pair<int,int>(xLeft,y) );
            pixels.push_back( std::pair<int,int>(xRight,y) );
        }
    }
}




//given a line defined by bounds, find all points near it
void OccLine::getPointsOnLine( float* bounds, std::vector<int>& points )
{
    //find pixels along the line
    //std::vector< std::pair<int,int> > pixelsOnLine;
    pixelsOnLine.clear();

    lineTraceFat(bounds, pixelsOnLine);
    for( int i=0 ; i<pixelsOnLine.size() ; i++ )
    {
        int* cellPoints;
        int nPoints;
        image.GetPoints(pixelsOnLine[i].first, pixelsOnLine[i].second, cellPoints, nPoints);
        for( int j=0 ; j<nPoints ; j++ )
        {
            points.push_back(cellPoints[j]);
        }
    }
}





void OccLine::traverseLineForIntercept( std::vector<int>& pointIds_ordered,
        std::vector<float>& lineTs_ordered, LineCoords& line, int& closestPid,
        float& value, unsigned char& status )
{
    //no data = unknown
    if( pointIds_ordered.size()==0 )
    {
        value = line.length/2;
        status = UNKNOWN;
    }

    //finding the closest point to the line- keep the 'running minimum' here:
    float closestPointLineDist;
    float closestPointT;

    //now go along the line, inspecting each point in order. This bit of code follows a handy state diagram.
    LineComputeState state = INFRONT;
    //reverse iterator- we're going from the line end (in front of the surface) to the line start (behind).
    for( int i = pointIds_ordered.size()-1 ; i >= 0 ; i-- )
    {
        if( state == INFRONT )
        {
            //check for break in data
            if( i+1 < pointIds_ordered.size() )
            {
                //double deltaT = priorRit->first - rit->first;
                float deltaT = lineTs_ordered[i+1] - lineTs_ordered[i];
                //find the length of the line segment in bearing space
                float lineLengthBear =  deltaT / line.length * line.nBearNorm;
                if( lineLengthBear > params.angleNoDataThresh )
                    state = OCCLUDED;
                //this point may be at a surface intercept- keep going
            }

            //TRANSITION TO: INTERCEPTED_AND_NEAR, results needed for later steps
            // check if point is near the line
            bool isIntercepted;
            float depthDiff, interceptT, pointLineDist;
            checkLineProximity( pointIds_ordered[i], lineTs_ordered[i], line, depthDiff, isIntercepted, interceptT, pointLineDist );
            if( isIntercepted )
            {
                state = INTERCEPTED_AND_NEAR;
                value = interceptT - line.length/2; //distance along line from centre
                closestPointLineDist = pointLineDist;
                closestPointT = interceptT;
                closestPid = pointIds_ordered[i];
                continue;
            }
            //else if( isNear )
            //{
            //    closestPointT = rit->first; //keep tabs on where this near point was- needed later
            //}
            else if( depthDiff >= 0 && state == INFRONT )
            {
                //update up to where the line was observed as empty
                value = lineTs_ordered[i] - line.length/2;
            }
            else if( depthDiff < 0 )//&& !isNear )
            {
                state = OCCLUDED;
                continue;
            }
        }

        else if( state == INTERCEPTED_AND_NEAR ) //come close to points recently- just checking near points for closer ones.
        {
            //when we've moved a given 3d distance away from the best intercept point, stop looking further.
            if( fabs(closestPointT - lineTs_ordered[i]) > params.lineLengthToCheck )
            {
                state = INTERCEPTED;
                break;
            }

            // check if point is near the line
            bool isIntercepted;
            float depthDiff, interceptT, pointLineDist;
            checkLineProximity( pointIds_ordered[i], lineTs_ordered[i], line, depthDiff, isIntercepted, interceptT, pointLineDist );
            //if its an intercept far away-- but initial intercepts may be really crap
            if( isIntercepted && fabs(interceptT - closestPointT) > params.lineLengthToCheck )
            {
                state = INTERCEPTED;
                break;
            }
            //if its a better intercept nearby
            if( isIntercepted && pointLineDist > closestPointLineDist )
            {
                value = interceptT - line.length/2; //distance along line from centre
                closestPointLineDist = pointLineDist;
                closestPointT = interceptT;
                closestPid = pointIds_ordered[i];
                continue;
            }
        }

        else if( state == OCCLUDED )
        {
            // check if point is near the line
            bool isIntercepted;
            float depthDiff, interceptT, pointLineDist;
            checkLineProximity( pointIds_ordered[i], lineTs_ordered[i], line, depthDiff, isIntercepted, interceptT, pointLineDist );
            if( isIntercepted )
            {
                state = INTERCEPTED_AND_NEAR;
                value = interceptT - line.length/2; //distance along line from centre
                closestPointLineDist = pointLineDist;
                closestPointT = interceptT;
                closestPid = pointIds_ordered[i];
                continue;
            }
        }

    } //breaks go to here

    //if line is INFRONT, check if line edges are outside the bounds of the image.
    float minEl = image.elBounds.front(); //highest point in image
    float maxEl = image.elBounds.back(); //lowest point
    if( state == INFRONT )
    {
        // if the start of the line (behind the object, the last point of
        // traversal) is outside the image, leave the value (should be at the
        // last visible point, known empty space)
        if( line.startBear[1] > maxEl || line.startBear[1] < minEl )
        {
            state = OCCLUDED;
        }
        // if the end of the line is outside the image, set the value to
        // maximum (unknown at the earliest point of traversal).
        if( line.endBear[1] > maxEl || line.endBear[1] < minEl )
        {
            state = OCCLUDED;
            value = line.length/2;
        }
    }

    // OLD METHOD OF IMAGE BOUND CHECKING
    //float vc_min = std::min(line.startBear[1], line.endBear[1]);
    //float vc_max = std::max(line.startBear[1], line.endBear[1]);
    //if( state == INFRONT && (vc_min < image.elBounds.front() || vc_max > image.elBounds.back()) )
    //{
    //    state = OCCLUDED;
    //}

    //if the length of line between the last point and it's end is too high, it's occluded (no data).
    if( state == INFRONT )
    {
        //we've just gone along the line, from the END to the START.
        // the t values indicate 3d distance from the START.
        float deltaT = lineTs_ordered[0];

        //find the length of the line segment in bearing space
        float lineLengthBear =  deltaT / line.length * line.nBearNorm;
        if( lineLengthBear > params.angleNoDataThresh )
        {
            state = OCCLUDED;
        }
    }

    //finished state changes, set line status output based on state
    if( state == OCCLUDED )
        status = UNKNOWN;
    else if( state == INTERCEPTED || state == INTERCEPTED_AND_NEAR )
        status = VALUE;
    else if( state == INFRONT )
    {
        status = EMPTY;
        value = -line.length/2; //keep values consistant with line ending
    }
}


