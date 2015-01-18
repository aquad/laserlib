/*! BearingGraph.cpp
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
 * \date       18-05-2011
*/

#include <iostream>
#include <algorithm>
#include <vector>

#include <boost/math/constants/constants.hpp>
const double pi = boost::math::constants::pi<double>();

#include "BearingGraph.h"
#include "Common/argsort.h"
using namespace boost;
using namespace Eigen;

BearingGraphBuilder::BearingGraphBuilder( VelodyneDb& _db, unsigned int _maxNumPoints, float _wThresh )
    :   maxNumPoints( _maxNumPoints ),
        maxPointsPerLaser( _maxNumPoints/64 ),
        wThresh( _wThresh ),
        db(_db),
        graph( _maxNumPoints )
{
    // sort laser ids from bottom (+ve) to top (-ve)
    // idToLNo = vc_id.argsort().argsort()
    boost::array<double,64> vc;
    memcpy(&vc[0], _db.vc, 64*sizeof(double));
    boost::array<unsigned int,64> vc_argsort;

    //consider vc as a map from laser id to vertical angle.
    //vc_argsort orders the lasers from bottom to top (max = bottom = first).
    //so it maps an ordered 'laser number' to laser id.
    argsort( vc.begin(), vc.end(), vc_argsort.begin(), vc_argsort.end() );
    std::reverse(vc_argsort.begin(), vc_argsort.end());

    //need a mapping from laser id to it's ordered 'laser number'.
    argsort( vc_argsort.begin(), vc_argsort.end(), idToLaserNo.begin(), idToLaserNo.end() );

    // allocate working mem
    for( int i=0 ; i<64 ; i++ )
        { order[i].reserve(maxPointsPerLaser); }
}



void BearingGraphBuilder::BuildGraph( unsigned char* id, double* w, unsigned int nPoints )
{
    if( nPoints > maxNumPoints )
    {
        graph.reset(nPoints);
        maxNumPoints = nPoints;
    }

    //clear preallocated arrays
    for( int i=0 ; i<64 ; i++ )
        { order[i].clear(); }
    graph.clear();

    //sort points
    boost::array<bool,64> laserIsPresent;
    laserIsPresent.assign(false);

    for( unsigned int i=0 ; i<nPoints ; i++ )
    {
        unsigned char lNo = idToLaserNo[ id[i] ];
        //dont accept out-of-order points
        if( order[lNo].size() > 0 )
        {
            double lastW = w[order[lNo].back()];
            double wDiff = w[i] - lastW; //lastW must be smaller, so wDiff is +ve.
            if( wDiff < 0 && wDiff > -pi) { continue; } //on wraparound, wDiff is large negative.
        }
        order[lNo].push_back(i);
        laserIsPresent[lNo] = true;
    }

    std::vector<unsigned char> lasersPresent;
    for( int i=0 ; i<64 ; i++ )
    {
        if( laserIsPresent[i] )
            lasersPresent.push_back(i);
    }

    //go through the lasers present, in order. Keep track of the nearest point (in azimuth) in the laser above.
    for( int i=0 ; i<lasersPresent.size() ; i++ )
    {
        int lNo = lasersPresent[i];
        bool isLastLaser = false;
        int lNoAbove = 0;
        int orderIdAbove = 0; //keep track of closest laser above
        if( i==lasersPresent.size()-1 ) //last laser?
            isLastLaser = true;
        else
            lNoAbove = lasersPresent[i+1];

        //for each point in this laser
        for( int j=0 ; j<order[lNo].size() ; j++ )
        {
            int pid = order[lNo][j];
            //connect left/right            
            //int leftOrderId = (j-1) % (int)nPointsInLaser[lNo]; //not working!
            int leftOrderId = j-1 > 0 ? j-1 : order[lNo].size()-1;
            int leftId = order[lNo][leftOrderId];
            if( fabs(w[pid] - w[leftId]) < wThresh )
            {
                graph(pid, LEFT) = leftId;
                graph(leftId, RIGHT) = pid;
            }

            //connect up/down...
            if( isLastLaser )
                continue;

            //find closest laser above
            int aboveId = 0;

            //prevent infinite loops due to duplicate points...
            int initialAbove = orderIdAbove;
            while(1)
            {
                aboveId = order[lNoAbove][orderIdAbove];
                if( order[lNoAbove].size() == 1 )
                    break;

                int aboveNextOrderId = (orderIdAbove+1) < order[lNoAbove].size() ?
                            (orderIdAbove+1) : 0;
                int aboveNextId = order[lNoAbove][aboveNextOrderId];

                double currentWDist = fabs(w[pid] - w[aboveId]);
                if( currentWDist > pi ) { currentWDist = 2*pi-currentWDist; }

                double nextWDist = fabs(w[pid] - w[aboveNextId]);
                if( nextWDist > pi ) { nextWDist = 2*pi-nextWDist; }

                if( nextWDist <= currentWDist ) //next is closer
                    orderIdAbove = aboveNextOrderId;
                else
                    break;

                if( orderIdAbove == initialAbove )
                    break;
            }

            if( fabs(w[pid] - w[aboveId]) < wThresh )
            {
                graph(pid, UP) = aboveId;
                graph(aboveId, DOWN) = pid;
            }
        }
    }
}


void BearingGraphBuilder::CleanGraph( Mat3<double>::type& P, MapVecXd& D, float maxLength, float relThresh, float convThresh )
{
    //TODO: compute all left & right lengths first, will avoid half the
    //computation time.

    //limit relative link lengths.
    //remove highly convex/concave links.
    int nPoints = P.rows();
    for( unsigned int i=0 ; i<nPoints ; i++ )
    {
        if( graph(i,LEFT) != -1 && graph(i,RIGHT) != -1 )
        {
            RowVector3d leftRel = P.row(i) - P.row( graph(i,LEFT) );
            RowVector3d rightRel = P.row(i) - P.row( graph(i,RIGHT) );
            double leftLength = leftRel.norm();
            double rightLength = rightRel.norm();

            double leftD = D[i] - D[ graph(i,LEFT) ];
            double rightD = D[i] - D[ graph(i,RIGHT) ];

            //relThresh > 1, so it's good if this is false:
            if( rightLength > leftLength * relThresh )
            {
                graph.remove(i,RIGHT);
            }
            else if( leftLength > rightLength * relThresh )
            {
                graph.remove(i,LEFT);
            }

            //if range differences are both positive or both negative (and large), disconnect both
            if( (leftD > convThresh && rightD > convThresh) ||
                (leftD < -convThresh && rightD < -convThresh) )
            {
                graph.remove(i,RIGHT);
                graph.remove(i,LEFT);
            }
        }

        if( graph(i,UP) != -1 && graph(i,DOWN) != -1 )
        {
            RowVector3d upRel = P.row(i) - P.row( graph(i,UP) );
            RowVector3d downRel = P.row(i) - P.row( graph(i,DOWN) );
            double upLength = upRel.norm();
            double downLength = downRel.norm();

            double upD = D[i] - D[ graph(i,UP) ];
            double downD = D[i] - D[ graph(i,DOWN) ];

            if( upLength > downLength * relThresh )
            {
                graph.remove(i,UP);
            }
            else if( downLength > upLength * relThresh )
            {
                graph.remove(i,DOWN);
            }

            //if range differences are both positive or both negative (and large), disconnect both
            if( (upD > convThresh && downD > convThresh) ||
                (upD < -convThresh && downD < -convThresh) )
            {
                graph.remove(i,DOWN);
                graph.remove(i,UP);
            }
        }
    }

    //remove long links.
    for( unsigned int i=0 ; i<nPoints ; i++ )
    {
        int neighLeft = graph(i,LEFT);
        if( neighLeft != -1 )
        {
            RowVector3d leftRel = P.row(i) - P.row( graph(i,LEFT) );
            double leftLength = leftRel.norm();
            if( leftLength > maxLength )
            {
                graph.remove(i,LEFT);
            }
        }
        int neighUp = graph(i,UP);
        if( neighUp != -1 )
        {
            RowVector3d upRel = P.row(i) - P.row( graph(i,UP) );
            double upLength = upRel.norm();
            if( upLength > maxLength )
            {
                graph.remove(i,UP);
            }
        }
    }
}



// Deprecated- delete at some point
void BearingGraphBuilder::CleanGraphFast( unsigned short* D, unsigned int nPoints, int maxLength, float relThresh, int convThresh )
{
    //limit relative link lengths.
    //remove highly convex/concave links.
    for( unsigned int i=0 ; i<nPoints ; i++ )
    {
        if( graph(i,LEFT) != -1 && graph(i,RIGHT) != -1 )
        {
            int leftD = D[i] - D[ graph(i,LEFT) ];
            int rightD = D[i] - D[ graph(i,LEFT) ];
            int leftLength = abs(leftD);
            int rightLength = abs(rightD);
            //relThresh > 1, so it's good if this is false:
            if( rightLength > leftLength * relThresh )
            {
                graph.remove(i,RIGHT);
            }
            else if( leftLength > rightLength * relThresh )
            {
                graph.remove(i,LEFT);
            }

            //if both positive or both negative (and large), disconnect both
            if( (leftD > convThresh && rightD > convThresh) ||
                (leftD < -convThresh && rightD < -convThresh) )
            {
                graph.remove(i,RIGHT);
                graph.remove(i,LEFT);
            }
        }

        if( graph(i,UP) != -1 && graph(i,DOWN) != -1 )
        {
            int upD = D[i] - D[ graph(i,UP) ];
            int downD = D[i] - D[ graph(i,DOWN) ];
            int upLength = abs(upD);
            int downLength = abs(downD);
            if( downLength > upLength * relThresh )
            {
                graph.remove(i,DOWN);
            }
            else if( upLength > downLength * relThresh )
            {
                graph.remove(i,UP);
            }

            //if both positive or both negative (and large), disconnect both
            if( (upD > convThresh && downD > convThresh) ||
                (upD < -convThresh && downD < -convThresh) )
            {
                graph.remove(i,DOWN);
                graph.remove(i,UP);
            }
        }
    }

    //remove long links.
    for( unsigned int i=0 ; i<nPoints ; i++ )
    {
        int neighLeft = graph(i,LEFT);
        if( neighLeft != -1 )
        {
            if( abs( D[i] - D[neighLeft] ) > maxLength )
            {
                graph.remove(i,LEFT);
            }
        }
        int neighUp = graph(i,UP);
        if( neighUp != -1 )
        {
            if( abs( D[i] - D[neighUp] ) > maxLength )
            {
                graph.remove(i,UP);
            }
        }
    }
}


