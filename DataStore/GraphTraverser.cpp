/*! GraphTraverser.cpp
 *
 * Copyright (C) 2010 Alastair Quadros.
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
 * \date       03-09-2010
*/

#include <iostream>
#include <string.h>
#include <list>
#include <set>
#include <algorithm>

#include "GraphTraverser.h"

using namespace std;
using namespace Eigen;



void Get4Neighs( Graph& neighs, Graph& graph, Mat3<double>::type& P, double rad )
{
    GraphSphereSelector sel( graph, P, rad );
    for( int i=0 ; i<graph.rows() ; i++ ) //for each point
    {
        for( int j=0 ; j<4 ; j++ ) //for each direction
        {
            neighs(i,j) = sel.FindNeigh(i, static_cast<Direction>(j));
        }
    }
}



//for each point, select the closest neighbours left/right/up/down.
void Get4NeighsClosest( Graph& neighs, Graph& graph )
{
    GraphTraverser trav( graph );
    for( int i=0 ; i<graph.rows() ; i++ )
    {
        for( int j=0 ; j<4 ; j++ )
        {
            neighs(i,j) = trav.FindClosestNeigh(i, static_cast<Direction>(j));
        }
    }
}


//as before, but ensures the 4 neighbours have non-zero surface normals
void Get4NeighsValid( Graph& neighs, Graph& graph, Mat3<double>::type& P, bool* valid, double rad )
{
    GraphSphereSelector sel( graph, P, rad );
    for( int i=0 ; i<graph.rows() ; i++ )
    {
        for( int j=0 ; j<4 ; j++ )
        {
            neighs(i,j) = sel.FindNeighValid(i, static_cast<Direction>(j), valid);
        }
    }
}



//GraphTraverser-----------------------


//traverse the graph in the given direction, finding the point closest to but not beyond dist.
//will travel perpendicular to dir for one step to try and get past small disconnections (travel diagonally).
//if there are no neighbours within dist in that direction, it will return the closest neighbour in that direction.
int GraphTraverser::FindNeigh( int pos, Direction dir )
{
    Direction perpDirs[2];
    GetPerpendicular(dir, perpDirs);

    int last = pos;
    int current = 0;
    int count = 0;

    while(1)
    {
        current = graph(last,dir);

        if( current == -1 )
        {
            //check other neighbours for a way through
            for( int i=0 ; i<2 ; i++ )
            {
                int perp = graph(last, perpDirs[i]);
                if( perp == -1 )
                    continue;
                current = graph(perp, dir);
                if( current == -1 )
                    continue;
                break;
            }

            //no way through
            if( current == -1 )
            {
                if( last == pos )
                    return -1;
                else
                    return last;
            }
            //otherwise, continue onward
        }

        if( !WithinCriteria(current, pos) )
        {
            if( last == pos )
                return current;
            return last;
        }

        last = current;
        count++;
        if( count > maxIters )
        {
            if( last == pos )
                return current;
            return last;
        }
    }
}




//same traversal as before, but want to return a point with a valid surface normal.
int GraphTraverser::FindNeighValid( int pos, Direction dir, bool* valid )
{
    Direction perpDirs[2];
    GetPerpendicular(dir, perpDirs);

    int lastValid = pos; //last position with a surf norm
    int last = pos; //previous position
    int current = 0; //current position
    int count = 0;

    while(1)
    {
        current = graph(last,dir);

        if( current == -1 )
        {
            //check other neighbours for a way through
            for( int i=0 ; i<2 ; i++ )
            {
                int perp = graph(last, perpDirs[i]); //point in perpendicular direction
                if( perp == -1 )
                    continue;
                current = graph(perp, dir); //point in corrent direction, one over to the side
                if( current == -1 )
                    continue;
                break;
            }

            //no way through
            if( current == -1 )
            {
                if( lastValid == pos ) //didn't get anywhere
                    return -1;
                return lastValid;
            }
            //otherwise, continue onward
        }

        if( !WithinCriteria(current, pos) )
        {
            //if we didnt get anywhere, return the next anyway.
            //might need to indicate scale as well...
            if( lastValid == pos )
            {
                if( valid[current] )
                    return current;
                return -1;
            }
            return lastValid;
        }

        last = current;
        if( valid[current] )
            lastValid = current;

        count++;
        if( count > maxIters )
        {
            if( last == pos )
                return current;
            return last;
        }
    }
}




//traverse the graph in the given direction, finding the point closest to but not beyond dist.
//will travel perpendicular to dir for one step to try and get past small disconnections (travel diagonally).
//if there are no neighbours within dist in that direction, it will return the closest neighbour in that direction.
int GraphTraverser::FindClosestNeigh( int pos, Direction dir )
{
    Direction perpDirs[2];
    GetPerpendicular(dir, perpDirs);

    int neigh = graph(pos,dir);
    if( neigh == -1 )
    {
        //check other neighbours for a way through
        for( int i=0 ; i<2 ; i++ )
        {
            int perp = graph(pos, perpDirs[i]);
            if( perp == -1 )
                continue;
            neigh = graph(perp, dir);
            if( neigh == -1 )
                continue;
            break;
        }
    }
    return neigh;
}




//select a region, basic breadth-first graph search:
//- starting at the top of the 'to do' list, get the point's neighbours.
//   if they're not in the accepted list yet, check if they're inside
//   the constraints. if so, add to the end of both lists.
//- pop the top of the 'to do' list, go to the next.
//- when the 'to do' list is empty, it's done.
//maybe try boost graph search functions?
std::vector<int>& GraphTraverser::SelectRegion( unsigned int centre )
{
    //neighbours to look into
    std::list<unsigned int> todo;
    todo.push_back( centre );

    //accepted neighbours
    neighs.clear();
    neighs.push_back( centre );

    std::set<unsigned int> done;
    done.insert( centre );

    while( todo.size() != 0 )
    {
        //top of todo list: get the point's neighbours
        unsigned int thisPoint = todo.front();
        todo.pop_front();
        for( unsigned int i=0 ; i<4 ; i++ )
        {
            int thisNeigh = graph( thisPoint, i );
            //skip disconnected values
            if( thisNeigh == -1 )
                continue;

            //check if points are done already
            if( done.end() != done.find(thisNeigh) )
                continue;

            //process this neighbour:
            //mark as done
            done.insert( thisNeigh );

            //check if point is inside the radius
            if( WithinCriteria(centre, thisNeigh) )
            {
                //add to neighs
                neighs.push_back( thisNeigh );
                //add to todo
                todo.push_back( thisNeigh );
            }
        }
    }
    return neighs;
}




//Convex Segmenter----------------
void RegionGrowConvex( unsigned int centre, unsigned int segId,
        Vect<int>::type& segs, std::vector< unsigned int >& mergedSegs,
        Graph& convex, Graph& graph,
        Mat3<double>::type& surfNorms, double eta3 )
{
    //neighbours to look into
    std::list<unsigned int> todo;
    todo.push_back( centre );

    //neighbours already processed locally
    std::set<unsigned int> done;
    done.insert( centre );

    segs(centre) = segId;

    while( todo.size() != 0 )
    {
        //top of todo list: get the point's neighbours
        unsigned int thisPoint = todo.front();
        todo.pop_front();
        for( unsigned int i=0 ; i<4 ; i++ )
        {
            int thisNeigh = graph( thisPoint, i );
            //skip disconnected values
            if( thisNeigh == -1 )
                continue;

            //check if points are done already
            if( done.end() != done.find(thisNeigh) )
                continue;

            //process this neighbour:
            //mark as done
            done.insert( thisNeigh );

            //not sure why the 'done' check doesn't work...
            if( segs(thisNeigh) == segId )
                continue;

            //check if angle at thisPoint, direction i, is above threshold (convex)
            if( convex(thisPoint,i) ||
                ( ( abs(surfNorms(thisPoint,2)) < eta3 ) &&
                  ( abs(surfNorms(thisNeigh,2)) < eta3 ) ) )
            {
                //check if already assigned- should this even happen?
                //local convexity is symmetric, but other criteria? and graph not symmetric.
                if( segs(thisNeigh) != 0 )
                {
                    //merge segments
                    //std::cout << "merging with seg " << segs(thisNeigh) << std::endl;
                    mergedSegs[segs(thisNeigh)] = segId;
                }
                else
                {
                    //assign segId
                    segs(thisNeigh) = segId;
                    //add to todo
                    todo.push_back( thisNeigh );
                }
            }
        }
    }
}



void ConvexSegment( Vect<int>::type segs, Graph& convex, Graph& graph,
                    Mat3<double>::type& surfNorms, double eta3 )
{
    unsigned int segId = 1;
    std::vector< unsigned int > mergedSegs;
    mergedSegs.reserve( graph.rows() );

    //can't multithread- merging segments will fail
    for( unsigned int i=0 ; i<graph.rows() ; i++ )
    {
        if( segs(i) != 0 )
            continue;
        //std::cout << "seg: " << segId << ", point " << i << std::endl;
        mergedSegs.push_back(segId);
        RegionGrowConvex( i, segId, segs, mergedSegs,
                          convex, graph, surfNorms, eta3 );
        segId++;
    }

    //compute final segment id from mergedSegs
    for( unsigned int i=0 ; i<segs.rows() ; i++ )
    {
        segs(i) = mergedSegs[ segs(i) ];
    }
}


