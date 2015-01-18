/*! BearingGraph.h
 * Classes for building a bearing graph from velodyne data.
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
 *
 * \details
 * The graph is basically a segmented range image. Each node is a 3d point,
 * edges connect points 'left' and 'right' to neighbouring points in azimuth,
 * and 'up and 'down' to neighbouring points in elevation. Edges are removed
 * loosely based on discontinous depth changes. The graph can then be traversed
 * for region selection tolerant to point density changes.  It is used largely
 * for surface normal generation.
 *
 * \author  Alastair Quadros
 * \date    18-05-2011
*/

#ifndef BEARING_GRAPH
#define BEARING_GRAPH

#include <vector>
#include <boost/array.hpp>
#include <boost/shared_array.hpp>
#include "VelodyneDb.h"
#include "Common/ArrayTypes.h"


enum Direction { LEFT=0, RIGHT=1, UP=2, DOWN=3 };


//! Get the opposite direction (eg- OppositeDir(LEFT) = RIGHT)
inline Direction OppositeDir( Direction dir )
{
    //if odd, subtract 1. if even, add 1.
    return static_cast<Direction>( (dir == 0 || dir == 2) ? dir+1 : dir-1 );
}


//! Holds the bearing graph data (an int array)
struct BearingGraph
{
    BearingGraph(unsigned int _maxNumPoints=300000)
        :   maxNumPoints(_maxNumPoints),
            graph( new int[_maxNumPoints*4] )
    {}

    virtual ~BearingGraph()    {}

    /*! Access neighbour of given point id
    \param id- point id
    \param dir- direction (LEFT, RIGHT, UP, DOWN)
    */
    int& operator()(unsigned int id, Direction dir)
        { return graph[ id*4 + dir ]; }

    /*! Remove link to neighbour (both entries in the matrix)
    \param id- point id
    \param dir- direction (LEFT, RIGHT, UP, DOWN)
    */
    void remove(unsigned int id, Direction dir)
    {
        int neigh = operator()(id,dir);
        if( neigh != -1 )
        {
            //remove link the other way first
            operator()(neigh, OppositeDir(dir)) = -1;
            operator()(id,dir) = -1;
        }
    }

    //! Fill the matrix with -1, clearing it
    void clear()
        { memset(graph.get(), -1, maxNumPoints*4*sizeof(int)); }

    //! Make a new array of different size
    void reset(unsigned int _maxNumPoints)
    {
        maxNumPoints = _maxNumPoints;
        graph.reset( new int[maxNumPoints*4] );
    }

    unsigned int maxNumPoints;
    boost::shared_array<int> graph;
};


/*! Builds a bearing graph from velodyne data.

Uses the integer azimuth and distance values from the velodyne to speed things up.
*/
class BearingGraphBuilder
{
public:

    /*! \param db- calibration parameters
    \param maxNumPoints- maximum number of points in the graph
    \param wThresh- don't connect points with an azimuth (rad) distance greater than this
    */
    BearingGraphBuilder( VelodyneDb& db, unsigned int maxNumPoints=300000, float wThresh=0.0349 );

    /*! Build the graph, connecting all points together.
    \param id- laser ids, range [0,63]. (nPoints,) array
    \param w- azimuth values, range (0,2pi), rad. (nPoints,) array
    */
    void BuildGraph( unsigned char* id, double* w, unsigned int nPoints );

    //! Return a pointer to the graph's int array.
    int* GetGraphPtr()
        { return graph.graph.get(); }

    BearingGraph& GetGraph()
        { return graph; }

    /*! Remove edges across discontinuities.

    Edge lengths are estimated by depth differences.
    \param P- 3D points, (nPoints,3) array
    \param D- range values, metres. (nPoints,) array

    Remove an edge:
    \param maxLength- If an edge is longer than this.
    \param relThresh- If one edge is more than many times longer than the other.
    \param convThresh- If both are more than this long, and convex/concave.
    */
    void CleanGraph( Mat3<double>::type& P, MapVecXd& D, float maxLength=5, float relThresh=3, float convThresh=0.5 );


    /*! Remove egdes across discontinuities.

    Formerly CleanGraph, this function assumes edge lengths can be estimated by
    depth differences- this is actually not the case for many points.

    \param D- range values, metres = D/500. (nPoints,) array

    Remove an edge:
    \param maxLength- If an edge is longer than this.
    \param relThresh- If one edge is more than many times longer than the other.
    \param convThresh- If both are more than this long, and convex/concave.
    */
    void CleanGraphFast( unsigned short* D, unsigned int nPoints, int maxLength=5*500, float relThresh=3, int convThresh=0.5*500 );

private:
    unsigned int maxNumPoints;
    unsigned int maxPointsPerLaser;
    float wThresh;
    boost::array< unsigned char, 64 > idToLaserNo;
    boost::array< std::vector<unsigned int>, 64 > order;
    VelodyneDb db;
    BearingGraph graph;
};




#endif //BEARING_GRAPH
