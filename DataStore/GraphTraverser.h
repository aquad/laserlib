/*! GraphTraverser.h
 *
 * Selects neighbours from a velodyne graph structure.
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
 *
 * The graph is basically a segmented range image. Each node is a 3d point,
 * edges connect points 'left' and 'right' to neighbouring points in azimuth,
 * and 'up and 'down' to neighbouring points in elevation. Edges are removed
 * loosely based on discontinous depth changes. The graph can then be traversed
 * for region selection tolerant to point density changes.  It is used largely
 * for surface normal generation.
 *
 * \author     Alastair Quadros
 * \date       03-09-2010
*/

#ifndef GRAPH_TRAVERSER
#define GRAPH_TRAVERSER

#include "export.h"
#include <vector>

//Eigen
#define EIGEN_DEFAULT_TO_ROW_MAJOR 1
#include <Eigen/Dense>

#include "BearingGraph.h"
#include "Common/ArrayTypes.h"
#include "Selector.h"

//@{
/*! Get 4 neighbours around each point.
For each point, select the neighbours left/right/up/down at least distance 'rad' away.
\param neighs - (output) neighbours, (n,4) int matrix, each row corresponds to a point's neighbours
\param graph - bearing graph, see <BearingGraphBuilder>"()", (n,4) int matrix
\param P - 3d points, (n,3) matrix
\param rad - Select points at least this far away.
*/
LASERLIB_DATASTORE_EXPORT void Get4Neighs( Graph& neighs, Graph& graph, Mat3<double>::type& P, double rad );

//! Get4Neighs with rad=0 actually does this... but this is slightly faster
LASERLIB_DATASTORE_EXPORT void Get4NeighsClosest( Graph& neighs, Graph& graph );

//! Neighbours must be valid
LASERLIB_DATASTORE_EXPORT void Get4NeighsValid( Graph& neighs, Graph& graph, Mat3<double>::type& P,
                                  bool* valid, double rad );
//@}


//! Grow a segment based on convexity criterion
LASERLIB_DATASTORE_EXPORT void RegionGrowConvex( unsigned int centre, unsigned int segId,
                       Vect<int>::type& segs, std::vector< unsigned int >& mergedSegs,
                       Graph& convex, Graph& graph,
                       Mat3<double>::type& surfNorms, double eta3 );

//! Segment a graph by region growing based on convexity
LASERLIB_DATASTORE_EXPORT void ConvexSegment( Vect<int>::type segs, Graph& convex, Graph& graph,
                    Mat3<double>::type& surfNorms,  double eta3 );


//this limits how many iterations a graph traversal takes- really a hack to fix an infinite loop.
const unsigned int maxIters = 100;

//this just reserves space in a vector, so timely reallocations don't occur often.
const unsigned int maxNeighs = 100;


/*! Base class for selecting regions using graphs.

SelectRegion will flood fill here- need to specify a real WithinCriteria() in a subclass.
Use copy constructors where necessary, can then be parallelised easily with OMP.
*/
class LASERLIB_DATASTORE_EXPORT GraphTraverser : virtual public Selector
{
public:
    GraphTraverser( Graph& graph_in )
        :   graph(graph_in)
    {
        neighs.reserve(maxNeighs);
    }

    boost::shared_ptr<Selector> clone()
    {
        return boost::shared_ptr<Selector>( new GraphTraverser(*this) );
    }

    //@{
    //! single, small graph traversals
    int FindNeigh( int pos, Direction dir );
    int FindNeighValid( int pos, Direction dir, bool* valid );
    int FindClosestNeigh( int pos, Direction dir );
    //@}

    //! Select a full region by graph traversal, bounded by WithinCriteria()
    std::vector<int>& SelectRegion( unsigned int centre );
    virtual bool WithinCriteria(int centre, int neigh)
        { return true; }

    //! use this for better speed if you're selecting large regions (>100 points)
    void allocateNeighs(int size)
        { neighs.reserve(size); }


protected:
    Graph graph;
    std::vector<int> neighs; //!< preallocated storage

private:
    void GetPerpendicular( Direction dir, Direction perpDirs[2] );
};


inline void GraphTraverser::GetPerpendicular( Direction dir, Direction perpDirs[2] )
{
    if( dir == LEFT || dir == RIGHT )
    {
        perpDirs[0] = UP;
        perpDirs[1] = DOWN;
    }
    else
    {
        perpDirs[0] = LEFT;
        perpDirs[1] = RIGHT;
    }
}




//For 3d spherical selections
class LASERLIB_DATASTORE_EXPORT GraphSphereSelector : public GraphTraverser
{
public:
    GraphSphereSelector( Graph& graph_in, Mat3<double>::type& P_in, double rad_in )
        :   GraphTraverser( graph_in ),
            P(P_in),
            rad(rad_in)
    {}

    bool WithinCriteria(int centre, int neigh)
    {
        rel = P.row(centre) - P.row(neigh);
        if( rel.norm() < rad )
        {
            return true;
        }
        return false;
    }

    boost::shared_ptr<Selector> clone()
    {
        return boost::shared_ptr<Selector>( new GraphSphereSelector(*this) );
    }

    void SetRadius(double r){ rad = r; }
    double GetRadius(){ return rad; }

    EIGEN_MAKE_ALIGNED_OPERATOR_NEW //needed because this class contains fixed sized eigen arrays.

private:
    Mat3<double>::type P;
    double rad;
    Eigen::RowVector3d rel;
};



//For a box selection in the range image space
class LASERLIB_DATASTORE_EXPORT BearingBoxSelector : public GraphTraverser
{
public:
    BearingBoxSelector( Graph& graph_in, Vect<double>::type& w_in,
                        Vect<int>::type& lNo_in, double az_in, int lasers_in )
        :   GraphTraverser( graph_in ),
            w(w_in),
            lNo(lNo_in),
            az(az_in),
            lasers(lasers_in)
    {}

    bool WithinCriteria(int centre, int neigh)
    {
        if( (fabs(w(neigh) - w(centre)) <= az) &&
            (abs(lNo(neigh) - lNo(centre)) <= lasers) )
            return true;
        return false;
    }

    boost::shared_ptr<Selector> clone()
    {
        return boost::shared_ptr<Selector>( new BearingBoxSelector(*this) );
    }

    void SetAz(double az_in){ az = az_in; }
    double GetAz(){ return az; }

    void SetLasers(int lasers_in){ lasers = lasers_in; }
    int GetLasers(){ return lasers; }

private:
    Vect<double>::type w;
    Vect<int>::type lNo;
    double az;
    int lasers;
};



//just so we can use the output from Get4Neighs on point-wise functions like PCA.
class LASERLIB_DATASTORE_EXPORT PreFourNeighSelector : virtual public Selector
{
public:
    PreFourNeighSelector( Graph& _neighs_all )
        :   neighs_all(_neighs_all)
    {
        neighs.reserve(4);
    }

    std::vector<int>& SelectRegion( unsigned int centre )
    {
        neighs.clear();
        for( int i=0 ; i<4 ; i++ )
        {
            int n = neighs_all(centre,i);
            if( n!=-1 )
                neighs.push_back(n);
        }
        return neighs;
    }

    boost::shared_ptr<Selector> clone()
    {
        return boost::shared_ptr<Selector>( new PreFourNeighSelector(*this) );
    }

protected:
    Graph neighs_all;
    std::vector<int> neighs;
};


#endif //GRAPH_TRAVERSER
