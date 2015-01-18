/*! GeometricConsistency.cpp
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
 * \date       15-07-2011
*/

#include <iostream>
#include <deque>
#include <set>

#include <boost/config.hpp>
#include <boost/graph/graph_concepts.hpp>
#include <boost/graph/lookup_edge.hpp>
#include <boost/concept/detail/concept_def.hpp>
#include <boost/graph/bron_kerbosch_all_cliques.hpp>
#include <boost/graph/adjacency_list.hpp>
#include <boost/graph/connected_components.hpp>

#include "Common/argsort.h"
#include "GeometricConsistency.h"


using namespace Eigen;
using namespace boost;



//find & store maximum clique
struct max_clique_store_visitor
{
    max_clique_store_visitor( std::size_t& _max, std::vector<int>& _maxClique)
        : maximum(_max), maxClique(_maxClique)
    {}

    template <typename Clique, typename Graph>
    inline void clique(const Clique& p, const Graph& g)
    {
        if( p.size() > maximum )
        {
            //std::cout << "p: " << p.size() << ">" << maximum << ", maxClique:";
            maxClique.assign( p.begin(), p.end() ); //expensive...
            maximum = p.size();
            //std::cout << maxClique.size() << std::endl;
        }
    }

    std::size_t& maximum;
    std::vector<int>& maxClique;
};



//modified from boost's code to actually work...
template <typename Graph, typename Visitor>
void bron_kerbosch_all_cliques_local(const Graph& g, Visitor vis, std::size_t min)
{
    using namespace boost;
    function_requires< IncidenceGraphConcept<Graph> >();
    function_requires< VertexListGraphConcept<Graph> >();
    function_requires< AdjacencyMatrixConcept<Graph> >(); // Structural requirement only
    typedef typename graph_traits<Graph>::vertex_descriptor Vertex;
    typedef typename graph_traits<Graph>::vertex_iterator VertexIterator;
    typedef std::vector<Vertex> VertexSet;
    typedef std::deque<Vertex> Clique;
    function_requires< CliqueVisitorConcept<Visitor,Clique,Graph> >();

    // NOTE: We're using a deque to implement the clique, because it provides
    // constant inserts and removals at the end and also a constant size.

    VertexIterator i, end;
    boost::tie(i, end) = vertices(g);
    VertexSet cands(i, end);    // start with all vertices as candidates
    VertexSet nots;             // start with no vertices visited

    Clique clique;              // the first clique is an empty vertex set
    detail::extend_clique(g, clique, cands, nots, vis, min);
}



void FindMaxClique( pairMatrix& matches,
                 Mat3<float>::type& testP, Mat3<float>::type& testVect,
                 Mat3<float>::type& trainP, Mat3<float>::type& trainVect,
                 float distThresh, float minDist, float angleThresh, int minCliqueSize, int maxEdges,
                 std::vector<int>& maxClique )
{
    //build graph
    typedef adjacency_list<vecS, vecS, undirectedS> Graph;
    Graph g( matches.rows() );
    for(int i = 0; i < matches.rows(); ++i)
    {
        int testSeed = matches(i,0);
        int trainSeed = matches(i,1);

        for(int j = 0; j < matches.rows(); ++j)
        {
            int testId = matches(j,0);
            int trainId = matches(j,1);

            //seed pair match is unique
            if( testId==testSeed || trainId==trainSeed )
                continue;

            RowVector3f testRel = testP.row(testId) - testP.row(testSeed);
            RowVector3f trainRel = trainP.row(trainId) - trainP.row(trainSeed);
            double testDist = testRel.norm();
            double trainDist = trainRel.norm();
            if( fabs(testDist - trainDist) > distThresh )
                continue;

            if( testDist < minDist || trainDist < minDist )
                continue;

            //angles
            float testVectorDot = testVect.row(testId).dot( testVect.row(testSeed) );
            float testVectorAngle = acos(testVectorDot);
            float trainVectorDot = trainVect.row(trainId).dot( trainVect.row(trainSeed) );
            float trainVectorAngle = acos(trainVectorDot);
            if( fabs(testVectorAngle - trainVectorAngle) > angleThresh )
                continue;

            add_edge( i, j, g);
        }
    }
    std::cout << num_edges(g) << " edges" << std::endl;
    if( num_edges(g) > maxEdges && maxEdges != 0 )
        return;

    std::size_t max = minCliqueSize;
    max_clique_store_visitor vis(max, maxClique);
    bron_kerbosch_all_cliques_local(g, vis, minCliqueSize);
    //std::cout << "clique size:" << maxClique.size() << std::endl;
}



void FindComponents( pairMatrix& matches,
                 Mat3<float>::type& testP, Mat3<float>::type& testVect,
                 Mat3<float>::type& trainP, Mat3<float>::type& trainVect,
                 float distThresh, float angleThresh,
                 Vect<int>::type& components )
{
    //build graph
    typedef adjacency_list<vecS, vecS, undirectedS> Graph;
    Graph g( matches.rows() );

    for(int i = 0; i < matches.rows(); ++i)
    {
        int testSeed = matches(i,0);
        int trainSeed = matches(i,1);

        for(int j = 0; j < matches.rows(); ++j)
        {
            int testId = matches(j,0);
            int trainId = matches(j,1);

            //seed pair match is unique
            if( testId==testSeed || trainId==trainSeed )
                continue;

            RowVector3f testRel = testP.row(testId) - testP.row(testSeed);
            RowVector3f trainRel = trainP.row(trainId) - trainP.row(trainSeed);
            if( fabs(testRel.norm() - trainRel.norm()) > distThresh )
                continue;

            //angles
            float testVectorDot = testVect.row(testId).dot( testVect.row(testSeed) );
            float testVectorAngle = acos(testVectorDot);
            float trainVectorDot = trainVect.row(trainId).dot( trainVect.row(trainSeed) );
            float trainVectorAngle = acos(trainVectorDot);
            if( fabs(testVectorAngle - trainVectorAngle) > angleThresh )
                continue;

            add_edge( i, j, g);
        }
    }
    std::cout << num_edges(g) << " edges" << std::endl;
    // instead of rigid clique, just find connected regions.
    //std::vector<int> component( num_vertices(g) );
    int num = connected_components(g, components.data() );
    std::cout << num << " components" << std::endl;
}




//old function
void FindConsistentSet( Vect<int>::type& seeds,
                       Vect<int>::type& testIds,
                       Mat3<float>::type& testP, Mat3<float>::type& testVect,
                       Vect<int>::type& trainIds,
                       Mat3<float>::type& trainP, Mat3<float>::type& trainVect,
                       MapMatXf& rmse,
                       float& bestScore, int& bestSeed, std::map<int,int>& bestPairs,
                       float distThresh, float angleThresh, float rmseWeight )
{
    bestScore = -10000.0;
    std::auto_ptr< std::map<int,int> > best_pairs_ptr(new std::map<int,int>());

    for( int i=0 ; i<seeds.size() ; i++ )
    {
        //get the current seed pair
        int seed = seeds(i);
        int testSeed = testIds(seed);
        int trainSeed = trainIds(seed);

        //find all pairs consistent with the seed
        std::auto_ptr< std::map<int,int> > pairs_ptr(new std::map<int,int>());
        std::map<int,int>& pairs = *pairs_ptr.get();
        //for each pair
        for( int j=0 ; j<testIds.size() ; j++ )
        {
            int testId = testIds(j);
            int trainId = trainIds(j);
            //compute the distance of each test/training point from the seed
            RowVector3f testRel = testP.row(testId) - testP.row(testSeed);
            RowVector3f trainRel = trainP.row(trainId) - trainP.row(trainSeed);
            if( fabs(testRel.norm() - trainRel.norm()) > distThresh )
                continue;

            //angles
            float testVectorDot = testVect.row(testId).dot( testVect.row(testSeed) );
            float testVectorAngle = acos(testVectorDot);
            float trainVectorDot = trainVect.row(trainId).dot( trainVect.row(trainSeed) );
            float trainVectorAngle = acos(trainVectorDot);
            if( fabs(testVectorAngle - trainVectorAngle) > angleThresh )
                continue;

            //if this test point is already matched with a training point
            if( pairs.count(testId) )
            {
                //overwrite if this has a lower rmse
                float oldRmse = rmse( testId, pairs[testId] );
                float thisRmse = rmse( testId, trainId );
                if( thisRmse < oldRmse )
                    pairs[testId] = trainId;
            }
            else
            {
                pairs[testId] = trainId;
            }
        }
        if( pairs.size() <= 1 )
            continue;

        //compute a score for this set, keep track of the best score
        float geoScore = 0;
        std::map<int,int>::iterator it = pairs.begin();
        for( ; it!=pairs.end() ; ++it )
        {
            RowVector3f rel = testP.row(it->first) - testP.row(testSeed);
            float thisRmse = rmse( it->first, it->second );
            geoScore += rel.norm() - thisRmse * rmseWeight;
        }
        if(geoScore > bestScore)
        {
            bestScore = geoScore;
            bestSeed = seed;
            best_pairs_ptr = pairs_ptr;
        }
    }
    //return the best set of pairs
    bestPairs = *best_pairs_ptr.get();
}




boost::shared_ptr< CorrGraph > BuildCorrGraph( pairMatrix& matches,
                     Mat3<float>::type& testP, Mat3<float>::type& testVect,
                     Mat3<float>::type& trainP, Mat3<float>::type& trainVect,
                     float distThresh, float angleThresh, float minDist )
{
    boost::shared_ptr< CorrGraph > g_ptr( new CorrGraph(matches.rows()) );
    CorrGraph& g( *g_ptr.get() );

    for(int i = 0; i < matches.rows(); ++i)
    {
        int testSeed = matches(i,0);
        int trainSeed = matches(i,1);

        for(int j = 0; j < matches.rows(); ++j)
        {
            int testId = matches(j,0);
            int trainId = matches(j,1);

            //seed pair match is unique
            if( testId==testSeed || trainId==trainSeed )
                continue;

            RowVector3f testRel = testP.row(testId) - testP.row(testSeed);
            RowVector3f trainRel = trainP.row(trainId) - trainP.row(trainSeed);
            double testDist = testRel.norm();
            double trainDist = trainRel.norm();
            if( fabs(testDist - trainDist) > distThresh )
                continue;

            if( testDist < minDist || trainDist < minDist )
                continue;

            //angles
            float testVectorDot = testVect.row(testId).dot( testVect.row(testSeed) );
            float testVectorAngle = acos(testVectorDot);
            float trainVectorDot = trainVect.row(trainId).dot( trainVect.row(trainSeed) );
            float trainVectorAngle = acos(trainVectorDot);
            if( fabs(testVectorAngle - trainVectorAngle) > angleThresh )
                continue;

            add_edge( i, j, g);
        }
    }
    return g_ptr;
}



boost::shared_ptr< CorrGraph > BuildCorrGraphZAligned( pairMatrix& matches,
                     Mat3<float>::type& testP, Mat3<float>::type& testVect,
                     Mat3<float>::type& trainP, Mat3<float>::type& trainVect,
                     float distThresh, float angleThresh, float minDist )
{
    boost::shared_ptr< CorrGraph > g_ptr( new CorrGraph(matches.rows()) );
    CorrGraph& g( *g_ptr.get() );

    for(int i = 0; i < matches.rows(); ++i)
    {
        int testSeed = matches(i,0);
        int trainSeed = matches(i,1);

        for(int j = 0; j < matches.rows(); ++j)
        {
            int testId = matches(j,0);
            int trainId = matches(j,1);

            //seed pair match is unique
            if( testId==testSeed || trainId==trainSeed )
                continue;

            RowVector3f testRel = testP.row(testId) - testP.row(testSeed);
            RowVector3f trainRel = trainP.row(trainId) - trainP.row(trainSeed);
            //presume no pitch/roll, only yaw: constraint the (signed) z-distance and (absolute) x/y-distance
            double test_xy_dist = sqrt( testRel.coeff(0) * testRel.coeff(0) +
                    testRel.coeff(1) * testRel.coeff(1) );
            double train_xy_dist = sqrt( trainRel.coeff(0) * trainRel.coeff(0) +
                    trainRel.coeff(1) * trainRel.coeff(1) );
            if( fabs( testRel.coeff(2) - trainRel.coeff(2) ) > distThresh ||
                    fabs(test_xy_dist - train_xy_dist) > distThresh )
                continue;

            //minimum separation
            double testDist = testRel.norm();
            double trainDist = trainRel.norm();
            if( testDist < minDist || trainDist < minDist )
                continue;

            //angles
            float testVectorDot = testVect.row(testId).dot( testVect.row(testSeed) );
            float testVectorAngle = acos(testVectorDot);
            float trainVectorDot = trainVect.row(trainId).dot( trainVect.row(trainSeed) );
            float trainVectorAngle = acos(trainVectorDot);
            if( fabs(testVectorAngle - trainVectorAngle) > angleThresh )
                continue;

            add_edge( i, j, g);
        }
    }
    return g_ptr;
}




void PairsOneToOne( pairMatrix& matches, Vect<float>::type& fDist, std::vector<int>& matchesUnique )
{
    //record the one-to-one pairs here
    matchesUnique.clear();

    //find the number of unique test and training point ids
    std::set<int> testIds, trainIds;
    for( int i=0 ; i<matches.rows() ; i++ )
    {
        testIds.insert( matches(i,0) );
        trainIds.insert( matches(i,1) );
    }

    //need to go from test/train id (in matches) to new id (which references the adjacency matrix)
    std::map<int,int> testIdToRow, trainIdToCol;
    std::set<int>::iterator it = testIds.begin();
    for( int i=0; it != testIds.end() ; ++it, ++i )
        testIdToRow.insert( std::pair<int,int>(*it, i) );

    it = trainIds.begin();
    for( int i=0; it != trainIds.end() ; ++it, ++i )
        trainIdToCol.insert( std::pair<int,int>(*it, i) );

    //construct adjacency matrix of matches
    //---but we don't actually use it...
    //Matrix< bool, Dynamic, Dynamic > adjacent( testIds.size(), trainIds.size() );
    //adjacent.setZero();
    //for( int i=0 ; i<matches.rows() ; i++ )
    //    adjacent( testIdToRow[matches(i,0)], trainIdToCol[matches(i,1)] ) = true;

    //indicate a row/col is removed from the adjacency matrix with these
    std::vector<bool> rowDone( testIds.size(), false);
    std::vector<bool> colDone( trainIds.size(), false);

    //go through each row of the adjacency matrix, but in order from the lowest fDist value
    std::vector<unsigned int> matchesArgSort(matches.rows());
    argsort( fDist.data(), fDist.data() + fDist.rows(),
             matchesArgSort.begin(), matchesArgSort.end());

    for( int i=0 ; i<matches.rows() ; i++ )
    {
        int matchId = matchesArgSort[i];
        int testId = matches(matchId,0);
        int row = testIdToRow[testId];
        int trainId = matches(matchId,1);
        int col = trainIdToCol[trainId];

        //can only assign a point once
        if( rowDone[row] || colDone[col] ) continue;

        //proceeding in order of fDist, so out of all pairs remaining containing trainId or testId,
        // this pair is the best, by feature distance.
        //output this pair, remove test/train points from adjacency matrix.
        matchesUnique.push_back(matchId);
        rowDone[row] = true;
        colDone[col] = true;
    }
}


