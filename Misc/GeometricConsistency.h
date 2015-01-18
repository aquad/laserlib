/*! GeometricConsistency.h
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

#ifndef GEOMETRIC_CONSISTENCY
#define GEOMETRIC_CONSISTENCY

#define EIGEN_DEFAULT_TO_ROW_MAJOR 1
#include <Eigen/Dense>
#include "Common/ArrayTypes.h"

#include <boost/graph/adjacency_list.hpp>
#include <boost/shared_ptr.hpp>

#include <vector>
#include <map>

typedef Eigen::Map< Eigen::Matrix<int, Eigen::Dynamic, 2, Eigen::RowMajor> > pairMatrix;
typedef boost::adjacency_list<boost::vecS, boost::vecS, boost::undirectedS> CorrGraph;



boost::shared_ptr< CorrGraph > BuildCorrGraph( pairMatrix& matches,
                     Mat3<float>::type& testP, Mat3<float>::type& testVect,
                     Mat3<float>::type& trainP, Mat3<float>::type& trainVect,
                     float distThresh, float angleThresh, float minDist );


boost::shared_ptr< CorrGraph > BuildCorrGraphZAligned( pairMatrix& matches,
                     Mat3<float>::type& testP, Mat3<float>::type& testVect,
                     Mat3<float>::type& trainP, Mat3<float>::type& trainVect,
                     float distThresh, float angleThresh, float minDist );

/*!
Given two objects (point clouds), and a matrix of similarity values,
find a pair (within a specified set of 'seed' pairs) that has a large set of other geometrically consistant pairs.
This approach is an attempt to bypass building a full correspondence graph by doing a ransac-style search for vertices with lots of edges.

--Inputs--
\param seeds: an array of point ids (referencing testIds and trainIds) indicating seed pairs- these are iterated over.

\param testIds: point ids (referencing testP)
\param trainIds: point ids (referencing trainP), where {testIds[0], trainIds[0]} are a pair of matched points.

\param testP: testing points
\param testVect: testing alignment vectors
\param trainP: training points
\param trainVect: training alignment vectors

\param rmse: a matrix describing the similarity between testing and training points. eg. rmse(testIds[0], trainIds[0]) gives the
\param similarity of the first pair. This matrix is from the line image feature comparison function.

--Outputs--
\param[out] bestScore: a value indicating how good the resulting set of pairs are (higher is better).
\param[out] bestSeed: the point id (referencing testIds, trainIds) of the seed which gave the best set.
\param[out] bestPairs: the best set of pairs found.

--Optional parameters--
\param distThresh: the difference in distance of a pair from the seed pair must be less than this (m).
\param angleThresh: the surface normal / linear direction angle difference threshold.
\param rmseWeight: the consistency score is given by [distance from seed - pair rmse * rmseWeight], higher is better.
*/
void FindConsistentSet( Vect<int>::type& seeds,
                       Vect<int>::type& testIds,
                       Mat3<float>::type& testP, Mat3<float>::type& testVect,
                       Vect<int>::type& trainIds,
                       Mat3<float>::type& trainP, Mat3<float>::type& trainVect,
                       MapMatXf& rmse,
                       float& bestScore, int& bestSeed, std::map<int,int>& bestPairs,
                       float distThresh=0.2, float angleThresh=0.5, float rmseWeight=3.0 );


/*!
Given two objects (pointclouds) with a set of pairwise matches, build a correspondence graph, and find connected components within the graph.

--Inputs--
\param matches: (n,2) array of matching pairs (point ids, int32)
\param testP: testing points
\param testVect: testing alignment vectors
\param trainP: training points
\param trainVect: training alignment vectors
\param distThresh: the difference in distance of a pair from the seed pair must be less than this (m).
\param angleThresh: the surface normal / linear direction angle difference threshold.

--Output--
\param[out] components: a (n,) int32 array indicating the 'id' of the component each pair belongs to (each id is a connected component).
*/
void FindComponents( pairMatrix& matches,
                 Mat3<float>::type& testP, Mat3<float>::type& testVect,
                 Mat3<float>::type& trainP, Mat3<float>::type& trainVect,
                 float distThresh, float angleThresh,
                 Vect<int>::type& components );


/*!
Given two objects (pointclouds) with a set of pairwise matches, build a correspondence graph, and find the maximal clique in the graph.
Warning- this can take a long long time for large graphs (>10,000 edges).

--Inputs--
\param matches: (n,2) array of matching pairs (point ids, int32)
\param testP: testing points
\param testVect: testing alignment vectors
\param trainP: training points
\param trainVect: training alignment vectors
\param distThresh: the difference in distance of a pair from the seed pair must be less than this (m).
\param minDist: pairs must be further away than this (m) to be put in the graph (prevents dense close pairs being added).
\param angleThresh: the surface normal / linear direction angle difference threshold.
\param minCliqueSize: minimum size of the clique to find.
\param maxEdges: if this is not 0, when the graph has more edges than this, it will return before searching for a clique.

--Output--
\param[out] maxClique: a vector of pair ids in the maximal clique.
*/
void FindMaxClique( pairMatrix& matches,
                 Mat3<float>::type& testP, Mat3<float>::type& testVect,
                 Mat3<float>::type& trainP, Mat3<float>::type& trainVect,
                 float distThresh, float minDist, float angleThresh, int minCliqueSize, int maxEdges,
                 std::vector<int>& maxClique );



/*! Given a set of point pair matches, each with an associated feature distance, reduce them
to one-to-one matches of the highest values.
\param matches: (n,2) array of matching pairs (point ids, int32)
\param fDist: (n,) array of feature distances of matches
\param[out] matchesUnique: references matches that are unique (one to one)
*/
void PairsOneToOne( pairMatrix& matches, Vect<float>::type& fDist, std::vector<int>& matchesUnique );


#endif //GEOMETRIC_CONSISTENCY
