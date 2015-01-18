/*! bron-kerbosch.h
 * Graph clique search for finding geometrically consistent feature match
 * pairs. Largely copy-pasted from boost.
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
 * \date       07-11-2011
*/

#ifndef BRON_KERBOSCH
#define BRON_KERBOSCH

#include <vector>
#include <deque>
#include <algorithm>
#include <boost/config.hpp>

#include <boost/graph/graph_concepts.hpp>
#include <boost/graph/lookup_edge.hpp>
#include <boost/graph/adjacency_list.hpp>


/*!
We want to successively call a function that returns different initial cliques of a specified size.
An external function will fully expand the clique.

We wish to save the state of the algorithm at the first call level.
At this level, the clique always remains empty, only cands and nots change.
After call to find_clique(), an initial clique of given size is found and returned.
Externally (in python), we will align the two models using this clique, then
expand it by finding nearby pairs. Note that no back-communication is needed,
as we have simply replaced the higher level recursive calls with a global alignment.
The following call to find_clique() will return another set.
*/
template <typename Graph>
class bron_kerbosch
{
public:
    bron_kerbosch( const Graph& g ) : g_(g)
    {
        boost::tie(i_vert, i_end) = vertices(g_);
    }

    typedef typename boost::graph_traits<Graph>::vertex_descriptor Vertex;
    typedef typename boost::graph_traits<Graph>::vertex_iterator VertexIterator;
    typedef std::vector<Vertex> Container;
    typedef std::deque<Vertex> Clique;

    //! Find a clique of given size
    void find_clique(Clique& clique, std::size_t min);


private:
    /*! When this returns true, a sufficiently sized clique has been found,
    instructing the caller to return likewise.
    */
    bool extend_clique( Clique& clique,
                        Container& cands, Container& nots,
                        std::size_t min );
/*
    inline bool
    is_connected_to_clique(const Graph& g,
                            typename boost::graph_traits<Graph>::vertex_descriptor u,
                            typename boost::graph_traits<Graph>::vertex_descriptor v,
                            typename boost::graph_traits<Graph>::undirected_category);
*/

    inline bool
    is_connected_to_clique(const Graph& g,
                            typename boost::graph_traits<Graph>::vertex_descriptor u,
                            typename boost::graph_traits<Graph>::vertex_descriptor v,
                            typename boost::graph_traits<Graph>::directed_category);

    inline void
    filter_unconnected_vertices(const Graph& g,
                                typename boost::graph_traits<Graph>::vertex_descriptor v,
                                const Container& in,
                                Container& out);

    VertexIterator i_vert, i_end;
    Container cands_, nots_;
    const Graph& g_;
};



template <typename Graph>
void bron_kerbosch<Graph>::find_clique( Clique& clique, std::size_t min )
{
    clique.clear(); // the first clique is an empty vertex set
    cands_.assign(i_vert, i_end);    // start with all vertices as candidates
    std::random_shuffle(cands_.begin(), cands_.end());
    extend_clique( clique, cands_, nots_, min );
}


template <typename Graph>
bool bron_kerbosch<Graph>::extend_clique( Clique& clique,
                    Container& cands, Container& nots, std::size_t min )
{
    // iterate over candidates
    typename Container::iterator i;
    for(i = cands.begin(); i != cands.end(); )
    {
        Vertex candidate = *i;

        // add the candidate to the clique (keeping the iterator!)
        // typename Clique::iterator ci = clique.insert(clique.end(), candidate);
        clique.push_back(candidate);

        // remove it from the candidate set
        i = cands.erase(i);

        // build new candidate and not sets by removing all vertices
        // that are not connected to the current candidate vertex.
        // these actually invert the operation, adding them to the new
        // sets if the vertices are connected. its semantically the same.
        Container new_cands, new_nots;
        filter_unconnected_vertices(g_, candidate, cands, new_cands);
        filter_unconnected_vertices(g_, candidate, nots, new_nots);

        //only want an initial clique of specified size
        if( clique.size() >= min ) { return true; }

        if(new_cands.empty() && new_nots.empty())
        {
            // our current clique is maximal since there's nothing
            // that's connected that we haven't already visited.
            // Return this (specified clique size was not possible)
            return true;
        }
        else
        {
            // recurse to explore the new candidates
            bool found_clique = extend_clique(
                        clique, new_cands, new_nots, min );
            if( found_clique ) { return true; }
        }

        // we're done with this vertex, so we need to move it
        // to the nots, and remove the candidate from the clique.
        nots.push_back(candidate);
        clique.pop_back();
    }
    return false;
}



/*
template <typename Graph>
inline bool
bron_kerbosch<Graph>::is_connected_to_clique(const Graph& g,
                        typename boost::graph_traits<Graph>::vertex_descriptor u,
                        typename boost::graph_traits<Graph>::vertex_descriptor v,
                        typename boost::graph_traits<Graph>::undirected_category)
{
    return lookup_edge(u, v, g).second;
}
*/

template <typename Graph>
inline bool
bron_kerbosch<Graph>::is_connected_to_clique(const Graph& g,
                        typename boost::graph_traits<Graph>::vertex_descriptor u,
                        typename boost::graph_traits<Graph>::vertex_descriptor v,
                        typename boost::graph_traits<Graph>::directed_category)
{
    // Note that this could alternate between using an || to determine
    // full connectivity. I believe that this should produce strongly
    // connected components. Note that using && instead of || will
    // change the results to a fully connected subgraph (i.e., symmetric
    // edges between all vertices s.t., if a->b, then b->a.
    return lookup_edge(u, v, g).second && lookup_edge(v, u, g).second;
}


template <typename Graph>
inline void
bron_kerbosch<Graph>::filter_unconnected_vertices(const Graph& g,
                            typename boost::graph_traits<Graph>::vertex_descriptor v,
                            const Container& in,
                            Container& out)
{
    typename boost::graph_traits<Graph>::directed_category cat;
    typename Container::const_iterator i, end = in.end();
    for(i = in.begin(); i != end; ++i)
    {
        if(is_connected_to_clique(g, v, *i, cat))
        {
            out.push_back(*i);
        }
    }
}


#endif //BRON_KERBOSCH
