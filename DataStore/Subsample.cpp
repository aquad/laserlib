/*! Subsample.cpp
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
 * \author  Alastair Quadros
 * \date    31-05-2011
*/

#include "Subsample.h"


void SubSampleEven( Selector& sel, int nTotal, std::vector<int>& sample )
{
    std::vector< bool > isDone; //referenced by global point id
    isDone.assign(nTotal, false);

    sample.reserve(nTotal);
    for( unsigned int i=0 ; i<nTotal ; i++ )
    {
        if( isDone[i] )
            continue;
        std::vector<int>& neighs = sel.SelectRegion(i);
        sample.push_back(i);
        for( unsigned int j=0 ; j<neighs.size() ; j++ )
        {
            isDone[neighs[j]] = true;
        }
    }
}




void SubSampleKeysEvenly( Selector& sel, int nTotal, Vect<int>::type& keys, std::vector<int>& sample )
{
    std::vector< bool > isDone; //referenced by global point id
    isDone.assign(nTotal, false);

    sample.reserve(nTotal);
    for( unsigned int i=0 ; i<keys.size() ; i++ )
    {
        int id = keys(i);

        if( isDone[id] )
            continue;

        std::vector<int>& neighs = sel.SelectRegion(id);
        //if( neighs.size() == 0 )
        //    continue;

        sample.push_back(id);
        for( unsigned int j=0 ; j<neighs.size() ; j++ )
        {
            isDone[neighs[j]] = true;
        }
    }
}




void LocalMax( Selector& sel, int nTotal, Vect<int>::type& items, Vect<double>::type& val, Vect<double>::type maxBy )
{
    std::vector< bool > isDone; //referenced by global point id
    isDone.assign(nTotal, false);

    for( int i=0 ; i<items.size() ; i++ )
    {
        unsigned int id = items(i);

        if( isDone[id] )
            continue;

        std::vector<int>& neighs = sel.SelectRegion(id);
        if( neighs.size() == 0 )
            continue;

        //find if its max
        bool isMax = true;
        for( unsigned int j=0 ; j<neighs.size() ; j++ )
        {
            if( neighs[j] == id ){ continue; }
            if( val(id) <= val(neighs[j]) )
            {
                isMax = false;
                break;
            }
        }

        if( isMax )
        {
            //mark isDone so we don't do these neighbours.
            //average amount val is above its neighbours.
            double sum=0.0;
            for( unsigned int j=0 ; j<neighs.size() ; j++ )
            {
                if( neighs[j] == id ){ continue; }
                isDone[neighs[j]] = true;
                sum += val(id) - val(neighs[j]);
            }
            maxBy(id) = sum/(neighs.size()-1);
        }

        neighs.clear();
    }
}




/*
Process: select a region, set all to 'done', if points have very different surface normals, set them to 'kept'.
Do not select regions about 'done' points. Finally, find the 'kept' status of the 'items' in question, store
kept ones in 'sample.'

if the dot product of the two surface normals is below 'thresh' (angle between them is large enough), keep them.
*/
void SubsampleBySurfNorm( Selector& sel, int nTotal, Vect<int>::type& items, Mat3<double>::type& sn,
                         double thresh, std::vector<int>& sample )
{
    std::vector<bool> isKept, isDone;
    isKept.assign(nTotal, false);
    isDone.assign(nTotal, false);

    for( int i=0 ; i<items.size() ; i++ )
    {
        unsigned int id = items.coeff(i);
        if( isDone[id] )
            continue;

        std::vector<int>& neighs = sel.SelectRegion(id);
        Eigen::MatrixBase< Mat3<double>::type >::RowXpr snId = sn.row(id);
        isKept[id] = true;
        for( unsigned int j=0 ; j<neighs.size() ; j++ )
        {
            //if surf norm is similar to point 'id', remove.
            int nid = neighs[j];
            isDone[nid] = true;
            double dotP = sn.row(nid).dot( snId );
            if( dotP < thresh )
                isKept[nid] = true;
        }
        neighs.clear();
    }
    //only care about 'items'
    for( int i=0 ; i<items.size() ; i++ )
    {
        unsigned int id = items.coeff(i);
        if( isKept[id] )
            sample.push_back(id);
    }
}


