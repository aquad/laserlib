/*! GraphSurfNorm.h
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
 * \date       19-05-2011
*/

#include <cmath>
#include "GraphSurfNorm.h"
#include "DataStore/GraphTraverser.h"
#include <Eigen/Geometry>
#include <boost/math/constants/constants.hpp>
#include <boost/math/distributions/normal.hpp>
const double pi = boost::math::constants::pi<double>();

using namespace Eigen;

//do neighbour selection externally
void CalcSurfNorms( Graph& neighs, Mat3<double>::type& P, Mat3<double>::type& sn, Vect<bool>::type& valid )
{
    valid.setOnes();
    //calc vectors from centre to neighbours
    Matrix<double,4,3> vects; //index: direction, then 3d component
    Matrix<double,4,3> crosses;
    for( int i=0 ; i<neighs.rows() ; i++ )
    {
        vects.setZero();
        for( int j=0 ; j<4 ; j++ )
        {
            int n = neighs(i,j);
            if( n == -1 ) { continue; }
            vects.row(j) = P.row(n) - P.row(i);
            vects.row(j).normalize();
        }
        //cross product (0's result in 0's)
        crosses.row(0) = vects.row(RIGHT).cross( vects.row(UP) );
        crosses.row(1) = vects.row(RIGHT).cross( vects.row(DOWN) );
        crosses.row(2) = vects.row(LEFT).cross( vects.row(UP) );
        crosses.row(3) = vects.row(LEFT).cross( vects.row(DOWN) );

        //normalise, ensure all vectors are pointing the right way
        double normP = P.row(i).norm();
        for( int j=0 ; j<4 ; j++ )
        {
            double norm = crosses.row(j).norm();
            if( norm == 0 )
                continue;
            else
                crosses.row(j) /= norm;
            double dotP = crosses.row(j).dot(-P.row(i));
            double angle = std::acos( dotP / normP );
            if( angle > pi/2 )
                crosses.row(j) *= -1;
        }

        //average
        Vector3d surfNorm = crosses.colwise().sum();
        double norm = surfNorm.norm();
        if(abs(norm) < 1e-9)
            valid(i) = false;
        else
            sn.row(i) = surfNorm / norm;
    }
}


//Gaussian blurs surface normals.
//note: can end up with more valid surf norms.
void BlurSurfNorms( Graph& neighs, Mat3<double>::type& P, Mat3<double>::type& sn,
                   Vect<bool>::type& valid, Mat3<double>::type& sn_blurred, double sd )
{
    boost::math::normal_distribution<double> normDistr(0,sd);
    for( int i=0 ; i<P.rows() ; i++ )
    {
        double sumGaussDist = 0;
        Vector3d thisSn = Vector3d::Zero();
        if( valid(i) )
        {
            double gaussDist = boost::math::pdf( normDistr, 0 );
            thisSn += sn.row(i) * gaussDist;
            sumGaussDist += gaussDist;
        }
        for( int j=0 ; j<4 ; j++ )
        {
            int n = neighs(i,j);
            if( n==-1 )
                continue;
            if( valid(n) )
            {
                Vector3d rel = P.row(n) - P.row(i);
                double gaussDist = boost::math::pdf( normDistr, rel.norm() );
                thisSn += sn.row(n) * gaussDist;
                sumGaussDist += gaussDist;
            }
        }
        //renormalise
        if( sumGaussDist > 0 )
        {
            thisSn /= thisSn.norm();
            sn_blurred.row(i) = thisSn;
        }
    }
}

