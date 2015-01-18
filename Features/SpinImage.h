/*! SpinImage.h
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

#ifndef SPIN_IMAGE
#define SPIN_IMAGE

#include <vector>

#define EIGEN_DEFAULT_TO_ROW_MAJOR 1
#include <Eigen/Core>

#include "Common/ArrayTypes.h"


class SpinImage
{
public:
    SpinImage(Mat3<double>::type& P, Mat3<double>::type& sn,
              double imageLength, int cellSide, double supportAngle);

    void compute( Eigen::Vector3d& centreP, Eigen::Vector3d& centreSn,
                  std::vector<int>& neigh, MapMatXf& image );

private:
    Mat3<double>::type P;
    Mat3<double>::type sn;
    double imageLength;
    int cellSide;
    double supportProj;
};



//! linear correlation coefficient
inline float SpinCorrelation( float* P, float* Q, int n )
{
    float PQsum = 0.0;
    float Psum = 0.0;
    float Qsum = 0.0;
    float PsqrSum = 0.0;
    float QsqrSum = 0.0;

    for( int i=0 ; i<n ; i++ )
    {
        float Pi = P[i];
        float Qi = Q[i];
        PQsum += Pi * Qi;
        Psum += Pi;
        Qsum += Qi;
        PsqrSum += Pi*Pi;
        QsqrSum += Qi*Qi;
    }
    float R = ( n * PQsum - Psum*Qsum ) /
               sqrt( ( n * PsqrSum - Psum*Psum ) *
                     ( n * QsqrSum - Qsum*Qsum ) );
    return R;
}


//! linear correlation coefficient, atanh^2 for 'better statistical properties' as in paper.
inline float SpinCorrAtanh( float* P, float* Q, int n )
{
    float R = SpinCorrelation(P,Q,n);
    float atanhR = 0.5*log((1+R)/(1-R));
    return atanhR*atanhR;
}


//! similarity from the paper (does not count empty cells).
inline float SpinSimilarity( float* P, float* Q, int n, float lamb )
{
    float PQsum = 0.0;
    float Psum = 0.0;
    float Qsum = 0.0;
    float PsqrSum = 0.0;
    float QsqrSum = 0.0;
    int N=n;

    for( int i=0 ; i<n ; i++ )
    {
        float Pi = P[i];
        float Qi = Q[i];
        if( Pi==0 || Qi==0 )
        {
            N-=1;
            continue;
        }
        PQsum += Pi * Qi;
        Psum += Pi;
        Qsum += Qi;
        PsqrSum += Pi*Pi;
        QsqrSum += Qi*Qi;
    }
    float R = ( N * PQsum - Psum*Qsum ) /
               sqrt( ( N * PsqrSum - Psum*Psum ) *
                     ( N * QsqrSum - Qsum*Qsum ) );
    float atanhR = 0.5*log((1+R)/(1-R));
    return atanhR*atanhR - lamb*(1.0/(N-3));
}


enum SpinMetric {CORR=0, CORR_ATANH=1, SIMILARITY=2};

float SpinDistance( float* P, float* Q, int n, float lamb, SpinMetric metric );




#endif //SPIN_IMAGE
