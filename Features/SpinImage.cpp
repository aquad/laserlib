/*! SpinImage.cpp
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
 * \date       15-11-2010
*/

#include <omp.h>
#include "SpinImage.h"
#include <boost/math/special_functions/fpclassify.hpp>

using namespace std;
using namespace Eigen;


SpinImage::SpinImage(Mat3<double>::type& _P, Mat3<double>::type& _sn,
                     double _imageLength, int _cellSide, double _supportAngle)
    :   P(_P),
        sn(_sn),
        imageLength(_imageLength),
        cellSide(_cellSide),
        supportProj(cos(_supportAngle))
{}


//template <typename Derived>
void SpinImage::compute( Vector3d& centreP, Vector3d& centreSn,
                         std::vector<int>& neigh, MapMatXf& image)
{
    //MatrixBase<Derived>& image = const_cast< MatrixBase<Derived>& >(image_);
    image.setZero();
    for(int c=0 ; c<neigh.size() ; c++)
    {
        int n = neigh[c];
        Vector3d Pn = P.row(n);
        Vector3d nn = sn.row(n);

        //convert to alpha, beta
        Vector3d rel = Pn - centreP;
        double beta = centreSn.dot(rel);
        double alpha = sqrt( rel.squaredNorm() - beta*beta );

        //range check
        double proj = nn.dot(centreSn);
        if( alpha != alpha || alpha>=imageLength || abs(beta)>=imageLength/2 ||
                proj<supportProj )
            continue;

        //bin coords
        int i = floor( alpha * cellSide/imageLength );
        int j = floor( (beta + imageLength/2) * cellSide/imageLength );

        //bilinear interpolation
        double a = alpha - (i+0.5)*imageLength/cellSide;
        double b = beta + imageLength/2 - (j+0.5)*imageLength/cellSide;
        a *= cellSide/imageLength;
        b *= cellSide/imageLength;
        int oi = floor(a) + i;
        int oj = floor(b) + j;
        a = abs(a);
        b = abs(b);

        bool oiv = oi<cellSide && oi>=0;
        bool ojv = oj<cellSide && oj>=0;

        image(i,j) += (1-a)*(1-b);
        if( oiv )
            image(oi,j) += (1-a)*b;
        if( ojv )
            image(i,oj) += a*(1-b);
        if( oiv && ojv )
            image(oi,oj) += a*b;
    }
    image.transposeInPlace();
}



float SpinDistance( float* P, float* Q, int n, float lamb, SpinMetric metric )
{
    float dist;
    if(metric == CORR)
        dist = -SpinCorrelation(P, Q, n);
    else if( metric == CORR_ATANH )
        dist = -SpinCorrAtanh(P, Q, n);
    else
        dist = -SpinSimilarity(P, Q, n, lamb);

    if( ! boost::math::isfinite(dist) )
        dist = std::numeric_limits<float>::max();

    return dist;
}

