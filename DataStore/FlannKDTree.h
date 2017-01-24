/*! FlannKDTree.h
 *
 * A wrapper for Flann's KDTree for selecting regions of points.
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
 * The graph is basically a segmented range image. Each node is a 3d point,
 * edges connect points 'left' and 'right' to neighbouring points in azimuth,
 * and 'up and 'down' to neighbouring points in elevation. Edges are removed
 * loosely based on discontinous depth changes. The graph can then be traversed
 * for region selection tolerant to point density changes.  It is used largely
 * for surface normal generation.
 *
 * \author     Alastair Quadros
 * \date       08-06-2011
*/

#ifndef FLANN_KD_TREE
#define FLANN_KD_TREE

#include "export.h"
#include <vector>
#include <string>
#include "Selector.h"
#include "Common/ArrayTypes.h"
#include <Eigen/Core>
#include <flann/flann.hpp>


template <typename T=double>
class FlannKDTree : public virtual Selector
{
public:
    typedef typename Mat3<T>::type PointType;
    FlannKDTree( PointType& P, T rad );
    FlannKDTree( PointType& P, T rad, std::string& filename );
    std::vector<int>& SelectRegion(unsigned int centre);
    std::vector<int>& Select3D( Eigen::Matrix<T,3,1>& centreP );
    std::vector<int>& SelectNumber(unsigned int num);
    std::vector<int>& SelectNumber3D(Eigen::Matrix<T,3,1>& centreP);

    void setRadius(double radius)
        { rad_ = radius; }

    void setKnn(int knn)
        { knn_ = knn; }

    boost::shared_ptr<Selector> clone()
        { return boost::shared_ptr<Selector>( new FlannKDTree<T>(*this) ); }

private:
    PointType P_;
    T rad_;
    flann::Index< flann::L2<T> > ind_;
    flann::SearchParams searchParams_;

    std::vector< std::vector<int> > indices_;
    std::vector< std::vector<T> > dists_;
    int knn_; //!< SelectNumber selects k neighbours only
};

template class FlannKDTree<double>;
template class FlannKDTree<float>;


template <typename T>
FlannKDTree<T>::FlannKDTree( PointType& P, T rad )
    :   P_(P),
        rad_(rad),
        ind_( flann::Matrix<T>(P_.data(), P_.rows(), P_.cols()),
              flann::KDTreeSingleIndexParams(10,true) ),
        searchParams_( flann::FLANN_CHECKS_UNLIMITED, 0, true ),
        knn_(1)
{
    ind_.buildIndex();
}


template <typename T>
FlannKDTree<T>::FlannKDTree( PointType& P, T rad, std::string& filename )
    :   P_(P),
        rad_(rad),
        ind_( flann::Matrix<T>(P_.data(), P_.rows(), P_.cols()),
              flann::SavedIndexParams(filename) ),
        searchParams_( flann::FLANN_CHECKS_UNLIMITED, 0, true ),
        knn_(1)
{}


template <typename T>
std::vector<int>& FlannKDTree<T>::SelectRegion(unsigned int centre)
{
    flann::Matrix<T> query(P_.row(centre).data(), 1, 3);
    int val = ind_.radiusSearch(query, indices_, dists_, rad_*rad_, searchParams_ );
    return indices_[0];
}


template <typename T>
std::vector<int>& FlannKDTree<T>::Select3D( Eigen::Matrix<T,3,1>& centreP )
{
    flann::Matrix<T> query(centreP.data(), 1, 3);
    int val = ind_.radiusSearch(query, indices_, dists_, rad_*rad_, searchParams_ );
    return indices_[0];
}


template <typename T>
std::vector<int>& FlannKDTree<T>::SelectNumber(unsigned int centre)
{
    flann::Matrix<T> query(P_.row(centre).data(), 1, 3);
    int val = ind_.knnSearch(query, indices_, dists_, knn_, searchParams_);
    return indices_[0];
}


template <typename T>
std::vector<int>& FlannKDTree<T>::SelectNumber3D(Eigen::Matrix<T,3,1>& centreP)
{
    flann::Matrix<T> query(centreP.data(), 1, 3);
    int val = ind_.knnSearch(query, indices_, dists_, knn_, searchParams_);
    return indices_[0];
}



#endif //FLANN_KD_TREE
