/*! PCAFrame.h
 * Compute a local alignment frame using PCA results
 *
 * Copyright (C) 2012 Alastair Quadros.
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
 * \date       25-01-2012
*/

#ifndef PCA_FRAME_HEADER_GUARD
#define PCA_FRAME_HEADER_GUARD

#include "export.h"
#define EIGEN_DEFAULT_TO_ROW_MAJOR 1
#include <Eigen/Dense>
#include <vector>
#include <boost/shared_ptr.hpp>
#include "PCA.h"


enum FrameType {FRAME_SURFACE=0, FRAME_LINEAR=1};


//! Stores results from ComputePCAFrames.
class PCAFrames
{
public:
    PCAFrames( int n )
        : size(n), P(n,3), R(n,9), frameType(n,1), alignVect(n,3), pcaId(n)
    {}

    PCAFrames()
        : size(0)
    {}

    virtual ~PCAFrames() {}

    void resize(int n)
    {
        size = n;
        P.resize(n,3);
        R.resize(n,9);
        frameType.resize(n,1);
        alignVect.resize(n,3);
        pcaId.resize(n);
    }

    //could add another layer so that memory reallocation is minimal...
    int size;
    Eigen::Matrix<float,Eigen::Dynamic,3> P; //!< the 3d centre point. shape (3,n)
    Eigen::Matrix<float,Eigen::Dynamic,9> R; //!< the 3x3 rotation matrix for each point. shape (9,n)
    Eigen::Matrix<unsigned char,Eigen::Dynamic,1> frameType; //!< aligned to a flat or linear-shaped region
    Eigen::Matrix<float,Eigen::Dynamic,3> alignVect; //!< an aligning vector (surf norm, or linear direction)
    Eigen::VectorXi pcaId; //!< reference to the original PCA result
};



/*! After computing PCA on regions in a point cloud, the eigenvalues/vectors
can be used to compute a frame of orientation to compute features at.
If a region forms a good plane, x,y is parallel (y faces downwards/towards origin), z faces away from origin.
If its a good line, x is parallel, z aligns to origin (facing outwards).
*/
LASERLIB_FEATURES_EXPORT void ComputePCAFrames(
        Mat3<double>::type meanP, Mat3<float>::type evals, MapMat33Xf evects,
        PCAFrames& frames, float surfThresh = 0.3, float linThresh = 0.5,
        float ssRad = 0.01 );

#endif //PCA_FRAME_HEADER_GUARD
