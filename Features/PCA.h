/*! PCA.h
 * Compute Principle Component Analysis on regions of points.
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
 * \date       24-11-2010
*/

#ifndef PCA_HEADER_GUARD
#define PCA_HEADER_GUARD

#include "export.h"
//Eigen
#define EIGEN_DEFAULT_TO_ROW_MAJOR 1
#include <Eigen/Dense>

#include <vector>
#include "Common/ArrayTypes.h"
#include "DataStore/Selector.h"


/*
This is used for an array of eigenvectors (ie 3 vectors per point, those 3 vectors
being flattened to a (1,9) array) for each point). It is in row-major order,
but the eigenvector matrix has an eigenvector in each column. As such, eigenvectors
in the final array are 'fragmented'. You need to make a new 3x3 Map referring to a single row,
then take the columns of that.
*/
typedef Eigen::Map< Eigen::Matrix< float, Eigen::Dynamic, 9, Eigen::RowMajor > > MapMat33Xf;

typedef Eigen::Map< Eigen::Matrix< float, 3, 3, Eigen::RowMajor > > MapMat3f;


//! Computes PCA on a set of points in a pointcloud
class LASERLIB_FEATURES_EXPORT PCA
{
public:
    //! \param P - raw pointcloud to compute PCA on
    PCA(Mat3<double>::type& P);
    void compute(std::vector<int>& neigh, Eigen::Map<Eigen::Vector3f>& evals,
                 MapMat3f& evects, Eigen::Map<Eigen::Vector3d>& meanP );

private:
    Mat3<double>::type& P_;
};


/*! Determine the minimum size sphere about each point that PCA can be computed on.
The selection must include a neighbouring horizontal and vertical point
(if they exist- valid to compute PCA on a pole with no horizontal neighbours).
This uses the 'bearing graph' (see Datastore/BearingGraph).
*/
LASERLIB_FEATURES_EXPORT void minRadiusSelection( Graph& graph, Mat3<double>::type& P, Vect<float>::type& rad, Vect<bool>::type& valid );


/*! \brief Compute surface normals with PCA
 *
 * Parameters:
 * \param [in] sel - Selector of points P
 * \param [in] P - Points in *sel*
 * \param [out] sn - Surface normals. Aligned to *P*, unless *ids* are specified, in which case *sn* must be that length.
 * \param [in] ids - (Optional) subset of point ids to compute. Default- all.
 * \param [in] surfThresh - (Optional) filters out bad results (long thin regions with no strong surface normals).
 *      Lower is stricter. Invalid surface normals are set to (0,0,0).
 */
template <typename T>
void surfNormPCA( Selector& sel, const typename Mat3<T>::type& P, Mat3<float>::type& sn,
                  const Vect<int>::type& ids = Vect<int>::type(NULL,1,1), float surfThresh = -0.9 )
{
    using namespace Eigen;
    bool subset = false;
    int nCalcs = P.rows();
    if( ids.data() )
    {
        subset = true;
        nCalcs = ids.size();
    }

    // check sizes
    if( nCalcs != sn.rows() )
    {
        throw "sn array length does not match";
    }

    for( int i=0 ; i<nCalcs ; i++ )
    {
        int pid = i;
        if(subset)
            pid = ids[i];
        std::vector<int>& neigh = sel.SelectRegion(pid);
        //get mean point:
        //Xbar = 1/N sum(Xi)
        Matrix<T,3,1> mean;
        mean.setZero();
        for(int c=0 ; c<neigh.size() ; c++)
        {
            int n = neigh[c];
            mean += P.row(n).transpose();
        }
        mean /= neigh.size();

        //symmetric covariance matrix
        //cov = 1/N sum( (Xi - Xbar) (Xi - Xbar)' )
        Vector3f Xim;
        Matrix3f cov = Matrix3f::Zero();
        for(int c=0 ; c<neigh.size() ; c++)
        {
            int n = neigh[c];
            Xim = (P.row(n).transpose() - mean).template cast<float>();
            cov += Xim * Xim.transpose();
        }
        cov /= neigh.size();

        SelfAdjointEigenSolver<Matrix3f> eigensolver(cov);
        Vector3f evals = eigensolver.eigenvalues();
        Matrix3f evects = eigensolver.eigenvectors();
        sn.row(i) = evects.col(0).transpose();

        // evaluate flatness from relative eigenvalues
        float surfness = (evals[1] - evals[2])/evals.norm();
        if(surfness > surfThresh)
            sn.row(i) = evects.col(0).transpose();
        else
            sn.row(i).setZero();
    }
}


#endif //PCA_HEADER_GUARD
