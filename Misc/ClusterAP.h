/*! ClusterAP.h
 * Affinity Propagation Clustering.
 *
 * Copyright (C) 2013 Alastair Quadros.
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
 * \detail
 * Frey, B.J, Dueck, D. Clustering by passing messages between data points. Science, 2007.
 *
 * \author     Alastair Quadros
 * \date       10-07-2013
*/



#ifndef CLUSTER_AP_HEADER_GUARD
#define CLUSTER_AP_HEADER_GUARD

#define EIGEN_DEFAULT_TO_ROW_MAJOR 1
#include <Eigen/Core>
#include "Common/ArrayTypes.h"
#include <boost/utility.hpp>


/*!
 * \brief Affinity Propagation
 */
class ClusterAP : boost::noncopyable
{
public:
    ClusterAP( MapMatXf& sim, bool showProgress=true, float damping=0.9,
            int minIter=1, int convergeIter=20, int maxIter=200, int nThreads=-1 );

    //! Run clustering until convergence
    int run();
    //! Run clustering for n iterations
    bool runIters( int nIters=1 );
    //! Get assignments of points to exemplars
    MapVecXi getAssignments();
    MapVecXf getAssignmentScores();

    int size()
    {
        return sim_.rows();
    }

protected:
    void updateAvailabilities();
    void updateResponsibilities();
    bool computeAssignments(); //!< Returns whether assignments changed

    MapMatXf sim_; //!< similarity matrix
    bool showProgress_;
    float damping_;
    int minIter_;
    int convergeIter_;
    int maxIter_;

    Eigen::MatrixXf avail_; //!< availabilities
    Eigen::MatrixXf respon_; //!< responsibilities
    Eigen::VectorXi assign_; //!< assignments of points to exemplars
    Eigen::VectorXf assignScores_; //!< assignments of points to exemplars
    bool assignDegenerate_; //!< whether assignments are degenerate (all points are clusters)

    // working variables:
    Eigen::VectorXf sumR_; // sum over columns of respon_, excluding negatives and diagonals
};



#endif //CLUSTER_AP_HEADER_GUARD
