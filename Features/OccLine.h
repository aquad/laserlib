/*! OccLine.h
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
 * \date       05-12-2012
*/

#ifndef OCC_LINE_HEADER_GUARD
#define OCC_LINE_HEADER_GUARD

#include "export.h"
#define EIGEN_DEFAULT_TO_ROW_MAJOR 1
#include <Eigen/Dense>

#include <vector>
#include <map>

#include "DataStore/VelodyneDb.h"
#include "Common/ArrayTypes.h"
#include "PCA.h"

class VeloRangeImage;
class VelodyneDb;

struct LASERLIB_FEATURES_EXPORT OccLineParams
{
    double wNearThresh; //width along line for point selection.
    double angleNoDataThresh; //line is occluded if no data for this angular length.
    //after surface is intercepted, continue this far along the line looking for a better intercept.
    double lineLengthToCheck;
    //the surface is approximated by a gaussian (from PCA results). the value must be higher that this to be valid.
    double pcaInterceptThresh;
};


enum LASERLIB_FEATURES_EXPORT LineStatus {UNKNOWN=0, EMPTY=1, VALUE=2};


//! Defines a line in 3D and 2D (range image coords)
struct LASERLIB_FEATURES_EXPORT LineCoords
{
    LineCoords( const Eigen::Vector3f& _start, const Eigen::Vector3f& _end, VeloRangeImage& image );

    float length;
    Eigen::Vector3f start;
    Eigen::Vector3f end;
    //line parameters: x = n*t + start
    Eigen::Vector3f n; //normalised

    Eigen::Vector2f startBear;
    Eigen::Vector2f endBear;
    //line parameters
    Eigen::Vector2f nBear; //not normalised
    float nBearNorm;
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
};


struct LASERLIB_FEATURES_EXPORT PCAResults
{
    PCAResults(Mat3<float>::type& _evals, MapMat33Xf& _evects, Mat3<double>::type& _meanP, Vect<int>::type& _pidToResult)
        :   evals(_evals), evects(_evects), meanP(_meanP), pidToResult(_pidToResult) {}

    Mat3<float>::type evals;
    MapMat33Xf evects;
    Mat3<double>::type meanP;
    Vect<int>::type pidToResult;
};


class LASERLIB_FEATURES_EXPORT OccLine
{
public:
    OccLine( OccLineParams& params, Mat3<double>::type& P, Vect<unsigned char>::type& id,
            Vect<double>::type& D, Vect<double>::type& w,
            PCAResults& pcaResults, VelodyneDb& db, VeloRangeImage& image );
    virtual ~OccLine();

    //set this if you know the background and don't want it in the feature
    void setObjectMask( std::vector<bool>& mask );

    //main function that does everything
    void compute( const Eigen::Vector3f& start, const Eigen::Vector3f& end,
            float& value, unsigned char& status );

    //helper functions that do all the work
    void getPointsOnLine( float* bounds, std::vector<int>& points );
    void lineTrace( float* bounds, std::vector< std::pair<int,int> >& pixels );
    void lineTraceFat( float* bounds, std::vector< std::pair<int,int> >& pixels );
    void orderPointsOnLine( LineCoords& line, std::vector<int>& pointsOnLine,
                           std::vector<int>& pointIds_ordered, std::vector<float>& lineTs_ordered );
    void traverseLineForIntercept( std::vector<int>& pointIds_ordered, std::vector<float>& lineTs_ordered,
                                  LineCoords& line, int& closestPid, float& value, unsigned char& status );
    void checkLineProximity(int pid, float pointT, LineCoords& line, float& depthDist,
                            bool& isIntercepted, float& interceptT, float& pointLineDist);

    EIGEN_MAKE_ALIGNED_OPERATOR_NEW //needed because this class contains fixed sized eigen arrays.

protected:
    OccLineParams params;
    VelodyneDb db;
    Mat3<double>::type P;
    Vect<unsigned char>::type id;
    Vect<double>::type D;
    Vect<double>::type w;
    PCAResults pcaResults;
    VeloRangeImage& image;
    std::vector<bool> isInObject;

    //orderPointsOnLine working variables:
    std::vector<int> pointIds_unordered;
    std::vector<float> lineTs_unordered;
    std::vector<unsigned int> lineTs_argsort;
    std::vector<int> pointIds;
    std::vector<float> lineTs;

    //checkLineProximity working variables:
    Eigen::Vector3f p1; // associated position along the line (as determined via 2d proximity).
    Eigen::Vector3f eval, evalSqrt, thisMeanP, nLocal, dLocal, nDash, dDash, xDash, x;
    Eigen::Matrix3f globalToLocalR, m, minv, covinv;

    std::vector< std::pair<int,int> > pixelsOnLine;
    std::vector<int> pointsOnLine;
};



#endif //OCC_LINE_HEADER_GUARD
