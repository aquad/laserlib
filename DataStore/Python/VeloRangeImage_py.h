/*! VeloRangeImage_py.h
 * Velodyne range image- python interface for DataStore/VeloRangeImage class.
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
 * \date       08-03-2011
*/

#ifndef VELODYNE_RANGE_IMAGE_PY
#define VELODYNE_RANGE_IMAGE_PY

#include <Python.h>
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <numpy/ndarraytypes.h>

#include <boost/shared_ptr.hpp>
#include "DataStore/VeloRangeImage.h"

class VelodyneDb;


class VeloRangeImage_py : public VeloRangeImage
{
public:
    VeloRangeImage_py(int xRes, int maxPointsPerPixel, VelodyneDb& db);
    ~VeloRangeImage_py();

    /*! Add points to the range image.
    \param w- azimuth array, size (n,), dtype uint16, (deg*100)
    \param id- laser id array, size (n,), dtype uint8, [0,63]
    */
    void AddPoints_py(PyObject* w, PyObject* id);

    /*! Add points to the range image.
    \param w- azimuth array, size (n,), dtype float64, rad
    \param id- laser id array, size (n,), dtype uint8, [0,63]
    */
    void AddPointsRad_py(PyObject* w, PyObject* id);

    /*! Add points to the range image.
    \param w- azimuth array, size (n,), dtype float64, rad
    \param el- elevation array, size (n,), dtype float64, rad
    */
    void AddPointsRadEl_py(PyObject* w, PyObject* el);

    /*! Given a set of azimuth/elevation 2D points, find the associated pixel coords of each.
    \param az- azimuth array, size (n,), dtype float64, rad
    \param el- elevation array, size (n,), dtype float64, rad
    \param x- (output) x coords, size (n,), dtype int32
    \param y- (output) y coords, size (n,), dtype int32
    */
    void GetNearestPixels_py(PyObject* az, PyObject* el, PyObject* x, PyObject* y);

    /*! Like GetNearestPixels_py, but gets all the points in the associated pixels.
    \param az- azimuth array, size (n,), dtype float64, rad
    \param el- elevation array, size (n,), dtype float64, rad
    \param result- (output) ids of points in each pixel, size(n,k), where k determines how many points you get per pixel. dtype int32
    \param nPoints- (output) number of points retrieved in each pixel, size (n,), dtype int32
    */
    void GetNearestPixelsPoints_py(PyObject* az, PyObject* el, PyObject* result, PyObject* nPoints);

    /*! Gets all the points in the specified pixels.
    \param x- x coord, size (n,), dtype int32
    \param y- y coord, size (n,), dtype int32
    \param result- (output) ids of points in each pixel, size(n,k), where k determines how many points you get per pixel. dtype int32
    \param nPoints- (output) number of points retrieved in each pixel, size (n,), dtype int32
    */
    void GetPoints_py(PyObject* x, PyObject* y, PyObject* result, PyObject* nPoints);

    /*! Converts arbitrary 3D points into 2D azimuth/elevation coordinates.
    \param p3d- 3D points, size (n,3), dtype float64
    \param p2d- (output) azimuth/elevation coords, size (n,2), dtype float64
    */
    void Point3dToRangeImage_py( PyObject* p3d, PyObject* p2d );

    //! Returns a tuple (max,min) specifying the elevation borders of the range image.
    boost::python::tuple GetElevationBorders_py();

    //! Returns a (n,) int32 array, specifying how many points are in each pixel.
    PyObject* GetPointsPerPixel_py();

    //! Returns a (n,) int32 array of point ids in the range image.
    PyObject* GetAllPoints_py(PyObject* x, PyObject* y);

    //pickling functions
    PyObject* __getinitargs__();
    //PyObject* __getnewargs__();
    PyObject* __getstate__();
    void __setstate__(PyObject* state);

    //exposed to python for pickling...
    //these three are provided in 'getinitargs'
    //int xRes;
    //int maxPointsPerPixel;
    PyObject* db_pyobj;
    //and these two in 'getstate'
    PyArrayObject* image_py;
    PyArrayObject* nPointsInPixel_py;
};


boost::shared_ptr<VeloRangeImage_py> VeloRangeImage_py_constructor(
        int xRes, int maxPointsPerPixel, PyObject* db_pyobj);



//Other functions that use VeloRangeImage

/*!
For a set of arbitrary 2D (azimuth/elevation) points, find the nearest neighbours (in 2D) in the range image.
To do simple occupancy checking of a set of 3D 'model' points:
- Use VeloRangeImage_py.Point3dToRangeImage() to convert arbitrary 3D points to 2D.
- Call this function to get 2D neighbours in the scan.
- Compute the range of each model point (just the 3D norm).
- Compute the difference between model and scan ranges. If a model point is far behind a scan point, it's occluded.
\param image- range image with points added.
\param image2dPoints- azimuth + elevation array of all points in the range image, size (n,2), dtype float64, rad.
\param queryPoints- azimuth + elevation array of points to matched, size (k,2), dtype float64, rad.
Returns a (k,) array of int32, specifying the indices of nearest neighbours in the scan.
(note- no associated 'pure c++' function)
*/
PyObject* NNImageQuery_py( VeloRangeImage& image, PyObject* image2dPoints, PyObject* queryPoints);


#endif //VELODYNE_RANGE_IMAGE_PY
