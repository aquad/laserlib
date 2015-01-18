/*! narf_py.h
 * Wraps PCL/narf_interface.h
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
 * \date       04-07-2011
*/


#ifndef NARF_PY
#define NARF_PY

#include <Python.h>
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <numpy/ndarraytypes.h>
#include <boost/shared_ptr.hpp>

#define EIGEN_DEFAULT_TO_ROW_MAJOR 1
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/range_image/range_image.h>
#include <pcl/features/narf_descriptor.h>

#include "PCL/narf_interface.h"


//! contains the PCL RangeImage class, and some numpy arrays (now copied from, not mapped to the data).
class pcl_range_image_py
{
public:
    pcl_range_image_py( boost::shared_ptr<pcl::RangeImage> _image );
    PyObject* getP()
    {
        Py_INCREF(P_py);
        return PyArray_Return((PyArrayObject*)P_py);
    }

    PyObject* getRanges()
    {
        Py_INCREF(ranges_py);
        return PyArray_Return((PyArrayObject*)ranges_py);
    }

    boost::shared_ptr<pcl::RangeImage> image;
    PyArrayObject *ranges_py, *P_py;
    int height, width;
};

//! contains the PCL Narf data
class pcl_narf_feature_py
{
public:
    pcl_narf_feature_py( boost::shared_ptr< pcl::PointCloud<pcl::Narf36> > _narf );
    PyObject* getP()
    {
        Py_INCREF(pts_py);
        return PyArray_Return((PyArrayObject*)pts_py);
    }

    PyObject* getOrient()
    {
        Py_INCREF(orient_py);
        return PyArray_Return((PyArrayObject*)orient_py);
    }

    PyObject* getDesc()
    {
        Py_INCREF(desc_py);
        return PyArray_Return((PyArrayObject*)desc_py);
    }

    boost::shared_ptr< pcl::PointCloud<pcl::Narf36> > narf;
    PyArrayObject *pts_py, *orient_py, *desc_py;
};




boost::shared_ptr<pcl_range_image_py> make_pcl_rangeimage_py(PyObject* P_py, float angular_resolution, bool setUnseenToMaxRange);

PyObject* make_narf_keypoints_py( boost::shared_ptr<pcl_range_image_py>, PyObject* params_py );

boost::shared_ptr<pcl_narf_feature_py> make_narf_features_py( boost::shared_ptr<pcl_range_image_py> range_image_py,
    PyObject* keys_py, float support_size, bool rotation_invariant);


PyObject* points_to_range_image_index_py( boost::shared_ptr<pcl_range_image_py> range_image_ptr, PyObject* P_py );

PyObject* range_image_index_to_points_py( boost::shared_ptr<pcl_range_image_py> range_image_ptr, PyObject* index_py );

PyObject* make_narf_interest_image_py( boost::shared_ptr<pcl_range_image_py> range_image_ptr, PyObject* params_py );




class NarfKnnAligned_py : public NarfKnnAligned
{
public:
    NarfKnnAligned_py( MapMatXf& test, MapVecXf& testAlign, MapMatXf& train, MapVecXf& trainAlign, float alignThresh, bool showProgress=false )
        :   NarfKnnAligned( test, testAlign, train, trainAlign, alignThresh, showProgress)
    {}
    void Match( PyObject* test_keys_py, PyObject* train_keys_py,
                PyObject* matches_py, PyObject* distances_py );
};

boost::shared_ptr<NarfKnnAligned_py> NarfKnnAligned_py_constructor(
        PyObject* test_py, PyObject* test_align_py, PyObject* train_py, PyObject* train_align_py,
        float alignThresh, bool showProgress=false );


#endif //NARF_PY
