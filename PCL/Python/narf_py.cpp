/*! narf_py.cpp
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

#include <Python.h>
#define PY_ARRAY_UNIQUE_SYMBOL _LaserPy_ARRAY_API
#define NO_IMPORT_ARRAY
#include "LaserPy/numpy_to_eigen.h"

#include <vector>
#include <iostream>
#include <boost/python.hpp>

#define EIGEN_DEFAULT_TO_ROW_MAJOR 1

#include "pcl/features/range_image_border_extractor.h"
#include "pcl/keypoints/narf_keypoint.h"
#include "pcl/features/narf_descriptor.h"

#include "PCL/narf_interface.h"
#include "narf_py.h"



void export_narf()
{
    using namespace boost::python;
    class_<pcl_range_image_py, boost::shared_ptr<pcl_range_image_py> >("PCL_RangeImage", no_init)
            .def("getRanges", &pcl_range_image_py::getRanges)
            .def("getP", &pcl_range_image_py::getP)
            .def_readonly("width", &pcl_range_image_py::width)
            .def_readonly("height", &pcl_range_image_py::height);


    class_<pcl_narf_feature_py, boost::shared_ptr<pcl_narf_feature_py> >("PCL_NARF", no_init)
            .def("getP", &pcl_narf_feature_py::getP)
            .def("getOrient", &pcl_narf_feature_py::getOrient)
            .def("getDesc", &pcl_narf_feature_py::getDesc);


    def("make_pcl_rangeimage", &make_pcl_rangeimage_py,
        "make_pcl_rangeimage(P, angular_resolution, setUnseenToMaxRange)\n\n"
        "Parameters\n"
        "----------\n"
        "P : ndarray float32 (n,3)\n"
        "angular_resolution : float\n"
        "setUnseenToMaxRange : bool\n\n"
        "Returns\n"
        "-------\n"
        "image : :class:`PCL_RangeImage`\n");

    def("make_narf_keypoints", &make_narf_keypoints_py);
    def("make_narf_features", &make_narf_features_py);
    def("points_to_range_image_index", &points_to_range_image_index_py);
    def("range_image_index_to_points", &range_image_index_to_points_py);
    def("make_narf_interest_image", &make_narf_interest_image_py);


    class_<NarfKnnAligned_py, boost::shared_ptr<NarfKnnAligned_py> >("NarfKnnAligned", no_init)
            .def( "__init__", boost::python::make_constructor( &NarfKnnAligned_py_constructor ) )
            .def("Match", &NarfKnnAligned_py::Match);
}


using namespace pcl;
using namespace Eigen;


pcl_range_image_py::pcl_range_image_py( boost::shared_ptr<RangeImage> _image )
        : image(_image),
          height(_image->height),
          width(_image->width)
{
    npy_intp dims[2] = {static_cast<npy_intp>(image->points.size()), 3};

    ranges_py = (PyArrayObject*) PyArray_SimpleNew(1, dims, NPY_FLOAT);
    P_py = (PyArrayObject*) PyArray_SimpleNew(2, dims, NPY_FLOAT);
    float* ranges_p = (float*)PyArray_DATA(ranges_py);
    for( int i=0 ; i<dims[0] ; i++ )
    {
        ranges_p[i] = image->points[i].range;
        *(float*)PyArray_GETPTR2(P_py,i,0) = image->points[i].x;
        *(float*)PyArray_GETPTR2(P_py,i,1) = image->points[i].y;
        *(float*)PyArray_GETPTR2(P_py,i,2) = image->points[i].z;
    }
}




pcl_narf_feature_py::pcl_narf_feature_py( boost::shared_ptr< PointCloud<Narf36> > _narf )
        : narf(_narf)
{
    //Narf36:
    //float x, y, z, roll, pitch, yaw;
    //float descriptor[36];
    //output 3 arrays- xyz, rpy, descriptor

    int nPoints = narf->points.size();
    npy_intp dims[2] = {nPoints, 3};

    pts_py = (PyArrayObject*) PyArray_SimpleNew(2, dims, NPY_FLOAT);
    orient_py = (PyArrayObject*) PyArray_SimpleNew(2, dims, NPY_FLOAT);
    dims[1] = 36;
    desc_py = (PyArrayObject*) PyArray_SimpleNew(2, dims, NPY_FLOAT);
    for( int i=0 ; i<dims[0] ; i++ )
    {
        *(float*)PyArray_GETPTR2(pts_py,i,0) = narf->points[i].x;
        *(float*)PyArray_GETPTR2(pts_py,i,1) = narf->points[i].y;
        *(float*)PyArray_GETPTR2(pts_py,i,2) = narf->points[i].z;

        *(float*)PyArray_GETPTR2(orient_py,i,0) = narf->points[i].roll;
        *(float*)PyArray_GETPTR2(orient_py,i,1) = narf->points[i].pitch;
        *(float*)PyArray_GETPTR2(orient_py,i,2) = narf->points[i].yaw;

        memcpy( PyArray_GETPTR2(desc_py,i,0), narf->points[i].descriptor, 36*sizeof(float) );
    }
}




inline PyObject* GetAttr(PyObject *ob, const char *attrName)
{
    PyObject* attr = PyObject_GetAttrString(ob, attrName);
    if( attr == NULL )
    {
        std::cerr << "no" << attrName << std::endl;
    }
    return attr;
}


void narf_keypoint_params_from_py( pcl::NarfKeypoint::Parameters& params, PyObject* params_py )
{
    params.support_size = PyFloat_AsDouble( GetAttr(params_py, "support_size") );
    params.max_no_of_interest_points = PyInt_AsLong( GetAttr(params_py, "max_no_of_interest_points") );
    params.min_distance_between_interest_points = PyFloat_AsDouble( GetAttr(params_py, "min_distance_between_interest_points") );
    params.optimal_distance_to_high_surface_change = PyFloat_AsDouble( GetAttr(params_py, "optimal_distance_to_high_surface_change") );
    params.min_interest_value = PyFloat_AsDouble( GetAttr(params_py, "min_interest_value") );
    params.min_surface_change_score = PyFloat_AsDouble( GetAttr(params_py, "min_surface_change_score") );
    params.optimal_range_image_patch_size = PyInt_AsLong( GetAttr(params_py, "optimal_range_image_patch_size") );
    params.distance_for_additional_points = PyFloat_AsDouble( GetAttr(params_py, "distance_for_additional_points") );
    params.add_points_on_straight_edges = PyObject_IsTrue( GetAttr(params_py, "add_points_on_straight_edges") );
    params.do_non_maximum_suppression = PyObject_IsTrue( GetAttr(params_py, "do_non_maximum_suppression") );
    params.no_of_polynomial_approximations_per_point = PyObject_IsTrue( GetAttr(params_py, "no_of_polynomial_approximations_per_point") );
    params.max_no_of_threads = PyInt_AsLong( GetAttr(params_py, "max_no_of_threads") );
}



//--------------------------------------------------

PyObject* make_narf_keypoints_py( boost::shared_ptr<pcl_range_image_py> range_image_py, PyObject* params_py )
{
    pcl::NarfKeypoint::Parameters params;
    narf_keypoint_params_from_py( params, params_py );

    std::vector<int> keys;
    make_narf_keypoints( range_image_py->image, params, keys );

    npy_intp dims[1] = {static_cast<npy_intp>(keys.size())};
    PyArrayObject* keys_py = (PyArrayObject*)PyArray_SimpleNew(1, dims, NPY_INT);
    memcpy( PyArray_DATA(keys_py), &keys[0], keys.size()*sizeof(int));
    return PyArray_Return(keys_py);
}



boost::shared_ptr<pcl_narf_feature_py> make_narf_features_py( boost::shared_ptr<pcl_range_image_py> range_image_py,
                          PyObject* keys_py, float support_size, bool rotation_invariant)
{
    checkNumpyArray( keys_py, "keys", NPY_INT, -1 );
    std::vector<int> keys;
    keys.resize( PyArray_DIM((PyArrayObject*)keys_py,0) );
    memcpy( &keys[0], PyArray_DATA((PyArrayObject*)keys_py), PyArray_DIM((PyArrayObject*)keys_py,0)*sizeof(int) );

    boost::shared_ptr< PointCloud<Narf36> > narf = make_narf_features(
                range_image_py->image, keys, support_size, rotation_invariant);

    boost::shared_ptr<pcl_narf_feature_py> narf_py( new pcl_narf_feature_py(narf) );
    return narf_py;
}



boost::shared_ptr<pcl_range_image_py> make_pcl_rangeimage_py(PyObject* P_py, float angular_resolution, bool setUnseenToMaxRange)
{
    Mat3<float>::type P = numpy_to_eigen<float, Dynamic, 3>( P_py, "P", NPY_FLOAT );
    boost::shared_ptr<pcl::RangeImage> range_image_ptr = make_pcl_rangeimage( P, angular_resolution, setUnseenToMaxRange );
    boost::shared_ptr<pcl_range_image_py> range_image_py( new pcl_range_image_py(range_image_ptr) );
    return range_image_py;
}




PyObject* make_narf_interest_image_py( boost::shared_ptr<pcl_range_image_py> range_image_ptr, PyObject* params_py )
{
    pcl::NarfKeypoint::Parameters params;
    narf_keypoint_params_from_py( params, params_py );

    RangeImage& range_image = *range_image_ptr->image.get();
    RangeImageBorderExtractor range_image_border_extractor;
    NarfKeypoint narf_keypoint_detector (&range_image_border_extractor);
    narf_keypoint_detector.setRangeImage (&range_image);
    narf_keypoint_detector.getParameters () = params;

    float* interest_image = narf_keypoint_detector.getInterestImage();

    npy_intp dims[1] = {static_cast<npy_intp>(range_image.size())};
    PyArrayObject* interest_image_py = (PyArrayObject*)PyArray_SimpleNew(1, dims, NPY_FLOAT);
    memcpy( PyArray_DATA(interest_image_py), interest_image, dims[0]*sizeof(float));
    return PyArray_Return(interest_image_py);
}




PyObject* points_to_range_image_index_py( boost::shared_ptr<pcl_range_image_py> range_image_ptr, PyObject* P_py )
{
    Mat3<float>::type P = numpy_to_eigen<float, Dynamic, 3>( P_py, "P", NPY_FLOAT );
    RangeImage& range_image = *range_image_ptr->image.get();

    npy_intp dims[1] = {static_cast<npy_intp>(P.rows())};
    PyArrayObject* indices_py = (PyArrayObject*)PyArray_SimpleNew(1, dims, NPY_INT);
    int* ind_p = (int*)PyArray_DATA(indices_py);

    //convert to index form
    Eigen::Vector3f current_point;
    float x_real, y_real, range_of_current_point;
    int x, y;
    for( int i=0 ; i<P.rows() ; i++ )
    {
        current_point = P.row(i);
        range_image.getImagePoint(current_point, x_real, y_real, range_of_current_point);
        range_image.real2DToInt2D(x_real, y_real, x, y);
        ind_p[i] = y*range_image.width + x;
    }
    return PyArray_Return(indices_py);
}


PyObject* range_image_index_to_points_py( boost::shared_ptr<pcl_range_image_py> range_image_ptr, PyObject* index_py )
{
    Vect<int>::type index = numpy_to_eigen<int, Dynamic, 1>( index_py, "index", NPY_INT );
    RangeImage& range_image = *range_image_ptr->image.get();

    npy_intp dims[2] = {static_cast<npy_intp>(index.size()), 3};
    PyArrayObject* P_py = (PyArrayObject*)PyArray_SimpleNew(2, dims, NPY_FLOAT);
    Mat3<float>::type P( (float*)PyArray_DATA(P_py), dims[0], 3 );

    //convert to index form
    PointWithRange current_point;
    for( int i=0 ; i<index.size() ; i++ )
    {
        current_point = range_image.getPoint( index(i) );
        P(i,0) = current_point.x;
        P(i,1) = current_point.y;
        P(i,2) = current_point.z;
    }
    return PyArray_Return(P_py);
}



boost::shared_ptr<NarfKnnAligned_py> NarfKnnAligned_py_constructor(
        PyObject* test_py, PyObject* test_align_py, PyObject* train_py, PyObject* train_align_py,
        float alignThresh, bool showProgress )
{
    MapMatXf test = numpy_to_eigen<float, Dynamic, Dynamic>(
                test_py, "test", NPY_FLOAT );
    Vect<float>::type test_align = numpy_to_eigen<float, Dynamic, 1>(
                test_align_py, "test_align", NPY_FLOAT, test.rows() );

    MapMatXf train = numpy_to_eigen<float, Dynamic, Dynamic>(
                train_py, "train", NPY_FLOAT, -1, test.cols() );
    Vect<float>::type train_align = numpy_to_eigen<float, Dynamic, 1>(
                train_align_py, "train_align", NPY_FLOAT, train.rows() );

    return boost::shared_ptr<NarfKnnAligned_py>(
        new NarfKnnAligned_py( test, test_align, train, train_align, alignThresh, showProgress ) );
}



void NarfKnnAligned_py::Match( PyObject* test_keys_py, PyObject* train_keys_py,
            PyObject* matches_py, PyObject* distances_py )
{
    Vect<int>::type test_keys = numpy_to_eigen<int, Dynamic, 1>(
                test_keys_py, "test_keys", NPY_INT );
    Vect<int>::type train_keys = numpy_to_eigen<int, Dynamic, 1>(
                train_keys_py, "train_keys", NPY_INT );

    MapMatXi matches = numpy_to_eigen<int, Dynamic, Dynamic>(
                matches_py, "matches", NPY_INT, test_keys.rows() );
    MapMatXf distances = numpy_to_eigen<float, Dynamic, Dynamic>(
                distances_py, "distances", NPY_FLOAT, test_keys.rows(), matches.cols() );

    ClassifyKeys( test_keys, train_keys, matches, distances );
}


