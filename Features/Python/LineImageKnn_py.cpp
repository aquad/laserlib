/*! LineImageKnnr_py.cpp
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
 * \date       26-05-2012
*/

#include <Python.h>
#define PY_ARRAY_UNIQUE_SYMBOL _LaserPy_ARRAY_API
#define NO_IMPORT_ARRAY
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <numpy/arrayobject.h>

#define EIGEN_DEFAULT_TO_ROW_MAJOR 1

#include <iostream>
#include <boost/python.hpp>

#include "LineImage_py.h"
#include "LaserPy/numpy_to_std.h"
#include "LaserPy/numpy_to_eigen.h"
#include "LineImageMatcher_py.h"
#include "LineImageKnn_py.h"
#include "Common/ProgressIndicator.h"

using namespace Eigen;
using namespace boost::python;



void export_LineImageKnn()
{
    class_<LineImageKnn_py, boost::shared_ptr<LineImageKnn_py> >(
            "LineImageKnn",
            "LineImageKnn(params, metricNo, trainObjData, rmseThresh, knownThresh, knownWeight[, showProgress])\n\n"
            "Classify a each point on a test object using KNN.\n\n"
            "Parameters\n"
            "----------\n"
            "params : :class:`~pyception.algorithms.line_image.lineImageParams` \n"
            "metricNo : int\n"
            "    Distance function to use (2 works well)\n"
            "trainObjData : list\n"
            "    Each element is a python object (eg. :class:`~pyception.point_io.PointData`) "
            "which contains the feature data for a given 3D object. "
            "Each python object must contain these attributes: \n\n"
            "    * :obj:`values`: line image data, (n,nLines) float32\n\n"
            "    * :obj:`status`: line image datat, (n,nLines) uint8\n"
            "rmseThresh : float\n"
            "    RMSE threshold for matching.\n"
            "knownThresh : float\n"
            "    '% known' threshold for matching\n"
            "knownWeight : float\n"
            "    '% known' weighting in final distance measure (rmseThresh + knownWeight * knownThresh)\n"
            "showProgress : bool, optional\n"
            "    Display progress indicator\n",
            no_init)

            .def( "__init__", boost::python::make_constructor( &LineImageKnn_py_constructor ) )

            .def("ClassifyObj", &LineImageKnn_py::ClassifyObj,
                 "ClassifyObj(testObjData, k)\n\n"
                 "Find the k nearest neighbours of each point on a test object.\n\n"
                 "Parameters\n"
                 "----------\n"
                 "testObjData : python object\n"
                 "    The test object feature data (eg. :class:`~pyception.point_io.PointData`)."
                 "It must contain these attributes: \n\n"
                 "    * :obj:`values`: line image data, (n,nLines) float32\n\n"
                 "    * :obj:`status`: line image datat, (n,nLines) uint8\n"
                 "k : int\n"
                 "    number of nearest neighbours to find\n\n"
                 "Returns\n"
                 "-------\n"
                 "match_objects : ndarray (n,k) int\n"
                 "    For each keypoint, the matching objects (referencing trainObjData)\n"
                 "match_points : ndarray (n,k) int\n"
                 "    For each keypoint, the matching keypoint on the corresponding matching object\n"
                 "match_dists : ndarray (n,k) float32\n"
                 "    For each keypoint, the distances to the matching"
                 "keypoints. Note that very large values are assigned to"
                 "'invalid' matches\n")

            .def("ClassifySet", &LineImageKnn_py::ClassifySet,
                 "ClassifySet(testObjDataset, k)\n\n"
                 "Find the k nearest neighbours of each point on each test object in the dataset.\n\n"
                 "Parameters\n"
                 "----------\n"
                 "testObjDataset: list\n"
                 "    Each element is a python object (eg. :class:`~pyception.point_io.PointData`) "
                 "which contains the feature data for a given 3D object. "
                 "Each python object must contain these attributes: \n\n"
                 "    * :obj:`values`: line image data, (n,nLines) float32\n\n"
                 "    * :obj:`status`: line image datat, (n,nLines) uint8\n"
                 "k : int\n"
                 "    number of nearest neighbours to find\n\n"
                 "Returns nothing, instead it adds the results (listed in "
                 ":meth:`ClassifyObj`) as attributes to each object in "
                 "*testObjDataset*. Also adds ``match_time``, the number of "
                 "microseconds the operation took.")

            .def("GetComputeTime", &LineImageKnn_py::GetComputeTime,
                 "GetComputeTime()\n\n"
                 "Returns\n"
                 "-------\n"
                 "time : int\n"
                 "       Computation time of last object classified, microseconds")

            .enable_pickling()
            .def("__getinitargs__", &LineImageKnnAligned_py::__getinitargs__);



    class_<LineImageKnnAligned_py, boost::shared_ptr<LineImageKnnAligned_py> >(
            "LineImageKnnAligned",
            "LineImageKnnAligned(params, metricNo, trainObjData, alignThresh, rmseThresh, knownThresh, knownWeight[, showProgress])\n\n"
            "Classify a each point on a test object using KNN."
            "Presume that objects are oriented the right way up, so the surface normal "
            "(or linear vector for poles etc) will only be rotated about the z axis. "
            "If the difference of the z component of the aligning vector for two points is "
            "above alignThresh, don't match them.\n\n"
            "Parameters\n"
            "----------\n"
            "params : :class:`~pyception.algorithms.line_image.lineImageParams` \n"
            "metricNo : int\n"
            "    Distance function to use (2 works well)\n"
            "trainObjData : list\n"
            "    Each element is a python object (eg. :class:`~pyception.point_io.PointData`) "
            "which contains the feature data for a given 3D object. "
            "Each python object must contain these attributes: \n\n"
            "    * :obj:`alignVectors`: alignment vector (surf norm or linear vector), (n,3) float32\n\n"
            "    * :obj:`alignType`: flat or linear keypoint, (n,) uint8\n\n"
            "    * :obj:`values`: line image data, (n,nLines) float32\n\n"
            "    * :obj:`status`: line image datat, (n,nLines) uint8\n"
            "alignThresh : float\n"
            "    Different in z component of surface normal / linear direction must be within this threshold.\n"
            "rmseThresh : float\n"
            "    RMSE threshold for matching.\n"
            "knownThresh : float\n"
            "    '% known' threshold for matching\n"
            "knownWeight : float\n"
            "    '% known' weighting in final distance measure (rmseThresh + knownWeight * knownThresh)\n"
            "showProgress : bool, optional\n"
            "    Display progress indicator\n",
            no_init)

            .def( "__init__", boost::python::make_constructor( &LineImageKnnAligned_py_constructor ) )

            .def("ClassifyObj", &LineImageKnnAligned_py::ClassifyObj,
                 "ClassifyObj(testObjData, k)\n\n"
                 "Find the k nearest neighbours of each point on a test object.\n\n"
                 "Parameters\n"
                 "----------\n"
                 "testObjData : python object\n"
                 "    The test object feature data (eg. :class:`~pyception.point_io.PointData`)."
                 "It must contain the attributes of *trainObjData* (see constructor).\n"
                 "k : int\n"
                 "    number of nearest neighbours to find\n\n"
                 "Returns\n"
                 "-------\n"
                 "match_objects : ndarray (n,k) int\n"
                 "    For each keypoint, the matching objects (referencing trainObjData)\n"
                 "match_points : ndarray (n,k) int\n"
                 "    For each keypoint, the matching keypoint on the corresponding matching object\n"
                 "match_dists : ndarray (n,k) float32\n"
                 "    For each keypoint, the distances to the matching"
                 "keypoints. Note that very large values are assigned to"
                 "'invalid' matches\n")

            .def("ClassifySet", &LineImageKnnAligned_py::ClassifySet,
                 "ClassifySet(testObjDataset, k)\n\n"
                 "Find the k nearest neighbours of each point on each test object in the dataset.\n\n"
                 "Parameters\n"
                 "----------\n"
                 "testObjDataset: list\n"
                 "    Each element is a python object (eg. :class:`~pyception.point_io.PointData`) "
                 "which contains the feature data for a given 3D object. "
                 "It must contain the attributes of *trainObjData* (see constructor).\n"
                 "k : int\n"
                 "    number of nearest neighbours to find\n\n"
                 "Returns nothing, instead it adds the results (listed in "
                 ":meth:`ClassifyObj`) as attributes to each object in "
                 "*testObjDataset*. Also adds ``match_time``, the number of "
                 "microseconds the operation took.")

            .def("GetComputeTime", &LineImageKnnAligned_py::GetComputeTime,
                 "GetComputeTime()\n\n"
                 "Returns\n"
                 "-------\n"
                 "time : int\n"
                 "       Computation time of last object classified, microseconds")

            .enable_pickling()
            .def("__getinitargs__", &LineImageKnnAligned_py::__getinitargs__);



    class_<ObjectMatchHistogram_py, boost::shared_ptr<ObjectMatchHistogram_py> >("ObjectMatchHistogram", no_init)
            .def( "__init__", boost::python::make_constructor( &ObjectMatchHistogram_py_constructor ) )
            .def("MatchObject", &ObjectMatchHistogram_py::MatchObject_py);
}




LineImageKnn_py::LineImageKnn_py( LineImageParams& params, int metricNo,
                        std::vector<ObjLineImages>& trainObjData,
                        std::vector<int>& nTrainPts,
                        float rmseThresh, float knownThresh,
                        float knownWeight, bool showProgress )
    :   LineImageKnn(params, metricNo, trainObjData, nTrainPts,
            rmseThresh, knownThresh, knownWeight, showProgress)
{}


boost::shared_ptr<LineImageKnn_py> LineImageKnn_py_constructor(
        PyObject* params_py, int metricNo, PyObject* trainData_py,
        float rmseThresh, float knownThresh,
        float knownWeight, bool showProgress )
{
    LineImageParams_py params_py_derived(params_py);
    LineImageParams& params = *dynamic_cast<LineImageParams*>(&params_py_derived);

    std::vector<ObjLineImages> dataset;
    ObjLineImagesDataSet_from_py( trainData_py, dataset );
    std::vector<int> nTrainPts( dataset.size() );
    for( int i=0 ; i<dataset.size() ; i++ )
        nTrainPts[i] = dataset[i].nPoints;

    boost::shared_ptr<LineImageKnn_py> result(
        new LineImageKnn_py( params, metricNo, dataset, nTrainPts,
                             rmseThresh, knownThresh, knownWeight, showProgress ) );
    //init python objects for pickling
    result->params_py_ = params_py;
    Py_INCREF(params_py);
    result->metricNo_ = metricNo;
    result->showProgress_ = showProgress;
    result-> trainData_py_ = trainData_py;
    Py_INCREF(trainData_py);
    return result;
}


LineImageKnn_py::~LineImageKnn_py()
{
    Py_XDECREF(params_py_);
    Py_XDECREF(trainData_py_);
}


void LineImageKnn_py::SetTestObject_py( PyObject* testObjData_py )
{
    ObjLineImages testObjData = ObjLineImages_from_py( testObjData_py );
    SetTestObject( testObjData );
}


PyObject* LineImageKnn_py::__getinitargs__()
{
    PyObject* tup = PyTuple_Pack(7, params_py_, PyInt_FromLong(metricNo_),
            trainData_py_,
            PyFloat_FromDouble(rmseThresh_), PyFloat_FromDouble(knownThresh_),
            PyFloat_FromDouble(knownWeight_), PyBool_FromLong(showProgress_));
    return tup;
}



//------------------------------------------------------------




LineImageKnnAligned_py::LineImageKnnAligned_py( LineImageParams& params, int metricNo,
                        std::vector<ObjLineImagesAligned>& trainObjData,
                        std::vector<int>& nTrainPts,
                        float alignThresh, float rmseThresh, float knownThresh,
                        float knownWeight, bool showProgress )
    :   LineImageKnnAligned(params, metricNo, trainObjData, nTrainPts,
            alignThresh, rmseThresh, knownThresh, knownWeight, showProgress)
{}


boost::shared_ptr<LineImageKnnAligned_py> LineImageKnnAligned_py_constructor(
        PyObject* params_py, int metricNo, PyObject* trainData_py,
        float alignThresh, float rmseThresh, float knownThresh,
        float knownWeight, bool showProgress )
{
    LineImageParams_py params_py_derived(params_py);
    LineImageParams& params = *dynamic_cast<LineImageParams*>(&params_py_derived);

    std::vector<ObjLineImagesAligned> dataset;
    ObjLineImagesAlignedDataSet_from_py( trainData_py, dataset );
    std::vector<int> nTrainPts( dataset.size() );
    for( int i=0 ; i<dataset.size() ; i++ )
        nTrainPts[i] = dataset[i].nPoints;

    boost::shared_ptr<LineImageKnnAligned_py> result(
        new LineImageKnnAligned_py( params, metricNo, dataset, nTrainPts,
                                    alignThresh, rmseThresh, knownThresh, knownWeight, showProgress ) );
    //init python objects for pickling
    result->params_py_ = params_py;
    Py_INCREF(params_py);
    result->metricNo_ = metricNo;
    result->showProgress_ = showProgress;
    result-> trainData_py_ = trainData_py;
    Py_INCREF(trainData_py);
    return result;
}


LineImageKnnAligned_py::~LineImageKnnAligned_py()
{
    Py_XDECREF(params_py_);
    Py_XDECREF(trainData_py_);
}


void LineImageKnnAligned_py::SetTestObject_py( PyObject* testObjData_py )
{
    ObjLineImagesAligned testObjData = ObjLineImagesAligned_from_py( testObjData_py );
    SetTestObject( testObjData );
}



PyObject* LineImageKnnAligned_py::__getinitargs__()
{
    PyObject* tup = PyTuple_Pack(8, params_py_, PyInt_FromLong(metricNo_),
            trainData_py_, PyFloat_FromDouble(alignThresh_),
            PyFloat_FromDouble(rmseThresh_), PyFloat_FromDouble(knownThresh_),
            PyFloat_FromDouble(knownWeight_), PyBool_FromLong(showProgress_));
    return tup;
}


void ObjectMatchHistogram_py::MatchObject_py( PyObject* testObj_py, PyObject* hist_py )
{
    ObjLineImagesAligned testObj = ObjLineImagesAligned_from_py( testObj_py );
    MapMatXi hist = numpy_to_eigen<int, Dynamic, Dynamic>( hist_py, "hist", NPY_INT, trainFData_.size(), nBins_ );
    MatchObject( testObj, hist );
}



boost::shared_ptr<ObjectMatchHistogram_py> ObjectMatchHistogram_py_constructor(
    PyObject* params_py, int metricNo, PyObject* trainData_py,
    float alignThresh, float rmseThresh, float knownThresh,
    float knownWeight, float binWidth, int nBins )
{
    LineImageParams_py params_py_derived(params_py);
    LineImageParams& params = *dynamic_cast<LineImageParams*>(&params_py_derived);

    std::vector<ObjLineImagesAligned> dataset;
    ObjLineImagesAlignedDataSet_from_py( trainData_py, dataset );

    return boost::shared_ptr<ObjectMatchHistogram_py>(
        new ObjectMatchHistogram_py( params, metricNo, dataset,
                                    alignThresh, rmseThresh, knownThresh,
                                    knownWeight, binWidth, nBins ) );
}


