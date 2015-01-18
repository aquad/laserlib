/*! SpinImageKnn_py.cpp
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
 * \author     Alastair Quadros
 * \date       12-01-2013
*/

#include <Python.h>
#define PY_ARRAY_UNIQUE_SYMBOL _LaserPy_ARRAY_API
#define NO_IMPORT_ARRAY
#include "LaserPy/numpy_to_eigen.h"

#include <boost/python.hpp>
#include "SpinImageKnn_py.h"
#include "Common/ArrayTypes.h"

using namespace Eigen;
using namespace boost::python;


void export_SpinImageKnn()
{
    class_<SpinImageKnn_py, boost::shared_ptr<SpinImageKnn_py> >(
        "SpinImageKnn",
        "SpinImageKnn(trainObjData, metricNo, lamb[, showProgress])\n\n"
        "Classify a each point on a test object using KNN.\n\n"
        "Parameters\n"
        "----------\n"
        "trainObjData : list\n"
        "    Each element is a python object (eg. :class:`~pyception.point_io.PointData`) "
        "which contains the attribute: \n\n"
        "    * :obj:`images`: image data, (n,nCells) float32\n"
        "(n,nCells) float32. Each row is a flattened spin image.\n"
        "metricNo : SpinMetric (int)\n"
        "    0 - correlation, 1 - correlation atanh^2, 2 - similarity\n"
        "lamb : float\n"
        "showProgress : bool, optional\n"
        "    Display progress indicator\n",
        no_init)

        .def( "__init__", boost::python::make_constructor( &SpinImageKnn_py_constructor ) )

        .def("ClassifyObj", &SpinImageKnn_py::ClassifyObj,
             "ClassifyObj(testObjData, k)\n\n"
             "Find the k nearest neighbours of each point on a test object.\n\n"
             "Parameters\n"
             "----------\n"
             "testObjData : ndarray\n"
             "    The test object feature data containing the attribute: \n\n"
             "    * :obj:`images`: image data, (n,nCells) float32\n"
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

        .def("ClassifySet", &SpinImageKnn_py::ClassifySet,
             "ClassifySet(testObjDataset, k)\n\n"
             "Find the k nearest neighbours of each point on each test object in the dataset.\n\n"
             "Adds the results (listed in :meth:`ClassifyObj`) as attributes to "
             "each object in *testObjDataset*. Also adds ``match_time``, the "
             "number of microseconds the operation took.\n\n"
             "Parameters\n"
             "----------\n"
             "testObjDataset : list\n"
             "    Each element is a python object containing 'images', as in *trainObjData* (see constructor).\n"
             "k : int\n"
             "    number of nearest neighbours to find")

        .def("GetComputeTime", &SpinImageKnn_py::GetComputeTime,
             "GetComputeTime()\n\n"
             "Returns\n"
             "-------\n"
             "time : int\n"
             "       Computation time of last object classified, microseconds")

        .enable_pickling()
        .def("__getinitargs__", &SpinImageKnn_py::__getinitargs__);




    class_<SpinImageKnnAligned_py, boost::shared_ptr<SpinImageKnnAligned_py> >(
        "SpinImageKnnAligned",
        "SpinImageKnnAligned(trainObjData, metricNo, alignThresh, lamb[, showProgress])\n\n"
        "Classify a each point on a test object using KNN."
        "Presume that objects are oriented the right way up, so the surface normal "
        "(or linear vector for poles etc) will only be rotated about the z axis. "
        "If the difference of the z component of the aligning vector for two points is "
        "above alignThresh, don't match them.\n\n"
        "Parameters\n"
        "----------\n"
        "trainObjData : list\n"
        "    Each element is a python object (eg. :class:`~pyception.point_io.PointData`) "
        "which contains the feature data for a given 3D object. "
        "Each python object must contain these attributes: \n\n"
        "    * :obj:`alignVectors`: alignment vector (surf norm or linear vector), (n,3) float32\n\n"
        "    * :obj:`alignType`: flat or linear keypoint, (n,) uint8\n\n"
        "    * :obj:`images`: image data, (n,nCells) float32\n"
        "metricNo : SpinMetric (int)\n"
        "    0 - correlation, 1 - correlation atanh^2, 2 - similarity\n"
        "alignThresh : float\n"
        "    Different in z component of surface normal / linear direction must be within this threshold.\n"
        "lamb : float\n"
        "showProgress : bool, optional\n"
        "    Display progress indicator\n",
        no_init)

        .def( "__init__", boost::python::make_constructor( &SpinImageKnnAligned_py_constructor ) )

        .def("ClassifyObj", &SpinImageKnnAligned_py::ClassifyObj,
             "ClassifyObj(testObjData, k)\n\n"
             "Find the k nearest neighbours of each point on a test object.\n\n"
             "Parameters\n"
             "----------\n"
             "testObjData : python object\n"
             "    The test object feature data (eg. :class:`~pyception.point_io.PointData`)."
             "It must contain the attributes of *trainObjData* (see :class:`constructor <SpinImageKnnAligned>`).\n"
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

        .def("ClassifySet", &SpinImageKnnAligned_py::ClassifySet,
             "ClassifySet(testObjDataset, k)\n\n"
             "Find the k nearest neighbours of each point on each test object in the dataset.\n\n"
             "Adds the results (listed in :meth:`ClassifyObj`) as attributes to "
             "each object in *testObjDataset*. Also adds ``match_time``, the "
             "number of microseconds the operation took.\n\n"
             "Parameters\n"
             "----------\n"
             "testObjDataset : list\n"
             "    Each element is a python object (eg. :class:`~pyception.point_io.PointData`) "
             "which contains the feature data for a given 3D object. "
             "It must contain the attributes of *trainObjData* (see constructor).\n"
             "k : int\n"
             "    number of nearest neighbours to find")

        .def("GetComputeTime", &SpinImageKnnAligned_py::GetComputeTime,
             "GetComputeTime()\n\n"
             "Returns\n"
             "-------\n"
             "time : int\n"
             "       Computation time of last object classified, microseconds")

        .enable_pickling()
        .def("__getinitargs__", &SpinImageKnnAligned_py::__getinitargs__);


    enum_<SpinMetric>("SpinMetric")
        .value("correlation", CORR)
        .value("corr_atanh", CORR_ATANH)
        .value("similarity", SIMILARITY);
}



SpinImageKnn_py::SpinImageKnn_py( std::vector<MapMatXf>& trainObjData,
                std::vector<int>& nTrainPts, SpinMetric metricNo, float lamb,
                bool showProgress )
    :   SpinImageKnn( trainObjData, nTrainPts, metricNo, lamb, showProgress )
{}


boost::shared_ptr<SpinImageKnn_py> SpinImageKnn_py_constructor(
        PyObject* trainObjData, SpinMetric metricNo, float lamb,
        bool showProgress)
{
    std::vector<MapMatXf> dataset;
    ObjSpinImagesDataSet_from_py( trainObjData, dataset );
    std::vector<int> nTrainPts( dataset.size() );
    for( int i=0 ; i<dataset.size() ; i++ )
        nTrainPts[i] = dataset[i].rows();

    boost::shared_ptr<SpinImageKnn_py> result(
        new SpinImageKnn_py( dataset, nTrainPts, metricNo,
                             lamb, showProgress ) );
    //init python objects for pickling
    result->showProgress_ = showProgress;
    result->trainData_py_ = trainObjData;
    Py_INCREF(trainObjData);
    return result;
}


SpinImageKnn_py::~SpinImageKnn_py()
{
    Py_XDECREF(trainData_py_);
}


void SpinImageKnn_py::SetTestObject_py( PyObject* testObjData_py )
{
    MapMatXf testObjData = get_attribute_numpy_array<float, Dynamic, Dynamic>(
                testObjData_py, "images", NPY_FLOAT, -1, nCells_ );
    SetTestObject( testObjData );
}


PyObject* SpinImageKnn_py::__getinitargs__()
{
    PyObject* tup = PyTuple_Pack(4, trainData_py_, PyInt_FromLong(metric_),
            PyFloat_FromDouble(lamb_), PyBool_FromLong(showProgress_));
    return tup;
}



//--------------------------------------------------------

SpinImageKnnAligned_py::SpinImageKnnAligned_py(
                std::vector<ObjSpinImagesAligned>& trainObjData,
                std::vector<int>& nTrainPts, SpinMetric metricNo,
                float alignThresh, float lamb, bool showProgress )
    :   SpinImageKnnAligned( trainObjData, nTrainPts, metricNo, alignThresh,
            lamb, showProgress )
{}


boost::shared_ptr<SpinImageKnnAligned_py> SpinImageKnnAligned_py_constructor(
        PyObject* trainObjData, SpinMetric metricNo, float alignThresh,
        float lamb, bool showProgress)
{
    std::vector<ObjSpinImagesAligned> dataset;
    ObjSpinImagesAlignedDataSet_from_py( trainObjData, dataset );
    std::vector<int> nTrainPts( dataset.size() );
    for( int i=0 ; i<dataset.size() ; i++ )
        nTrainPts[i] = dataset[i].nPoints;

    boost::shared_ptr<SpinImageKnnAligned_py> result(
        new SpinImageKnnAligned_py( dataset, nTrainPts, metricNo, alignThresh,
                             lamb, showProgress ) );

    //init python objects for pickling
    result->showProgress_ = showProgress;
    result->trainData_py_ = trainObjData;
    Py_INCREF(trainObjData);
    return result;
}


SpinImageKnnAligned_py::~SpinImageKnnAligned_py()
{
    Py_XDECREF(trainData_py_);
}

void SpinImageKnnAligned_py::SetTestObject_py( PyObject* testObjData_py )
{
    ObjSpinImagesAligned testObjData = ObjSpinImagesAligned_from_py( testObjData_py );
    SetTestObject( testObjData );
}


PyObject* SpinImageKnnAligned_py::__getinitargs__()
{
    PyObject* tup = PyTuple_Pack(5, trainData_py_, PyInt_FromLong(metric_),
            PyFloat_FromDouble(alignThresh_), PyFloat_FromDouble(lamb_),
            PyBool_FromLong(showProgress_));
    return tup;
}



//--------------------------------------------------------




void ObjSpinImagesDataSet_from_py( PyObject* dataset_py, std::vector<MapMatXf>& dataset )
{
    if( !PyList_Check(dataset_py) )
    {
        PyErr_SetString(PyExc_ValueError, "dataset must be a list" );
        throw_error_already_set();
    }
    Py_ssize_t length = PyList_Size(dataset_py);
    for( Py_ssize_t i=0 ; i<length ; i++ )
    {
        dataset.push_back( get_attribute_numpy_array<float, Dynamic, Dynamic>(
                               PyList_GetItem(dataset_py, i), "images", NPY_FLOAT) );
    }
}



//--------------------------------------------------------




ObjSpinImagesAligned ObjSpinImagesAligned_from_py( PyObject* data_py )
{
    Mat3<float>::type alignVectors =
            get_attribute_numpy_array<float, Dynamic, 3>(
                data_py, "alignVectors", NPY_FLOAT);
    int nPoints = alignVectors.rows();

    Vect<unsigned char>::type alignType =
            get_attribute_numpy_array<unsigned char, Dynamic, 1>(
                data_py, "alignType", NPY_UBYTE, nPoints);

    MapMatXf images =
            get_attribute_numpy_array<float, Dynamic, Dynamic>(
                data_py, "images", NPY_FLOAT, nPoints);

    return ObjSpinImagesAligned( alignVectors, alignType, images, images.rows() );
}




void ObjSpinImagesAlignedDataSet_from_py( PyObject* dataset_py,
        std::vector<ObjSpinImagesAligned>& dataset )
{
    if( !PyList_Check(dataset_py) )
    {
        PyErr_SetString(PyExc_ValueError, "dataset must be a list" );
        throw_error_already_set();
    }
    Py_ssize_t length = PyList_Size(dataset_py);
    for( Py_ssize_t i=0 ; i<length ; i++ )
    {
        dataset.push_back( ObjSpinImagesAligned_from_py( PyList_GetItem( dataset_py, i ) ) );
    }
}



