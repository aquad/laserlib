/*! SpinImageKnn_py.h
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

#ifndef SPIN_IMAGE_KNN_PY
#define SPIN_IMAGE_KNN_PY

#include <Python.h>
#include <numpy/arrayobject.h>
#include <boost/shared_ptr.hpp>

#include "Common/ArrayTypes.h"
#include "Features/SpinImage.h"
#include "Features/SpinImageKnn.h"
#include "KnnClassifier_py.h"



void ObjSpinImagesDataSet_from_py( PyObject* dataset_py,
        std::vector<MapMatXf>& dataset );


ObjSpinImagesAligned ObjSpinImagesAligned_from_py( PyObject* data_py );

void ObjSpinImagesAlignedDataSet_from_py( PyObject* dataset_py,
        std::vector<ObjSpinImagesAligned>& dataset );



class SpinImageKnn_py : public SpinImageKnn, public ObjectKnnClassifier_py
{
public:
    SpinImageKnn_py( std::vector<MapMatXf>& trainObjData,
                     std::vector<int>& nTrainPts, SpinMetric metricNo,
                     float lamb = 0, bool showProgress = false );

    ~SpinImageKnn_py();

    void SetTestObject_py( PyObject* testObjData_py );

    PyObject* __getinitargs__();

    friend boost::shared_ptr<SpinImageKnn_py> SpinImageKnn_py_constructor(
        PyObject* trainObjData, SpinMetric metricNo, float lamb, bool showProgress);

private:
    //for pickling
    bool showProgress_;
    PyObject* trainData_py_;
};

boost::shared_ptr<SpinImageKnn_py> SpinImageKnn_py_constructor(
        PyObject* trainObjData, SpinMetric metricNo, float lamb = 0,
        bool showProgress = false);





class SpinImageKnnAligned_py : public SpinImageKnnAligned, public ObjectKnnClassifier_py
{
public:
    SpinImageKnnAligned_py( std::vector<ObjSpinImagesAligned>& trainObjData,
                         std::vector<int>& nTrainPts, SpinMetric metricNo,
                         float alignThresh, float lamb = 0,
                         bool showProgress = false );

    ~SpinImageKnnAligned_py();

    void SetTestObject_py( PyObject* testObjData_py );

    PyObject* __getinitargs__();

    friend boost::shared_ptr<SpinImageKnnAligned_py> SpinImageKnnAligned_py_constructor(
        PyObject* trainObjData, SpinMetric metricNo, float alignThresh,
        float lamb, bool showProgress);

private:
    //for pickling
    bool showProgress_;
    PyObject* trainData_py_;
};


boost::shared_ptr<SpinImageKnnAligned_py> SpinImageKnnAligned_py_constructor(
        PyObject* trainObjData, SpinMetric metricNo, float alignThresh,
        float lamb = 0, bool showProgress = false);


#endif //SPIN_IMAGE_KNN_PY
