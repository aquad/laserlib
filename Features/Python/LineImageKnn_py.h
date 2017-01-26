/*! LineImageKnn_py.h
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

#ifndef LINE_IMAGE_KNN_PY
#define LINE_IMAGE_KNN_PY

#include "../export.h"
#include <Python.h>
#include <numpy/arrayobject.h>
#include <boost/shared_ptr.hpp>

#include "Common/ArrayTypes.h"
#include "Features/LineImage.h"
#include "Features/LineImageMatcher.h"
#include "Features/LineImageKnn.h"
#include "KnnClassifier_py.h"


class LASERLIB_FEATURES_EXPORT LineImageKnn_py : public LineImageKnn, public ObjectKnnClassifier_py
{
public:
    LineImageKnn_py( LineImageParams& params, int metricNo,
                    std::vector<ObjLineImages>& trainObjData,
                    std::vector<int>& nTrainPts,
                    float rmseThresh, float knownThresh,
                    float knownWeight, bool showProgress=false );

    ~LineImageKnn_py();

    void SetTestObject_py( PyObject* testObjData_py );

    PyObject* __getinitargs__();

    friend boost::shared_ptr<LineImageKnn_py> LineImageKnn_py_constructor(
        PyObject* params_py, int metricNo, PyObject* trainData_py,
        float rmseThresh, float knownThresh,
        float knownWeight, bool showProgress );

private:
    //for pickling
    PyObject* params_py_;
    int metricNo_;
    bool showProgress_;
    PyObject* trainData_py_;
};

boost::shared_ptr<LineImageKnn_py> LineImageKnn_py_constructor(
        PyObject* params_py, int metricNo, PyObject* trainData_py,
        float rmseThresh=10.0, float knownThresh=0.4,
        float knownWeight=1.0, bool showProgress=true );
LASERLIB_FEATURES_EXTERN template LineImageKnn_py const volatile * LASERLIB_FEATURES_IMPORT boost::get_pointer(LineImageKnn_py const volatile *);



class LASERLIB_FEATURES_EXPORT LineImageKnnAligned_py : public LineImageKnnAligned, public ObjectKnnClassifier_py
{
public:
    LineImageKnnAligned_py( LineImageParams& params, int metricNo,
                            std::vector<ObjLineImagesAligned>& trainObjData,
                            std::vector<int>& nTrainPts,
                            float alignThresh, float rmseThresh, float knownThresh,
                            float knownWeight, bool showProgress=false );

    ~LineImageKnnAligned_py();

    void SetTestObject_py( PyObject* testObjData_py );

    PyObject* __getinitargs__();

    friend boost::shared_ptr<LineImageKnnAligned_py> LineImageKnnAligned_py_constructor(
        PyObject* params_py, int metricNo, PyObject* trainData_py,
        float alignThresh, float rmseThresh, float knownThresh,
        float knownWeight, bool showProgress );

private:
    //for pickling
    PyObject* params_py_;
    int metricNo_;
    bool showProgress_;
    PyObject* trainData_py_;
};


boost::shared_ptr<LineImageKnnAligned_py> LineImageKnnAligned_py_constructor(
        PyObject* params_py, int metricNo, PyObject* trainData_py,
        float alignThresh, float rmseThresh, float knownThresh,
        float knownWeight, bool showProgress );
LASERLIB_FEATURES_EXTERN template LineImageKnnAligned_py const volatile * LASERLIB_FEATURES_IMPORT boost::get_pointer(LineImageKnnAligned_py const volatile *);



class LASERLIB_FEATURES_EXPORT ObjectMatchHistogram_py : public ObjectMatchHistogram
{
public:
    ObjectMatchHistogram_py( LineImageParams& params, int metricNo,
                          std::vector<ObjLineImagesAligned>& trainFData,
                          float alignThresh, float rmseThresh, float knownThresh,
                          float knownWeight, float binWidth, int nBins )
        : ObjectMatchHistogram( params, metricNo, trainFData, alignThresh,
                                rmseThresh, knownThresh, knownWeight,
                                binWidth, nBins )
    {}

    void MatchObject_py( PyObject* testObj_py, PyObject* hist_py );
};



LASERLIB_FEATURES_EXPORT boost::shared_ptr<ObjectMatchHistogram_py> ObjectMatchHistogram_py_constructor(
        PyObject* params_py, int metricNo, PyObject* trainData_py,
        float alignThresh, float rmseThresh, float knownThresh,
        float knownWeight, float binWidth, int nBins );





#endif //LINE_IMAGE_KNN_PY
