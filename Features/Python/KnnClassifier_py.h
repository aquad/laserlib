/*! KnnClassifier_py.h
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
 * \date       15-08-2012
*/

#ifndef KNN_CLASSIFIER_PY
#define KNN_CLASSIFIER_PY

#include <Python.h>
#include "Common/ArrayTypes.h"
#include "Features/KnnClassifier.h"



/*! Handles python interface for object-wise knn classifiers.
 * Usage example: LineImageKnnClassifier_py inherits LineImageKnnClassifier for
 * the actual classifying code, and also inherits ObjectKnnClassifier_py to
 * provide python functions for the code.
 */
class ObjectKnnClassifier_py : virtual public ObjectKnnClassifierInterface
{
public:
    ObjectKnnClassifier_py() {}

    virtual ~ObjectKnnClassifier_py() {}

    /* convert python to required object data type, call ObjectKnnClassifier's
     * inheritor's 'SetTestObject()' method
     */
    virtual void SetTestObject_py( PyObject* testObjData_py ) = 0;

    /*! Classify the object set by SetTestObject_py
    */
    PyObject* Classify_py( int k );

    /*! Find the k nearest neighbours of each point on a test object.
    Returns a tuple (match_objects, match_points, match_dists).
    */
    PyObject* ClassifyObj( PyObject* testObjData_py, int k );

    /*! Find the k nearest neighbours of each point on each test object in the dataset.
    Adds results as attributes to each object PointData.
    */
    void ClassifySet( PyObject* testObjDataset_py, int k );

    //! Return the computation time of the last object (microseconds)
    long GetComputeTime();
};


//Encapsulates Knn results
struct MatchData
{
    MatchData( MapMatXi& object, MapMatXi& point, MapMatXf& dist );
    MatchData( PyObject* object_py, PyObject* point_py, PyObject* dist_py );
    MapMatXi object_;
    MapMatXi point_;
    MapMatXf dist_;
};



#endif //KNN_CLASSIFIER_PY
