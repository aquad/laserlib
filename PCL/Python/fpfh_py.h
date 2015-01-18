/*! narf_py.h
 *
 * Copyright (C) 2010 Alastair Quadros.
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
 * \date       25-11-2010
*/
#ifndef NARF_PY
#define NARF_PY

#include <Python.h>
#include <vector>
#include "DataStore/Selector.h"
#include "PCL/fpfh_interface.h"


void fpfh_py( PyObject* P_py, PyObject* sn_py, Selector& sel,
              PyObject* keys_py, PyObject* noBins_py, PyObject* hist_py );

void fpfh_knn_py( PyObject* test_py, PyObject* train_py, PyObject* matches_py, PyObject* values_py );



class FPFHObjectKnn_py : public FPFHObjectKnn
{
public:
    FPFHObjectKnn_py( std::vector<MapMatXf>& trainObjData, std::vector<int>& nTrainPts,
                      bool showProgress=false, int nThreads=1 );

    /*! Find the k nearest neighbours of each point on each test object in the dataset.
    Adds results as attributes to each object PointData.
    */
    void ClassifySet( PyObject* testDataset_py, int k );

private:
    int nThreads_;
};

//! trainObjData is a list of numpy arrays (fpfh features, float32, shape=(n,nBins) )
boost::shared_ptr<FPFHObjectKnn_py> FPFHObjectKnn_py_constructor(
        PyObject* trainObjData, bool showProgress, int nThreads=1 );



#endif //NARF_PY
