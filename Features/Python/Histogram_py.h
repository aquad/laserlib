/*! Histogram_py.h
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
 * \date       01-12-2010
*/

#ifndef HISTOGRAM_PY
#define HISTOGRAM_PY

#include "../export.h"
#include <Python.h>
#include "LaserPy/numpy_to_eigen.h"
#include "Common/ArrayTypes.h"
#include "Features/Histogram.h"

//Given a set of all histograms, compare the histogram at index 'key' with all histograms,
//returning the similarity values in 'dist'
void hist_intersection_compare_py( PyObject* all_py, int key, PyObject* dist_py )
{
    MapMatXf all = numpy_to_eigen<float, Eigen::Dynamic, Eigen::Dynamic>( all_py, "all", NPY_FLOAT );
    MapVecXf dist = numpy_to_eigen<float, Eigen::Dynamic, 1>( dist_py, "dist", NPY_FLOAT, all.rows() );
    Eigen::MatrixBase< MapMatXf >::RowXpr rowK = all.row(key);

    for( int i=0 ; i<all.rows() ; i++ )
    {
        dist(i) = hist_intersection_kernel(rowK, all.row(i));
    }
}


//as above, but between a set of histograms and a single histogram
void hist_intersection_external_py( PyObject* set_py, PyObject* single_py, PyObject* dist_py )
{
    MapMatXf set = numpy_to_eigen<float, Eigen::Dynamic, Eigen::Dynamic>( set_py, "set", NPY_FLOAT );
    int histSize = set.cols();
    MapVecXf single = numpy_to_eigen<float, Eigen::Dynamic, 1>( single_py, "single", NPY_FLOAT, histSize );
    MapVecXf dist = numpy_to_eigen<float, Eigen::Dynamic, 1>( dist_py, "dist", NPY_FLOAT, set.rows() );

    for( int i=0 ; i<set.rows() ; i++ )
    {
        dist(i) = hist_intersection_kernel(single, set.row(i));
    }
}

#endif //HISTOGRAM_PY
