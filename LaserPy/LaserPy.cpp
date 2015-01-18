/*! LaserPy.cpp
 * The python LaserPy module is defined here. All classes/functions to be
 * accessed from python are included here.
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
 * \date       10-11-2010
*/

#include <Python.h>
#define PY_ARRAY_UNIQUE_SYMBOL _LaserPy_ARRAY_API
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <numpy/arrayobject.h>
#include <boost/python.hpp>
#include <boost/python/scope.hpp>

//header-only functions/classes
#include "Common/ProgressIndicator.h"
#include "Features/Python/Histogram_py.h"

//cmake defines come from here
#include "LaserLibConfig.h"


void export_numpy_eigen_convert();

//Selection
void export_Selector();
void export_GraphTraverser();
void export_VeloImageSelect();
void export_FlannKDTree(); //ok if not defined?
void export_Subsample();

//Range Image
void export_VeloRangeImage();
void export_BearingGraph();

//Features
void export_SpinImage();
void export_SpinImageMatcher();
void export_SpinImageKnn();
void export_PCA();
void export_PCAFrame();
void export_PCAGrid();
void export_OccLine();
void export_LineImage();
void export_LineImageMatcher();
void export_LineImageKnn();
void export_LineDistrib();

//PCL
void export_narf();
void export_fpfh();
void export_RayTrace();

//Misc
void export_ICP();
void export_GeometricConsistency();
void export_ClusterKMeans();
void export_ClusterAP();

//Python-only
void export_Rotations();
void export_StreamBuffer();

//Tests
void export_TestConversion();

using namespace boost::python;
BOOST_PYTHON_MODULE(LaserPy)
{
    boost::python::docstring_options doc_options(
            true, // show the docstrings from here
            false, // don't show Python signatures.
            false); // Don't mention the C++ method signatures in the generated docstrings.

    // Add some constants to the current (module) scope.
    #ifdef LaserLib_USE_FLANN
    scope().attr("_has_flann") = true;
    #else
    scope().attr("_has_flann") = false;
    #endif

    #ifdef LaserLib_USE_OPENMP
    scope().attr("_has_openmp") = true;
    #else
    scope().attr("_has_openmp") = false;
    #endif

    #ifdef LaserLib_USE_PCL
    scope().attr("_has_pcl") = true;
    #else
    scope().attr("_has_pcl") = false;
    #endif

    //versions would be nice...

    export_numpy_eigen_convert();

    //Selection and Search
    export_Selector();
    export_GraphTraverser();
    export_VeloImageSelect();
#ifdef LaserLib_USE_FLANN
    export_FlannKDTree();
#endif
    export_Subsample();

    //Range Image
    export_VeloRangeImage();
    export_BearingGraph();

    //Features
    export_SpinImage();
    export_SpinImageMatcher();
    export_SpinImageKnn();

    export_PCA();

    export_OccLine();
    export_LineImage();
    export_LineImageMatcher();
    export_LineImageKnn();
    export_LineDistrib();

#ifdef LaserLib_USE_FLANN
    export_PCAFrame();
    export_PCAGrid();
#endif

    //PCL
#ifdef LaserLib_USE_PCL
    export_narf();
    export_fpfh();
    export_RayTrace();
#endif

    //Misc
    export_ICP();
    export_GeometricConsistency();
    export_ClusterKMeans();
    export_ClusterAP();

    //Python-only
    export_Rotations();
    export_StreamBuffer();

    //Tests
    export_TestConversion();

    //---------header-only-------
    def("HistIntersection", &hist_intersection_compare_py);
    def("HistIntExt", &hist_intersection_external_py);


    class_<ProgressIndicator, boost::noncopyable>("ProgressIndicator",
                "ProgressIndicator(total, printPeriod)\n\n"
                "Prints an eta for a loop.\n\n"
                "Parameters\n"
                "----------\n"
                "total : int\n"
                "   Number of iterations in loop\n"
                "printPeriod : int\n"
                "   Period in seconds to update the printed eta",
                init<int, int>())

            .def("start", &ProgressIndicator::start, "start()")
            .def("stop", &ProgressIndicator::stop, "stop()")
            .def("reset", &ProgressIndicator::reset, "reset()")

            .def("addIters", &ProgressIndicator::operator +=,
                 "addIters(n)\n\n"
                 "Call after n iterations of the loop have completed\n\n"
                 "Parameters\n"
                 "----------\n"
                 "n : int");


    import_array() //for numpy
}

