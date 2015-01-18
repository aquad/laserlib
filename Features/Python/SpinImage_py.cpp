/*! SpinImage_py.cpp
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
#define NO_IMPORT_ARRAY
#include "LaserPy/numpy_to_eigen.h"
#include <boost/python.hpp>
#include "SpinImage_py.h"
#include "Features/SpinImage.h"
#include "LaserLibConfig.h" //cmake option for flann


void export_SpinImage()
{
    using namespace boost::python;
    class_<SpinImageBatch_py>(
            "SpinImage",
            "SpinImage(P, sn, imageLength, cellSide, supportAngle[, nThreads=1])\n\n"
            "Parameters\n"
            "----------\n"
            "P : (3,n) ndarray, float64\n"
            "   3D points\n"
            "sn : (3,n) ndarray, float64\n"
            "   Surface normals\n"
            "imageLength : double\n"
            "   Image side length (metres)\n"
            "cellSide : int\n"
            "   Number of cells to a side\n"
            "supportAngle : double\n"
            "   Points are only used if their surface normal is within this "
            "angle of the central spin image orientation (radians)\n"
            "nThreads : int (optional)\n"
            "   -1 for auto (if compiled with openmp)",
            init<PyObject*, PyObject*, double, int, double, optional<int> >())

            #ifdef LaserLib_USE_FLANN
            .def("ComputeAll", &SpinImageBatch_py::ComputeAll,
                 "ComputeAll(sel, centreP, centreSn, images)\n\n"
                 "Compute spin images at arbitrary 3D points. Requires LaserLib "
                 "is build with Flann\n\n"
                 "Parameters\n"
                 "----------\n"
                 "sel : :class:`FlannKDTree`\n"
                 "   KDTree built with all the sensed points to use in the calculation\n"
                 "centreP : (n,3) ndarray, float64\n"
                 "   3D points where spin images will be centred & calculated\n"
                 "centreSn : (n,3) ndarray, float64\n"
                 "   Surface normals of each :obj:`centreP`, which defines the "
                 "spin image orientation\n"
                 "images : (n, :obj:`cellSide` * :obj:`cellSide`) ndarray, float32\n"
                 "   (output) Spin images. Each spin image is row-ordered and "
                 "flattened, so each row in the :obj:`images` array is a "
                 "complete spin image")
            #endif

            .def("ComputeAllKeys", &SpinImageBatch_py::ComputeAllKeys,
                 "ComputeAllKeys(sel, keys, images)\n\n"
                 "Compute spin images at sensed points (ie. the points provided "
                 "on construction)\n\n"
                 "Parameters\n"
                 "----------\n"
                 "sel : :class:`FlannKDTree`\n"
                 "   KDTree built with all the sensed points to use in the calculation\n"
                 "keys : (n,) ndarray, int32\n"
                 "   Keypoints (referring to :obj:`P` and :obj:`sn` provided on "
                 "construction) at which to centre & calculate spin images\n"
                 "images : (n, :obj:`cellSide` * :obj:`cellSide`) ndarray, float32\n"
                 "   (output) Spin images. Each spin image is row-ordered and "
                 "flattened, so each row in the :obj:`images` array is a "
                 "complete spin image");

}


using namespace Eigen;

SpinImageBatch_py::SpinImageBatch_py( PyObject* P_py, PyObject* sn_py,
                            double imageLength, int cellSide, double supportAngle, int nThreads )
    :   P_( numpy_to_eigen<double, Dynamic, 3>( P_py, "P", NPY_DOUBLE ) ),
        sn_( numpy_to_eigen<double, Dynamic, 3>( sn_py, "sn", NPY_DOUBLE, P_.rows() ) ),
        imageLength_(imageLength),
        cellSide_(cellSide),
        supportAngle_(supportAngle),
        nThreads_(nThreads)
{}



void SpinImageBatch_py::ComputeAllKeys( Selector& sel, PyObject* keys_py, PyObject* images_py )
{
    Vect<int>::type keys = numpy_to_eigen<int, Dynamic, 1>( keys_py, "keys", NPY_INT );
    MapMatXf images = numpy_to_eigen<float, Dynamic, Dynamic>(
                images_py, "images", NPY_FLOAT, keys.rows(), cellSide_*cellSide_ );

    SpinImage spin( P_, sn_, imageLength_, cellSide_, supportAngle_ );

    #ifdef LaserLib_USE_OPENMP
    if( nThreads_ > 0 )
        omp_set_num_threads(nThreads_);
    #endif

    std::vector< boost::shared_ptr<Selector> > sels;
    int i;
    int threadId = 0; //will work even if omp is disabled
    int nThreadsUsed = 1;
    #pragma omp parallel private(threadId) shared(sels)
    {
        //copy the selector, one for each thread
        #pragma omp master
        {
            #ifdef LaserLib_USE_OPENMP
            nThreadsUsed = omp_get_num_threads();
            #endif
            for( int j=0 ; j<nThreadsUsed ; j++ )
                { sels.push_back( sel.clone() ); }
        }
        #pragma omp barrier //need to wait for master to finish
        #pragma omp for
        for(i=0 ; i<keys.size() ; i++)
        {
            #ifdef LaserLib_USE_OPENMP
            threadId = omp_get_thread_num();
            #endif
            float* blockData = const_cast<float*>( images.block(i, 0, 1, images.cols()).data() );
            MapMatXf imageBlock(blockData, cellSide_, cellSide_);
            std::vector<int>& neigh = sels[threadId]->SelectRegion( keys(i) );
            Vector3d thisCentreP = P_.row(i);
            Vector3d thisCentreSn = sn_.row(i);
            spin.compute( thisCentreP, thisCentreSn, neigh, imageBlock );
        }
    }
}


#ifdef LaserLib_USE_FLANN
void SpinImageBatch_py::ComputeAll( FlannKDTree_py<double>& sel, PyObject* centreP_py,
                                    PyObject* centreSn_py, PyObject* images_py )
{
    Mat3<double>::type centreP = numpy_to_eigen<double, Dynamic, 3>( centreP_py, "centreP", NPY_DOUBLE );
    int nKeys = centreP.rows();
    Mat3<double>::type centreSn = numpy_to_eigen<double, Dynamic, 3>( centreSn_py, "centreSn", NPY_DOUBLE, nKeys );
    MapMatXf images = numpy_to_eigen<float, Dynamic, Dynamic>(
                images_py, "images", NPY_FLOAT, nKeys, cellSide_*cellSide_ );

    SpinImage spin( P_, sn_, imageLength_, cellSide_, supportAngle_ );

    //TODO copying kdtree is stupid. replace with FlannSelector when it's done
    FlannKDTree<double> selLocal( sel );
    int i;
    #pragma omp parallel for firstprivate(selLocal)
    for(i=0 ; i<centreP.rows() ; i++)
    {
        float* blockData = const_cast<float*>( images.block(i, 0, 1, images.cols()).data() );
        MapMatXf imageBlock(blockData, cellSide_, cellSide_);
        Vector3d thisCentreP = centreP.row(i);
        Vector3d thisCentreSn = centreSn.row(i);
        std::vector<int>& neigh = selLocal.Select3D( thisCentreP );
        spin.compute( thisCentreP, thisCentreSn, neigh, imageBlock );
    }
}
#endif



