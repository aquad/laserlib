/*! VeloImageSelect_py.h
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
 * \date       23-05-2011
*/

#ifndef VELODYNE_IMAGE_SELECT_PY
#define VELODYNE_IMAGE_SELECT_PY

#include "../export.h"
#include <Python.h>
#include "DataStore/VeloImageSelect.h"
#include "Selector_py.h"

class VeloRangeImage;
class VelodyneDb;


class LASERLIB_DATASTORE_EXPORT ImagePlusSelector_py : public ImagePlusSelector, public Selector_py
{
public:
    ImagePlusSelector_py( VeloRangeImage& image, VelodyneDb& db,
                           double* w, unsigned char* id, double* D, double rad )
        :   ImagePlusSelector( image, db, w, id, D, rad )
    {}
};

LASERLIB_DATASTORE_EXPORT boost::shared_ptr<ImagePlusSelector_py> ImagePlusSelector_py_constructor( VeloRangeImage& image, PyObject* db_pyobj,
            PyObject* w_py, PyObject* id_py, PyObject* D_py, double rad );
LASERLIB_DATASTORE_EXTERN template ImagePlusSelector_py const volatile * LASERLIB_DATASTORE_IMPORT boost::get_pointer(ImagePlusSelector_py const volatile *);



class LASERLIB_DATASTORE_EXPORT ImageSphereSelector_py : public ImageSphereSelector, public Selector_py
{
public:
    ImageSphereSelector_py( VeloRangeImage& image, VelodyneDb& db,
                           double* w, unsigned char* id, double* D, Mat3<double>::type& P, double rad )
        :   ImageSphereSelector( image, db, w, id, D, P, rad )
    {}
};

LASERLIB_DATASTORE_EXPORT boost::shared_ptr<ImageSphereSelector_py> ImageSphereSelector_py_constructor( VeloRangeImage& image, PyObject* db_pyobj,
            PyObject* w_py, PyObject* id_py, PyObject* D_py, PyObject* P_py, double rad );
LASERLIB_DATASTORE_EXTERN template ImageSphereSelector_py const volatile * LASERLIB_DATASTORE_IMPORT boost::get_pointer(ImageSphereSelector_py const volatile *);


LASERLIB_DATASTORE_EXPORT PyObject* RangeImageMatch_py( VeloRangeImage& image, PyObject* t_py, PyObject* P_py,
                              PyObject* srcLid_py, PyObject* srcW_py, PyObject* srcT_py, PyObject* srcP_py );

#endif //VELODYNE_IMAGE_SELECT_PY
