/*! VelodyneDb.h
 * Simple structure for the velodyne calibration db.xml file.
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
 * \date       08-03-2011
*/

#ifndef VELODYNE_DB_HEADER_GUARD
#define VELODYNE_DB_HEADER_GUARD

#include <string.h>

struct VelodyneDb
{
    VelodyneDb(){}
    VelodyneDb( double* _dc, double* _rc, double* _vc, double* _voffc, double* _hoffc )
    {
        memcpy( dc, _dc, 64*sizeof(double) );
        memcpy( rc, _rc, 64*sizeof(double) );
        memcpy( vc, _vc, 64*sizeof(double) );
        memcpy( voffc, _voffc, 64*sizeof(double) );
        memcpy( hoffc, _hoffc, 64*sizeof(double) );
    }

    VelodyneDb( const VelodyneDb& other )
    {
        memcpy( dc, other.dc, 64*sizeof(double) );
        memcpy( rc, other.rc, 64*sizeof(double) );
        memcpy( vc, other.vc, 64*sizeof(double) );
        memcpy( voffc, other.voffc, 64*sizeof(double) );
        memcpy( hoffc, other.hoffc, 64*sizeof(double) );
    }

    double dc[64]; //!< distance correction, m
    double rc[64]; //!< rotation angle correction, rad
    double vc[64]; //!< vertical angle correction, rad (downwards)
    double voffc[64]; //!< vertical offset correction, m (downwards)
    double hoffc[64]; //!< horizontal offset correction, m
};

#endif //VELODYNE_DB_HEADER_GUARD
