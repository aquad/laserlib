/*! Selector.h
 *
 * The base class for selecting regions of points.
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
 * \date       16-05-2011
*/

#ifndef SELECTOR_HEADER
#define SELECTOR_HEADER

#include "export.h"
#include <vector>
#include <boost/shared_ptr.hpp>

/*! Base class for selecting regions.

Make your own region selector, then pass it to any algorithm needing region selection using this interface.
Note: inheritance works through python too!
*/
class LASERLIB_DATASTORE_EXPORT Selector
{
public:
    virtual std::vector<int>& SelectRegion( unsigned int centre ) = 0;
    virtual boost::shared_ptr<Selector> clone() = 0;
    virtual ~Selector() {}
};



#endif //SELECTOR_HEADER
