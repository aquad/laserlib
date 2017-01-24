/* ProgressIndicator.h
 *
 * A progress indicator for use in loops. Prints an eta every 5 seconds. OMP
 * for loop friendly.
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
 * \date       02-12-2010
*/

#ifndef PROGRESS_INDICATOR_HEADER_GUARD
#define PROGRESS_INDICATOR_HEADER_GUARD

#include "export.h"
#include <boost/shared_ptr.hpp>

class ProgressIndicatorImpl;


class LASERLIB_COMMON_EXPORT ProgressIndicator
{
public:
    ProgressIndicator(int total, int _printPeriod);
    ~ProgressIndicator();

    //! starts a new thread that does the waiting
    void start();
    void run();
    //! print the total elapsed time, stop callback timer and join threads.
    void stop();

    void print();

    void operator+=(int iters);

    void reset(int totalCount_);

private:
    boost::shared_ptr<ProgressIndicatorImpl> pImpl;
};


#endif //PROGRESS_INDICATOR_HEADER_GUARD
