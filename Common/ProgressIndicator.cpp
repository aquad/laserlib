/* ProgressIndicator.cpp
 *
 * Copyright (C) 2013 Alastair Quadros.
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
 * \date       14-01-2013
*/

#include "ProgressIndicator.h"
#include "ProgressIndicatorImpl.h"


ProgressIndicator::ProgressIndicator(int total, int printPeriod)
    : pImpl( new ProgressIndicatorImpl(total, printPeriod) )
{}

ProgressIndicator::~ProgressIndicator() {}

void ProgressIndicator::start()
    { pImpl->start(); }

void ProgressIndicator::run()
    { pImpl->run(); }

void ProgressIndicator::stop()
    { pImpl->stop(); }

void ProgressIndicator::print()
    { pImpl->print(); }

void ProgressIndicator::operator+=(int iters)
    { pImpl->operator+=(iters); }

void ProgressIndicator::reset(int totalCount)
    { pImpl->reset(totalCount); }
