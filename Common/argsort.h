/* argsort.h
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
 * \date       16-06-2011
*/

#ifndef ARG_SORT_HEADER_GUARD
#define ARG_SORT_HEADER_GUARD


template <class RandomAccessIterator>
struct argsorter
{
    argsorter(RandomAccessIterator _first, RandomAccessIterator _last)
    :   first(_first),
        last(_last)
    {}

    bool operator()(unsigned int s1, unsigned int s2)
        { return first[s1] < first[s2]; }

    RandomAccessIterator first, last;
};


/*! Return the indices that would sort the array.
\param valueFirst, valueLast: the values to sort.
\param argFirst, argLast: the integer output array of indices.
*/
template <class ValueIterator, class ArgIterator>
void argsort(ValueIterator valueFirst, ValueIterator valueLast,
                   ArgIterator argFirst, ArgIterator argLast)
{
    unsigned int i=0;
    for( ArgIterator it = argFirst; it != argLast ; ++it )
        { *it = i++; }

    argsorter<ValueIterator> sorter(valueFirst, valueLast);
    std::sort(argFirst, argLast, sorter);
}


//! Argsort, but only up to argMid (uses std::partial_sort)
template <class ValueIterator, class ArgIterator>
void partial_argsort(ValueIterator valueFirst, ValueIterator valueLast,
                   ArgIterator argFirst, ArgIterator argMid, ArgIterator argLast)
{
    unsigned int i=0;
    for( ArgIterator it = argFirst; it != argLast ; ++it )
        { *it = i++; }

    argsorter<ValueIterator> sorter(valueFirst, valueLast);
    std::partial_sort(argFirst, argMid, argLast, sorter);
}


#endif //ARG_SORT_HEADER_GUARD
