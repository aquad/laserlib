/*! StreamBuffer.h
 * Buffer a stream of binary data (eg. from velodyne-to-csv).
 * 
 * Copyright (C) 2012 Alastair Quadros.
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
 *
 * \detail
 * Comma/Snark/Ark utilise streaming applications, allowing velodyne data to be
 * segmented, tracked etc. In order to insert a custom python application in
 * such a pipeline, python must constantly read from stdin. However, if
 * processing is too slow, stale data is received. The StreamScanBuffer class
 * constantly reads in data in another thread, allowing a python application to
 * retrieve the latest scan at it's leisure. It expects binary-csv data, with
 * an int field which counts the scan number. The data is stored into a
 * structured numpy array, so any data format can be specified.
 *
 * \author     Alastair Quadros
 * \date       30-03-2012
*/

#ifndef STREAM_BUFFER_HEADER_GUARD
#define STREAM_BUFFER_HEADER_GUARD

#include <Python.h>
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <numpy/ndarraytypes.h>
#include <stdio.h>
#include <boost/thread.hpp>

class StreamScanBuffer
{
public:
    // blockByteOffset- the byte offset of the integer 'block' field which delimits scans.
    StreamScanBuffer( PyObject* file_py, PyObject* dtype_py, int blockByteOffset, int maxPoints );
    virtual ~StreamScanBuffer();

    //start reading (start another thread)
    void Start();

    //stop reading
    void Stop();

    //get the buffered scan
    PyObject* GetScan();

private:

    void WriterFunction();
    inline void SwapBuffers();

    PyFileObject* file_py_;
    PyArray_Descr* dtype_;
    FILE* file_;
    int blockByteOffset_;
    char* readBuff_;
    char* writeBuff_;
    int readBuffLen_;
    int writeBuffLen_;
    int packetSize_;
    int maxPoints_;
    int maxBuffSize_;

    bool swapRequested_;
    boost::thread writerThread_;
    boost::condition_variable cond;
    boost::mutex mut;
    bool stop_;

    int lastBlock_;
};


#endif //STREAM_BUFFER_HEADER_GUARD
