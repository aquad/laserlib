/*! StreamBuffer.cpp
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
 * \author     Alastair Quadros
 * \date       30-03-2012
*/

#include <Python.h>
#define PY_ARRAY_UNIQUE_SYMBOL _LaserPy_ARRAY_API
#define NO_IMPORT_ARRAY

// From numpy 1.7, NPY_CARRAY is NPY_ARRAY_CARRAY, but not backwards-compatible...
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <numpy/arrayobject.h>

#include <boost/python.hpp>
#include <iostream>
#include "Common/ArrayTypes.h"
#include "StreamBuffer.h"

using namespace boost::python;


void export_StreamBuffer()
{
    class_<StreamScanBuffer, boost::shared_ptr<StreamScanBuffer>, boost::noncopyable >(
                "StreamScanBuffer",
                "StreamScanBuffer(file, dtype, blockByteOffset, maxPoints)\n\n"
                "Buffer a stream of binary data (eg. from velodyne-to-csv).\n\n"
                "Parameters\n"
                "----------\n"
                "file : file\n"
                "   sys.stdin\n"
                "dtype : dtype\n"
                "   Data type of input stream, as specified for reading it into a recarray."
                "See :func:`~pyception.point_io.named_dtype_from_format` for conversion from a comma type string.\n"
                "blockByteOffset : int\n"
                "   Byte offset of the integer 'block' field which delimits scans.",
                init< PyObject*, PyObject*, int, int>())

            .def("Start", &StreamScanBuffer::Start,
                 "Start()\n\n")

            .def("Stop", &StreamScanBuffer::Stop,
                 "Stop()\n\n")

            .def("GetScan", &StreamScanBuffer::GetScan,
                 "GetScan()\n\n"
                 "Wait for currently-reading scan to finish and return it.\n\n"
                 "Returns\n"
                 "-------\n"
                 "data : ndarray\n"
                 "  Scan data recarray");
}



StreamScanBuffer::StreamScanBuffer( PyObject* file_py, PyObject* dtype_py, int blockByteOffset, int maxPoints )
    : file_py_( (PyFileObject*)file_py ),
      file_( PyFile_AsFile(file_py) ),
      blockByteOffset_(blockByteOffset),
      readBuffLen_(0), writeBuffLen_(0),
      maxPoints_(maxPoints),
      swapRequested_(false), stop_(true),
      lastBlock_(0)
{
    if( !PyFile_Check(file_py) || file_ == NULL )
    {
        PyErr_SetString(PyExc_ValueError, "not a file type" );
        throw_error_already_set();
    }
    PyFile_IncUseCount(file_py_);

    PyArray_DescrConverter(dtype_py, &dtype_);
    packetSize_ = dtype_->elsize;
    maxBuffSize_ = packetSize_ * maxPoints_;

    readBuff_ = new char[maxPoints * dtype_->elsize];
    writeBuff_ = new char[maxPoints * dtype_->elsize];
}



StreamScanBuffer::~StreamScanBuffer()
{
    Stop();
    delete[] readBuff_;
    delete[] writeBuff_;
    PyFile_DecUseCount(file_py_);
}



PyObject* StreamScanBuffer::GetScan()
{
    //tell the writer thread to swap buffers when a scan is done, wait for completion.
    Py_BEGIN_ALLOW_THREADS;
    swapRequested_ = true;

    boost::unique_lock<boost::mutex> lock(mut);
    while(swapRequested_ && !stop_)
    {
        cond.wait(lock);
    }
    Py_END_ALLOW_THREADS;

    //need a new python array of the right length
    npy_intp dims[1] = {readBuffLen_ / packetSize_};

#if NPY_API_VERSION <= 6
    PyObject* data = PyArray_NewFromDescr(&PyArray_Type, dtype_, 1, dims, NULL, readBuff_, NPY_CARRAY, NULL);
#else
    PyObject* data = PyArray_NewFromDescr(&PyArray_Type, dtype_, 1, dims, NULL, readBuff_, NPY_ARRAY_CARRAY, NULL);
#endif
    Py_INCREF(data);
    return PyArray_Return((PyArrayObject*)data);
}



void StreamScanBuffer::Start()
{
    stop_ = false;
    writerThread_ = boost::thread(boost::bind(&StreamScanBuffer::WriterFunction, this));
}


void StreamScanBuffer::WriterFunction()
{
    while( !stop_ || !feof(file_) )
    {
        char* data = writeBuff_ + writeBuffLen_;
        if( fread(data, packetSize_, 1, file_) != 1 )
        {
            std::cerr << "read error" << std::endl;
            return;
        }
        writeBuffLen_ += packetSize_;

        //now test for scan end condition
        int block = *(int*)(data + blockByteOffset_);
        if( block != lastBlock_ || writeBuffLen_ + packetSize_ >= maxBuffSize_ )
        {
            //std::cout << "block: " << block << ", writeBuffLen: " << writeBuffLen_ << std::endl;
            //end scan
            lastBlock_ = block;
            if( swapRequested_ )
            {
                SwapBuffers();
                {
                    boost::lock_guard<boost::mutex> lock(mut);
                    swapRequested_=false;
                }
                cond.notify_one();
            }
            else
            {
                writeBuffLen_ = 0; //clear write buffer
            }
        }
    }
}


void StreamScanBuffer::SwapBuffers()
{
    char* tmp = readBuff_;
    readBuff_ = writeBuff_;
    writeBuff_ = tmp;

    readBuffLen_ = writeBuffLen_;
    writeBuffLen_ = 0;
}



void StreamScanBuffer::Stop()
{
    stop_ = true;
    cond.notify_all();
    writerThread_.join();
}

