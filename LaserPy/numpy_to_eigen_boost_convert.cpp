/*! numpy_to_eigen_boost_convert.cpp
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
*/

#include <Python.h>
#define PY_ARRAY_UNIQUE_SYMBOL _LaserPy_ARRAY_API
#define NO_IMPORT_ARRAY
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <numpy/arrayobject.h>

#include <boost/python.hpp>

#define EIGEN_DEFAULT_TO_ROW_MAJOR 1
#include <Eigen/Core>


using namespace boost::python;


// specialise for float, double etc
template <typename T>
bool array_is_type( PyArrayObject* obj );



//! Convert a numpy array to an Eigen Map (no memory copying)
template <typename Derived>
struct eigen_map_from_python
{
    typedef typename Derived::Scalar Scalar;
    typedef typename Eigen::Map<Derived> EigenType;

    eigen_map_from_python()
    {
        // for Function( EigenType arr )
        converter::registry::push_back(
            &convertible,
            &construct,
            type_id<EigenType>());

        // for Function( EigenType* arr )
        //     Function( EigenType& arr )
        converter::registry::insert( &convert, type_id<EigenType>() );
    }


    // Determine if arr can be converted to eigen
    static void* convertible(PyObject* arr)
    {
        if( !PyArray_Check(arr) )
        {
            return 0;
        }
        PyArrayObject* arr_npy = (PyArrayObject*)arr;
        if( !array_is_type<Scalar>(arr_npy) )
        {
            return 0;
        }

        //check dimensions
        int nDims = PyArray_NDIM(arr_npy);
        int actual_rows = PyArray_DIM(arr_npy, 0);
        int actual_cols = 1;
        if( nDims == 2 )
            actual_cols = PyArray_DIM(arr_npy, 1);
        if( Derived::RowsAtCompileTime != -1 &&
            Derived::RowsAtCompileTime != actual_rows )
        {
            return 0;
        }
        if( Derived::ColsAtCompileTime != -1 &&
            Derived::ColsAtCompileTime != actual_cols )
        {
            return 0;
        }
        return arr;
    }


    // Convert arr into an eigen matrix
    static void construct(
        PyObject* arr,
        converter::rvalue_from_python_stage1_data* data)
    {
        // Grab pointer to memory into which to construct the new eigen object
        void* storage = (
          (converter::rvalue_from_python_storage<EigenType>*)
          data)->storage.bytes;

        PyArrayObject* arr_npy = (PyArrayObject*)arr;
        Scalar* arr_data = (Scalar*)PyArray_DATA( arr_npy );
        int nDims = PyArray_NDIM(arr_npy);
        int actual_rows = PyArray_DIM(arr_npy, 0);
        int actual_cols = 1;
        if( nDims == 2 )
            actual_cols = PyArray_DIM(arr_npy, 1);
        // in-place construct the new eigen object using the data extraced from
        // the python object
        new (storage) EigenType(arr_data, actual_rows, actual_cols);

        // Stash the memory chunk pointer for later use by boost.python
        data->convertible = storage;
    }


    //return new eigen object, or throw error (or return 0??)
    static void* convert(PyObject* arr)
    {
        if( !PyArray_Check(arr) )
        {
            throw_error_already_set();
        }
        PyArrayObject* arr_npy = (PyArrayObject*)arr;
        if( !array_is_type<Scalar>(arr_npy) )
        {
            return 0;
        }

        //check dimensions
        int nDims = PyArray_NDIM(arr_npy);
        int actual_rows = PyArray_DIM(arr_npy, 0);
        int actual_cols = 1;
        if( nDims == 2 )
            actual_cols = PyArray_DIM(arr_npy, 1);
        if( Derived::RowsAtCompileTime != -1 &&
            Derived::RowsAtCompileTime != actual_rows )
        {
            return 0;
        }
        if( Derived::ColsAtCompileTime != -1 &&
            Derived::ColsAtCompileTime != actual_cols )
        {
            return 0;
        }

        Scalar* arr_data = (Scalar*)PyArray_DATA( arr_npy );
        return new EigenType(arr_data, actual_rows, actual_cols);
    }
};




//! Convert a numpy array to an Eigen Matrix (memory copy)
template <typename EigenType>
struct eigen_matrix_from_python
{
    typedef typename EigenType::Scalar Scalar;

    eigen_matrix_from_python()
    {
        // for Function( EigenType arr )
        converter::registry::push_back(
            &convertible,
            &construct,
            type_id<EigenType>());

        // for Function( EigenType* arr )
        //     Function( EigenType& arr )
        converter::registry::insert( &convert, type_id<EigenType>() );
    }


    // Determine if arr can be converted to eigen
    static void* convertible(PyObject* arr)
    {
        if( !PyArray_Check(arr) )
        {
            return 0;
        }
        PyArrayObject* arr_npy = (PyArrayObject*)arr;
        if( !array_is_type<Scalar>(arr_npy) )
        {
            return 0;
        }

        //check dimensions
        int nDims = PyArray_NDIM(arr_npy);
        int actual_rows = PyArray_DIM(arr_npy, 0);
        int actual_cols = 1;
        if( nDims == 2 )
            actual_cols = PyArray_DIM(arr_npy, 1);
        if( EigenType::RowsAtCompileTime != -1 &&
            EigenType::RowsAtCompileTime != actual_rows )
        {
            return 0;
        }
        if( EigenType::ColsAtCompileTime != -1 &&
            EigenType::ColsAtCompileTime != actual_cols )
        {
            return 0;
        }
        return arr;
    }


    // Convert arr into an eigen matrix
    static void construct(
        PyObject* arr,
        converter::rvalue_from_python_stage1_data* data)
    {
        // Grab pointer to memory into which to construct the new eigen object
        void* storage = (
          (converter::rvalue_from_python_storage<EigenType>*)
          data)->storage.bytes;

        PyArrayObject* arr_npy = (PyArrayObject*)arr;
        Scalar* arr_data = (Scalar*)PyArray_DATA( arr_npy );
        int nDims = PyArray_NDIM(arr_npy);
        int actual_rows = PyArray_DIM(arr_npy, 0);
        int actual_cols = 1;
        int row_stride = PyArray_STRIDE(arr_npy, 0);
        int col_stride = 1;
        if( nDims == 2 )
            actual_cols = PyArray_DIM(arr_npy, 1);

        // in-place construct the new eigen object using the data extraced from
        // the python object
        new (storage) EigenType(actual_rows, actual_cols);
        EigenType* arr_eigen = reinterpret_cast<EigenType*>(storage);
        memcpy(arr_eigen->data(), arr_data, actual_rows * actual_cols * PyArray_ITEMSIZE(arr_npy));

        // Stash the memory chunk pointer for later use by boost.python
        data->convertible = storage;
    }


    //return new eigen object, or throw error (or return 0??)
    static void* convert(PyObject* arr)
    {
        if( !PyArray_Check(arr) )
        {
            throw_error_already_set();
        }
        PyArrayObject* arr_npy = (PyArrayObject*)arr;
        if( !array_is_type<Scalar>(arr_npy) )
        {
            return 0;
        }

        //check dimensions
        int nDims = PyArray_NDIM(arr_npy);
        int actual_rows = PyArray_DIM(arr_npy, 0);
        int actual_cols = 1;
        if( nDims == 2 )
            actual_cols = PyArray_DIM(arr_npy, 1);
        if( EigenType::RowsAtCompileTime != -1 &&
            EigenType::RowsAtCompileTime != actual_rows )
        {
            return 0;
        }
        if( EigenType::ColsAtCompileTime != -1 &&
            EigenType::ColsAtCompileTime != actual_cols )
        {
            return 0;
        }

        Scalar* arr_data = (Scalar*)PyArray_DATA( arr_npy );
        EigenType* arr_eigen = new EigenType(actual_rows, actual_cols);
        memcpy(arr_eigen->data(), arr_data, actual_rows * actual_cols * PyArray_ITEMSIZE(arr_npy));
        return arr_eigen;
    }
};






template<>
bool array_is_type<float>( PyArrayObject* obj )
{
    if( PyArray_DESCR(obj)->type_num == NPY_FLOAT )
        return true;
    return false;
}

template<>
bool array_is_type<double>( PyArrayObject* obj )
{
    if( PyArray_DESCR(obj)->type_num == NPY_DOUBLE )
        return true;
    return false;
}

template<>
bool array_is_type<char>( PyArrayObject* obj )
{
    if( PyArray_DESCR(obj)->type_num == NPY_BYTE )
        return true;
    return false;
}

template<>
bool array_is_type<unsigned char>( PyArrayObject* obj )
{
    if( PyArray_DESCR(obj)->type_num == NPY_UBYTE )
        return true;
    return false;
}

template<>
bool array_is_type<short>( PyArrayObject* obj )
{
    if( PyArray_DESCR(obj)->type_num == NPY_SHORT )
        return true;
    return false;
}

template<>
bool array_is_type<unsigned short>( PyArrayObject* obj )
{
    if( PyArray_DESCR(obj)->type_num == NPY_USHORT )
        return true;
    return false;
}

template<>
bool array_is_type<int>( PyArrayObject* obj )
{
    if( ( PyArray_DESCR(obj)->type_num == NPY_INT ) ||
        ( PyArray_DESCR(obj)->type_num == NPY_LONG && 
          PyArray_ITEMSIZE(obj) == 4 )
      )
        return true;
    return false;
}

template<>
bool array_is_type<unsigned int>( PyArrayObject* obj )
{
    if( PyArray_DESCR(obj)->type_num == NPY_UINT )
        return true;
    return false;
}

template<>
bool array_is_type<long long>( PyArrayObject* obj )
{
    if( PyArray_DESCR(obj)->type_num == NPY_LONGLONG )
        return true;
    return false;
}

template<>
bool array_is_type<unsigned long long>( PyArrayObject* obj )
{
    if( PyArray_DESCR(obj)->type_num == NPY_ULONGLONG )
        return true;
    return false;
}




void export_numpy_eigen_convert()
{
    using namespace Eigen;
    // (3,)
    eigen_map_from_python<Vector3d>();
    eigen_map_from_python<Vector3f>();
    eigen_map_from_python<Vector3i>();
    eigen_map_from_python< Matrix<unsigned char, 3, 1> >();

    eigen_matrix_from_python<Vector3d>();
    eigen_matrix_from_python<Vector3f>();
    eigen_matrix_from_python<Vector3i>();
    eigen_matrix_from_python< Matrix<unsigned char, 3, 1> >();

    // (n,)
    eigen_map_from_python<VectorXd>();
    eigen_map_from_python<VectorXf>();
    eigen_map_from_python<VectorXi>();
    eigen_map_from_python< Matrix<unsigned char, Dynamic, 1> >();
    eigen_map_from_python< Matrix<long long, Dynamic, 1> >();

    eigen_matrix_from_python<VectorXd>();
    eigen_matrix_from_python<VectorXf>();
    eigen_matrix_from_python<VectorXi>();
    eigen_matrix_from_python< Matrix<unsigned char, Dynamic, 1> >();
    eigen_matrix_from_python< Matrix<long long, Dynamic, 1> >();

    // (n,2)
    eigen_map_from_python< Matrix<double, Dynamic, 2, RowMajor> >();
    eigen_map_from_python< Matrix<float, Dynamic, 2, RowMajor> >();
    eigen_map_from_python< Matrix<int, Dynamic, 2, RowMajor> >();
    eigen_map_from_python< Matrix<unsigned char, Dynamic, 2, RowMajor> >();
    eigen_map_from_python< Matrix<long long, Dynamic, 2, RowMajor> >();

    // (n,3)
    eigen_map_from_python< Matrix<double, Dynamic, 3, RowMajor> >();
    eigen_map_from_python< Matrix<float, Dynamic, 3, RowMajor> >();
    eigen_map_from_python< Matrix<int, Dynamic, 3, RowMajor> >();
    eigen_map_from_python< Matrix<unsigned char, Dynamic, 3, RowMajor> >();
    eigen_map_from_python< Matrix<long long, Dynamic, 3, RowMajor> >();

    // (n,m)
    eigen_map_from_python< Matrix<double, Dynamic, Dynamic, RowMajor> >();
    eigen_map_from_python< Matrix<float, Dynamic, Dynamic, RowMajor> >();
    eigen_map_from_python< Matrix<int, Dynamic, Dynamic, RowMajor> >();
    eigen_map_from_python< Matrix<unsigned char, Dynamic, Dynamic, RowMajor> >();
    eigen_map_from_python< Matrix<long long, Dynamic, Dynamic, RowMajor> >();
}


