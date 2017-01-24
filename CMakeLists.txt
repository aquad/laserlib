SET( ProjectName LaserLib )
PROJECT(${ProjectName})
cmake_policy(SET CMP0015 NEW)
CMAKE_MINIMUM_REQUIRED(VERSION 2.5.0)

IF(WIN32)
  ADD_DEFINITIONS(-DNOMINMAX)
ENDIF(WIN32)

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/CMakeFiles/")

SET ( SOURCE_CODE_BASE_DIR ${CMAKE_CURRENT_SOURCE_DIR} )
SET ( LIBRARY_OUTPUT_PATH ${CMAKE_BINARY_DIR}/lib )

# Install locations
IF(NOT LaserLib_INSTALL_INCLUDE_DIR)
    SET(LaserLib_INSTALL_INCLUDE_DIR include/${ProjectName})
ENDIF()

IF(NOT LaserLib_INSTALL_LIB_DIR)
  SET(LaserLib_INSTALL_LIB_DIR lib)
ENDIF()

IF(NOT LaserLib_INSTALL_BIN_DIR)
  SET(LaserLib_INSTALL_BIN_DIR bin)
ENDIF()

# CMake scripts for packaging
IF(NOT LaserLib_INSTALL_PACKAGE_DIR)
  SET(LaserLib_INSTALL_PACKAGE_DIR CMakeFiles)
ENDIF()

# generate the UseLaserLib.cmake file
CONFIGURE_FILE(
  ${PROJECT_SOURCE_DIR}/CMakeFiles/Use${ProjectName}.cmake.in
  ${PROJECT_BINARY_DIR}/CMakeFiles/Use${ProjectName}.cmake COPYONLY IMMEDIATE
)

#Install cmake related files
#FILE( GLOB LaserLib_PACKAGE_CMAKEFILES ${CMAKE_CURRENT_SOURCE_DIR}/CMakeFiles/*.cmake )
SET( LaserLib_PACKAGE_CMAKEFILES ${LaserLib_PACKAGE_CMAKEFILES} ${PROJECT_BINARY_DIR}/CMakeFiles/Use${ProjectName}.cmake )
INSTALL( FILES ${LaserLib_PACKAGE_CMAKEFILES}
         DESTINATION ${CMAKE_INSTALL_PREFIX}/${LaserLib_INSTALL_PACKAGE_DIR} )

INCLUDE(${LaserLib_SOURCE_DIR}/CMakeFiles/LaserLibIncludeDirectories.cmake)

# generate the LaserLibConfig.cmake file
INCLUDE(${LaserLib_SOURCE_DIR}/CMakeFiles/${ProjectName}Generate${ProjectName}Config.cmake)

IF(NOT LaserLib_INSTALL_BIN_DIR)
  SET(LaserLib_INSTALL_BIN_DIR bin)
ENDIF()

#-----------------------------------------------------------------------------
SET (EXECUTABLE_OUTPUT_PATH ${PROJECT_BINARY_DIR}/bin CACHE PATH
   "Single output directory for building all executables.")

SET (LIBRARY_OUTPUT_PATH ${PROJECT_BINARY_DIR}/lib CACHE PATH
   "Single output directory for building all libraries.")

#-----------------------------------------------------------------------------
#Options
INCLUDE(CMakeFiles/CMakeOptions.cmake)

###
# this is copy paste form http://www.itk.org/Wiki/CMake_RPATH_handling
# in order to always make a full statement RPATH
###
SET(CMAKE_SKIP_BUILD_RPATH  FALSE)
SET(CMAKE_BUILD_WITH_INSTALL_RPATH FALSE) 
SET(CMAKE_INSTALL_RPATH "${CMAKE_INSTALL_PREFIX}/lib")

#-----------------------------------------------------------------------------
# Find Libraries
INCLUDE_DIRECTORIES ( ${SOURCE_CODE_BASE_DIR} )

# Find Eigen
IF( WIN32 )
  FIND_PATH( Eigen_INCLUDE_DIR NAMES signature_of_eigen3_matrix_library PATHS $ENV{EIGEN_ROOT} )
ELSE()
  FIND_PATH( Eigen_INCLUDE_DIR NAMES signature_of_eigen3_matrix_library PATHS $ENV{EIGEN_ROOT} /usr/include/eigen3 /usr/local/include/eigen3 )
ENDIF()
IF( NOT EXISTS ${Eigen_INCLUDE_DIR} )
  MESSAGE( "could not find Eigen - should be the path containing the Eigen 3 directory and /Eigen/Eigen, /Eigen/Core, etc..." )
ELSE()
  INCLUDE_DIRECTORIES( ${Eigen_INCLUDE_DIR} )
ENDIF()

IF( LaserLib_USE_FLANN )
  FIND_PACKAGE(Flann REQUIRED)
  INCLUDE_DIRECTORIES(${FLANN_INCLUDE_DIRS})
ENDIF()

IF(COMMAND cmake_policy)
  cmake_policy(SET CMP0007 OLD)
ENDIF()

IF(LaserLib_USE_PCL)
  FIND_PACKAGE(PCL REQUIRED COMPONENTS
    common keypoints features)
  INCLUDE_DIRECTORIES(${PCL_INCLUDE_DIRS})
  LINK_DIRECTORIES(${PCL_LINK_DIRECTORIES})
ENDIF()

#-----------------------------------------------------------------------------

IF( BUILD_SHARED_LIBS )
    SET( Boost_USE_STATIC_LIBS OFF )
ELSE()
    SET( Boost_USE_STATIC_LIBS ON )
ENDIF( NOT BUILD_SHARED_LIBS )
FIND_PACKAGE( Boost COMPONENTS filesystem system date_time REQUIRED)
INCLUDE_DIRECTORIES( ${Boost_INCLUDE_DIRS} )
LINK_DIRECTORIES( ${Boost_LIBRARY_DIRS} )

IF(LaserLib_USE_OPENMP)
  FIND_PACKAGE( OpenMP )
  SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
  SET(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
ENDIF()


#-----------------------------------------------------------------------------
SET( LaserLib_LIBRARY_PROPERTIES  ${LaserLib_LIBRARY_PROPERTIES} DEBUG_POSTFIX "-d" )
SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++0x")

IF(BUILD_LaserLib_SHARED)
  SET(LaserLib_LIBRARY_TYPE SHARED)
  ADD_DEFINITIONS(-DLASERLIB_EXPORT_API)
ELSE()
  SET(LaserLib_LIBRARY_TYPE STATIC)
ENDIF()

#pass options (using python/openmp) to a header file with '#define' entries:
CONFIGURE_FILE(${CMAKE_CURRENT_SOURCE_DIR}/LaserLibConfigure.h.in
  ${CMAKE_CURRENT_BINARY_DIR}/LaserLibConfig.h)
INCLUDE_DIRECTORIES( ${CMAKE_CURRENT_BINARY_DIR} )

# Directories
ADD_SUBDIRECTORY( Common )
ADD_SUBDIRECTORY( DataStore )
ADD_SUBDIRECTORY( Features )
IF( LaserLib_USE_FLANN )
  ADD_SUBDIRECTORY( Misc )
ENDIF()

IF( LaserLib_USE_PCL )
  ADD_SUBDIRECTORY( PCL )
ENDIF()

IF( LaserLib_USE_FLANN AND LaserLib_USE_PYTHON )
  ADD_SUBDIRECTORY( LaserPy )
ENDIF()

