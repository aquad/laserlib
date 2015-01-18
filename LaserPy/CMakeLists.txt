SET( PROJECT "LaserPy" )
SET( TARGET_NAME ${PROJECT} )
SET( dir ${SOURCE_CODE_BASE_DIR}/${PROJECT} )

IF(COMMAND cmake_policy)
 cmake_policy(SET CMP0003 NEW)
 cmake_policy(SET CMP0015 NEW)
ENDIF(COMMAND cmake_policy)

FILE( GLOB includes ${dir}/*.h)
FILE( GLOB source ${dir}/*.cpp)

FILE( GLOB datastore_includes ${SOURCE_CODE_BASE_DIR}/DataStore/Python/*.h )
FILE( GLOB datastore_source ${SOURCE_CODE_BASE_DIR}/DataStore/Python/*.cpp )
IF( NOT LaserLib_USE_FLANN )
  FILE( GLOB flann_includes ${SOURCE_CODE_BASE_DIR}/DataStore/Python/FlannKDTree_py.h )
  FILE( GLOB flann_source ${SOURCE_CODE_BASE_DIR}/DataStore/Python/FlannKDTree_py.cpp )
  LIST( REMOVE_ITEM datastore_includes ${flann_includes} )
  LIST( REMOVE_ITEM datastore_source ${flann_source} )
ENDIF()


FILE( GLOB features_includes ${SOURCE_CODE_BASE_DIR}/Features/Python/*.h )
FILE( GLOB features_source ${SOURCE_CODE_BASE_DIR}/Features/Python/*.cpp )
IF( NOT LaserLib_USE_FLANN )
  FILE( GLOB flann_source
    ${SOURCE_CODE_BASE_DIR}/Features/Python/PCAFrame_py.cpp
    ${SOURCE_CODE_BASE_DIR}/Features/Python/PCAGrid_py.cpp )
  LIST( REMOVE_ITEM features_source ${flann_source} )
ENDIF()


FILE( GLOB misc_includes ${SOURCE_CODE_BASE_DIR}/Misc/Python/*.h )
FILE( GLOB misc_source ${SOURCE_CODE_BASE_DIR}/Misc/Python/*.cpp )

IF( LaserLib_USE_PCL )
  FILE( GLOB pcl_includes ${SOURCE_CODE_BASE_DIR}/PCL/Python/*.h )
  FILE( GLOB pcl_source ${SOURCE_CODE_BASE_DIR}/PCL/Python/*.cpp )
  SOURCE_GROUP( "PCL\\Python" FILES ${pcl_source} ${pcl_includes} )
ENDIF()

IF( LaserLib_USE_FLANN AND LaserLib_USE_PCL )
    FILE( GLOB search_includes ${SOURCE_CODE_BASE_DIR}/Search/Python/*.h )
    FILE( GLOB search_source ${SOURCE_CODE_BASE_DIR}/Search/Python/*.cpp )
ENDIF( LaserLib_USE_FLANN AND LaserLib_USE_PCL )

FILE( GLOB graphics_includes ${SOURCE_CODE_BASE_DIR}/Graphics/Python/*.h )
FILE( GLOB graphics_source ${SOURCE_CODE_BASE_DIR}/Graphics/Python/*.cpp )

SOURCE_GROUP( ${PROJECT} FILES ${source} ${includes} )
SOURCE_GROUP( "DataStore\\Python" FILES ${datastore_source} ${datastore_includes} )
SOURCE_GROUP( "Features\\Python" FILES ${features_source} ${features_includes} )
SOURCE_GROUP( "Misc\\Python" FILES ${misc_source} ${misc_includes} )
SOURCE_GROUP( "Graphics\\Python" FILES ${graphics_source} ${graphics_includes} )
IF( LaserLib_USE_FLANN AND LaserLib_USE_PCL  )
    SOURCE_GROUP( "Search\\Python" FILES ${search_source} ${search_includes} )
ENDIF( LaserLib_USE_FLANN AND LaserLib_USE_PCL  )

#needs to be a shared library
ADD_LIBRARY( ${TARGET_NAME} SHARED 
    ${includes} ${source}
    ${datastore_includes} ${datastore_source}
    ${features_includes} ${features_source}
    ${misc_includes} ${misc_source}
    ${pcl_includes} ${pcl_source}
    ${search_includes} ${search_source}
    ${graphics_includes} ${graphics_source} )

SET_TARGET_PROPERTIES( ${TARGET_NAME} PROPERTIES ${LaserLib_LIBRARY_PROPERTIES} )
#ADD_DEPENDENCIES( ${TARGET_NAME} LaserDataStore LaserFeatures LaserMisc )
TARGET_LINK_LIBRARIES ( ${TARGET_NAME} 
	${FLANN_LIBRARY} 
    LaserCommon LaserDataStore LaserFeatures LaserMisc LaserGraphics)

IF( LaserLib_USE_PCL )
  TARGET_LINK_LIBRARIES( ${TARGET_NAME} 
	LaserPCL ) #${PCL_LIBRARIES} )
#	${PCL_RANGE_IMAGE_LIBRARIES}
#	${PCL_KEYPOINTS_LIBRARIES}
#	${PCL_RANGE_IMAGE_BORDER_EXTRACTOR_LIBRARIES} )
ENDIF()

IF( LaserLib_USE_FLANN AND LaserLib_USE_PCL )
    TARGET_LINK_LIBRARIES( ${TARGET_NAME} LaserSearch )
ENDIF()

IF( WIN32 )
  TARGET_LINK_LIBRARIES( ${TARGET_NAME}
                         debug libboost_python-vc90-mt-gd-${Boost_VERSION_SUFFIX}.lib
                         optimized libboost_python-vc90-mt-${Boost_VERSION_SUFFIX}.lib )

ELSE( WIN32 )
    # this should work everywhere, right?
    TARGET_LINK_LIBRARIES ( ${TARGET_NAME} ${Boost_PYTHON_LIBRARY} ) 
ENDIF( WIN32 )


#needed to link properly to LaserDataStore on install
#SET_TARGET_PROPERTIES( ${TARGET_NAME} PROPERTIES 
#	SKIP_BUILD_RPATH FALSE
#	BUILD_WITH_INSTALL_RPATH FALSE
#	INSTALL_RPATH ${CMAKE_INSTALL_PREFIX}/${LaserLib_INSTALL_LIB_DIR}
#	INSTALL_RPATH_USE_LINK_PATH FALSE )

#.pyd in windows, no lib prefix in linux (affects python import)
IF(WIN32)
  SET_TARGET_PROPERTIES( ${TARGET_NAME} PROPERTIES SUFFIX .pyd )
ELSE(WIN32)
  SET_TARGET_PROPERTIES( ${TARGET_NAME} PROPERTIES PREFIX "")
ENDIF(WIN32)
#debug -d screws up import
SET_TARGET_PROPERTIES( ${TARGET_NAME} PROPERTIES DEBUG_POSTFIX "" )

#install header files as arranged in the source directory
SET(instDir ${CMAKE_INSTALL_PREFIX}/${LaserLib_INSTALL_INCLUDE_DIR})
INSTALL( FILES ${includes} DESTINATION ${instDir}/${PROJECT} )
INSTALL( FILES ${datastore_includes} DESTINATION ${instDir}/DataStore/Python )
INSTALL( FILES ${features_includes} DESTINATION ${instDir}/Features/Python )
INSTALL( FILES ${misc_includes} DESTINATION ${instDir}/Misc/Python )
INSTALL( FILES ${graphics_includes} DESTINATION ${instDir}/Graphics/Python )
IF( LaserLib_USE_FLANN AND LaserLib_USE_PCL  )
    INSTALL( FILES ${search_includes} DESTINATION ${instDir}/Search/Python)
ENDIF()
IF( LaserLib_USE_PCL )
  INSTALL( FILES ${pcl_includes} DESTINATION ${instDir}/PCL/Python )
ENDIF()

INSTALL(
    TARGETS ${TARGET_NAME}
    LIBRARY DESTINATION ${PYTHON_EXTENSION_INSTALL_PATH} COMPONENT Runtime     # .so, mod.dll
)


