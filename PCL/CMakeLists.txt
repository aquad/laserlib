SET( PROJECT "PCL" )
SET ( TARGET_NAME Laser${PROJECT} )
SET( dir ${SOURCE_CODE_BASE_DIR}/${PROJECT} )

if(COMMAND cmake_policy)
 cmake_policy(SET CMP0003 NEW)
endif(COMMAND cmake_policy)

FILE( GLOB includes ${dir}/*.h)
FILE( GLOB source ${dir}/*.cpp)

SOURCE_GROUP( ${PROJECT} FILES ${includes} ${source})

ADD_LIBRARY( ${TARGET_NAME} ${LaserLib_LIBRARY_TYPE} ${includes} ${source} )
SET_TARGET_PROPERTIES( ${TARGET_NAME} PROPERTIES ${LaserLib_LIBRARY_PROPERTIES} )
TARGET_LINK_LIBRARIES ( ${TARGET_NAME} LaserDataStore# ${PCL_LIBRARIES} )
    ${PCL_COMMON_LIBRARIES}
	${PCL_RANGE_IMAGE_LIBRARIES}
	${PCL_KEYPOINTS_LIBRARIES}
	${PCL_RANGE_IMAGE_BORDER_EXTRACTOR_LIBRARIES} 
	${PCL_FEATURES_LIBRARIES} 
    ${PCL_OCTREE_LIBRARIES})

IF( WIN32 )
  TARGET_LINK_LIBRARIES( ${TARGET_NAME}
                         debug libboost_system-vc90-mt-gd-${Boost_VERSION_SUFFIX}.lib
                         optimized libboost_system-vc90-mt-${Boost_VERSION_SUFFIX}.lib
                         debug libboost_thread-vc90-mt-gd-${Boost_VERSION_SUFFIX}.lib
                         optimized libboost_thread-vc90-mt-${Boost_VERSION_SUFFIX}.lib )

ELSE( WIN32 )
  TARGET_LINK_LIBRARIES ( ${TARGET_NAME} boost_system boost_thread )
ENDIF( WIN32 )

# install
INSTALL( FILES ${includes} DESTINATION ${CMAKE_INSTALL_PREFIX}/${LaserLib_INSTALL_INCLUDE_DIR}/${PROJECT} )
INSTALL(
    TARGETS ${TARGET_NAME}
    RUNTIME DESTINATION ${CMAKE_INSTALL_PREFIX}/${LaserLib_INSTALL_BIN_DIR} COMPONENT Runtime     # .exe, .dll
    LIBRARY DESTINATION ${CMAKE_INSTALL_PREFIX}/${LaserLib_INSTALL_LIB_DIR} COMPONENT Runtime     # .so, mod.dll
    ARCHIVE DESTINATION ${CMAKE_INSTALL_PREFIX}/${LaserLib_INSTALL_LIB_DIR} COMPONENT Development # .a, .lib
)

