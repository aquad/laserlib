SET( PROJECT "Features" )
SET ( TARGET_NAME Laser${PROJECT} )
SET( dir ${SOURCE_CODE_BASE_DIR}/${PROJECT} )

if(COMMAND cmake_policy)
 cmake_policy(SET CMP0003 NEW)
endif(COMMAND cmake_policy)

FILE( GLOB includes ${dir}/*.h)
FILE( GLOB source ${dir}/*.cpp)

IF( NOT LaserLib_USE_FLANN )
  FILE( GLOB flann_includes ${dir}/PCAFrame.h ${dir}/PCAGrid.h )
  FILE( GLOB flann_source ${dir}/PCAFrame.cpp ${dir}/PCAGrid.cpp )
  LIST( REMOVE_ITEM includes ${flann_includes} )
  LIST( REMOVE_ITEM source ${flann_source} )
ENDIF()


SOURCE_GROUP( ${PROJECT} FILES ${includes} ${source})

#INCLUDE_DIRECTORIES ( ${SOURCE_CODE_BASE_DIR} )

ADD_LIBRARY( ${TARGET_NAME} ${LaserLib_LIBRARY_TYPE} ${includes} ${source} )
SET_TARGET_PROPERTIES( ${TARGET_NAME} PROPERTIES ${LaserLib_LIBRARY_PROPERTIES} )
#ADD_DEPENDENCIES( ${TARGET_NAME} LaserDataStore )
#TARGET_LINK_LIBRARIES ( ${TARGET_NAME} LaserDataStore )

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
