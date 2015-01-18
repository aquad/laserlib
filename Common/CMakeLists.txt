SET( PROJECT "Common" )
SET ( TARGET_NAME Laser${PROJECT} )
SET( dir ${SOURCE_CODE_BASE_DIR}/${PROJECT} )

FILE( GLOB includes ${dir}/*.h )
FILE( GLOB source ${dir}/*.cpp )
SOURCE_GROUP( ${PROJECT} FILES ${includes} ${source} )

##source_group doesn't work without add_library- so add this instead
# add_custom_target( ${TARGET_NAME} SOURCES ${includes} ${source} )

SOURCE_GROUP( ${PROJECT} FILES ${includes} ${source})

ADD_LIBRARY( ${TARGET_NAME} ${LaserLib_LIBRARY_TYPE} ${includes} ${source} )
SET_TARGET_PROPERTIES( ${TARGET_NAME} PROPERTIES ${LaserLib_LIBRARY_PROPERTIES} )
TARGET_LINK_LIBRARIES(${TARGET_NAME} ${FLANN_LIBRARY})

# install
INSTALL( FILES ${includes} DESTINATION ${CMAKE_INSTALL_PREFIX}/${LaserLib_INSTALL_INCLUDE_DIR}/${PROJECT} )
INSTALL(
    TARGETS ${TARGET_NAME}
    RUNTIME DESTINATION ${CMAKE_INSTALL_PREFIX}/${LaserLib_INSTALL_BIN_DIR} COMPONENT Runtime     # .exe, .dll
    LIBRARY DESTINATION ${CMAKE_INSTALL_PREFIX}/${LaserLib_INSTALL_LIB_DIR} COMPONENT Runtime     # .so, mod.dll
    ARCHIVE DESTINATION ${CMAKE_INSTALL_PREFIX}/${LaserLib_INSTALL_LIB_DIR} COMPONENT Development # .a, .lib
)

