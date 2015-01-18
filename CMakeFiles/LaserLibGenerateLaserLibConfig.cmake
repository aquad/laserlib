
# The install-only section is empty for the build tree.
SET(LaserLib_CONFIG_INSTALL_ONLY)

# The "use" file.
SET(LaserLib_USE_FILE ${LaserLib_BINARY_DIR}/CMakeFiles/UseLaserLib.cmake)

# Library directory.
SET(LaserLib_LIBRARY_DIRS_CONFIG ${LIBRARY_OUTPUT_PATH})

# Runtime library directory.
SET(LaserLib_RUNTIME_LIBRARY_DIRS_CONFIG ${LIBRARY_OUTPUT_PATH})

# Binary executable directory.
SET(LaserLib_EXECUTABLE_DIRS_CONFIG ${EXECUTABLE_OUTPUT_PATH})

# Determine the include directories needed.
#SET(LaserLib_INCLUDE_DIRS_CONFIG
#  ${LaserLib_INCLUDE_DIRS_BUILD_TREE}
#  ${LaserLib_INCLUDE_DIRS_SOURCE_TREE}
#  ${LaserLib_INCLUDE_DIRS_SYSTEM}
#)

#-----------------------------------------------------------------------------
# Configure QConfig.cmake for the build tree.
CONFIGURE_FILE(${LaserLib_SOURCE_DIR}/CMakeFiles/LaserLibConfig.cmake.in
               ${LaserLib_BINARY_DIR}/CMakeFiles/LaserLibConfig.cmake @ONLY IMMEDIATE)

#-----------------------------------------------------------------------------
# Settings specific to the install tree.

# The "use" file.
SET(LaserLib_USE_FILE \${LaserLib_INSTALL_PREFIX}/${LaserLib_INSTALL_PACKAGE_DIR}/UseLaserLib.cmake)

# Include directories.
SET(LaserLib_INCLUDE_DIRS_CONFIG
  \${LaserLib_INSTALL_PREFIX}/include
  ${LaserLib_INCLUDE_DIRS_SYSTEM}
)


# Link directories.
IF(CYGWIN AND BUILD_SHARED_LIBS)
  # In Cygwin programs directly link to the .dll files.
  SET(LaserLib_LIBRARY_DIRS_CONFIG \${LaserLib_INSTALL_PREFIX}/${LaserLib_INSTALL_BIN_DIR})
ELSE(CYGWIN AND BUILD_SHARED_LIBS)
  SET(LaserLib_LIBRARY_DIRS_CONFIG \${LaserLib_INSTALL_PREFIX}/${LaserLib_INSTALL_LIB_DIR})
ENDIF(CYGWIN AND BUILD_SHARED_LIBS)

# Executable and runtime library directories.
IF(WIN32)
  SET(LaserLib_EXECUTABLE_DIRS_CONFIG \${LaserLib_INSTALL_PREFIX}/${LaserLib_INSTALL_BIN_DIR})
  SET(LaserLib_RUNTIME_LIBRARY_DIRS_CONFIG \${LaserLib_INSTALL_PREFIX}/${LaserLib_INSTALL_BIN_DIR})
ELSE(WIN32)
  SET(LaserLib_EXECUTABLE_DIRS_CONFIG \${LaserLib_INSTALL_PREFIX}/${LaserLib_INSTALL_BIN_DIR})
  SET(LaserLib_RUNTIME_LIBRARY_DIRS_CONFIG \${LaserLib_INSTALL_PREFIX}/${LaserLib_INSTALL_LIB_DIR})
ENDIF(WIN32)

IF(WIN32)
  SET(LaserLib_EXE_EXT ".exe")
ENDIF(WIN32)

#-----------------------------------------------------------------------------
# Configure LaserLibConfig.cmake for the install tree.

# Construct the proper number of GET_FILENAME_COMPONENT(... PATH)
# calls to compute the installation prefix from LaserLib_DIR.
#STRING(REGEX REPLACE "/" ";" LaserLib_INSTALL_PACKAGE_DIR_COUNT
#  "${LaserLib_INSTALL_PACKAGE_DIR}")
SET(LaserLib_CONFIG_INSTALL_ONLY "
# Compute the installation prefix from LaserLib_DIR.
SET(LaserLib_INSTALL_PREFIX \"\${LaserLib_DIR}\")
")
#FOREACH(p ${LaserLib_INSTALL_PACKAGE_DIR_COUNT})
#  SET(LaserLib_CONFIG_INSTALL_ONLY
#    "${LaserLib_CONFIG_INSTALL_ONLY}GET_FILENAME_COMPONENT(LaserLib_INSTALL_PREFIX \"\${LaserLib_INSTALL_PREFIX}\" PATH)\n"
#    )
#ENDFOREACH(p)

# The install tree only has one configuration.
SET(LaserLib_CONFIGURATION_TYPES_CONFIG)

IF(CMAKE_CONFIGURATION_TYPES)
  # There are multiple build configurations.  Configure one
  # QConfig.cmake for each configuration.
  FOREACH(config ${CMAKE_CONFIGURATION_TYPES})
    SET(LaserLib_BUILD_TYPE_CONFIG ${config})
    CONFIGURE_FILE(${LaserLib_SOURCE_DIR}/CMakeFiles/LaserLibConfig.cmake.in
                   ${LaserLib_BINARY_DIR}/Utilities/${config}/LaserLibConfig.cmake
                   @ONLY IMMEDIATE)
  ENDFOREACH(config)

  # Install the config file corresponding to the build configuration
  # specified when building the install target.  The BUILD_TYPE variable
  # will be set while CMake is processing the install files.
  INSTALL(
    FILES
      "${LaserLib_BINARY_DIR}/Utilities/\${BUILD_TYPE}/LaserLibConfig.cmake"
    DESTINATION ${LaserLib_INSTALL_PACKAGE_DIR}
  )

ELSE(CMAKE_CONFIGURATION_TYPES)
  # There is only one build configuration.  Configure one QConfig.cmake.
  SET(LaserLib_BUILD_TYPE_CONFIG ${CMAKE_BUILD_TYPE})
  CONFIGURE_FILE(${LaserLib_SOURCE_DIR}/CMakeFiles/LaserLibConfig.cmake.in
                 ${LaserLib_BINARY_DIR}/Utilities/LaserLibConfig.cmake @ONLY IMMEDIATE)

  # Setup an install rule for the config file.
  INSTALL(
    FILES
      "${LaserLib_BINARY_DIR}/Utilities/LaserLibConfig.cmake"
    DESTINATION ${LaserLib_INSTALL_PACKAGE_DIR}
  )
ENDIF(CMAKE_CONFIGURATION_TYPES)
