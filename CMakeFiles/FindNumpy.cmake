# - Find numpy

# This file taken from  http://avogadro.openmolecules.net/
#      git://github.com/timvdm/avogadro.git

# Find the native numpy includes
# This module defines
#  NUMPY_INCLUDE_DIR, where to find numpy/arrayobject.h, etc.
#  NUMPY_FOUND, If false, do not try to use numpy headers.


exec_program ("${PYTHON_EXECUTABLE}"
ARGS "-c \"import numpy; print numpy.get_include()\""
OUTPUT_VARIABLE NUMPY_INCLUDE_DIR
RETURN_VALUE NUMPY_NOT_FOUND)

if (NUMPY_NOT_FOUND)
  # Did not successfully include numpy
  set(NUMPY_FOUND FALSE)
else ()
  # successful
  set (NUMPY_FOUND TRUE)
  set (NUMPY_INCLUDE_DIR ${NUMPY_INCLUDE_DIR} CACHE PATH "Numpy include path")
endif ()

if (NUMPY_FOUND)
  if (NOT NUMPY_FIND_QUIETLY)
    message (STATUS "Numpy headers found")
  endif (NOT NUMPY_FIND_QUIETLY)
else (NUMPY_FOUND)
  if (NUMPY_FIND_REQUIRED)
    message (FATAL_ERROR "Numpy headers missing")
  endif (NUMPY_FIND_REQUIRED)
endif (NUMPY_FOUND)

