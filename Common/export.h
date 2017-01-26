/* export.h
 *
 * \author     Hamdi Sahloul
 * \date       24-01-2017
*/

#ifndef LASERLIB_COMMON_EXPORT_H
#define LASERLIB_COMMON_EXPORT_H


#if defined(_WIN32) && defined(LASERLIB_EXPORT_API)
  #if defined(LaserCommon_EXPORTS)
    #define  LASERLIB_COMMON_EXPORT __declspec(dllexport)
  #else
    #define  LASERLIB_COMMON_EXPORT __declspec(dllimport)
  #endif /* LaserCommon_EXPORTS */
#else /* _WIN32 & LASERLIB_EXPORT_API */
 #define LASERLIB_COMMON_EXPORT
#endif

#if defined(_WIN32) && defined(LASERLIB_EXPORT_API) && defined(LaserCommon_EXPORTS)
 #define  LASERLIB_COMMON_EXTERN extern
#else
 #define  LASERLIB_COMMON_EXTERN
#endif

#if defined(_WIN32) && defined(LASERLIB_EXPORT_API) && !defined(LaserCommon_EXPORTS)
 #define  LASERLIB_COMMON_IMPORT __declspec(dllimport)
#else
 #define  LASERLIB_COMMON_IMPORT
#endif

#endif //LASERLIB_COMMON_EXPORT_H
