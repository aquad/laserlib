/* export.h
 *
 * \author     Hamdi Sahloul
 * \date       24-01-2017
*/

#ifndef LASERLIB_MISC_EXPORT_H
#define LASERLIB_MISC_EXPORT_H


#if defined(_WIN32) && defined(LASERLIB_EXPORT_API)
  #if defined(LaserMisc_EXPORTS)
    #define  LASERLIB_MISC_EXPORT __declspec(dllexport)
  #else
    #define  LASERLIB_MISC_EXPORT __declspec(dllimport)
  #endif /* LaserMisc_EXPORTS */
#else /* _WIN32 & LASERLIB_EXPORT_API */
 #define LASERLIB_MISC_EXPORT
#endif

#if defined(_WIN32) && defined(LASERLIB_EXPORT_API) && defined(LaserMisc_EXPORTS)
 #define  LASERLIB_MISC_EXTERN extern
#else
 #define  LASERLIB_MISC_EXTERN
#endif

#if defined(_WIN32) && defined(LASERLIB_EXPORT_API) && !defined(LaserMisc_EXPORTS)
 #define  LASERLIB_MISC_IMPORT __declspec(dllimport)
#else
 #define  LASERLIB_MISC_IMPORT
#endif

#endif //LASERLIB_MISC_EXPORT_H
