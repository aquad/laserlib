/* export.h
 *
 * \author     Hamdi Sahloul
 * \date       24-01-2017
*/

#ifndef LASERLIB_FEATURES_EXPORT_H
#define LASERLIB_FEATURES_EXPORT_H


#if defined(_WIN32) && defined(LASERLIB_EXPORT_API)
  #if defined(LaserFeatures_EXPORTS)
    #define  LASERLIB_FEATURES_EXPORT __declspec(dllexport)
  #else
    #define  LASERLIB_FEATURES_EXPORT __declspec(dllimport)
  #endif /* LaserFeatures_EXPORTS */
#else /* _WIN32 & LASERLIB_EXPORT_API */
 #define LASERLIB_FEATURES_EXPORT
#endif

#if defined(_WIN32) && defined(LASERLIB_EXPORT_API) && defined(LaserFeatures_EXPORTS)
 #define  LASERLIB_FEATURES_EXTERN extern
#else
 #define  LASERLIB_FEATURES_EXTERN
#endif

#if defined(_WIN32) && defined(LASERLIB_EXPORT_API) && !defined(LaserFeatures_EXPORTS)
 #define  LASERLIB_FEATURES_IMPORT __declspec(dllimport)
#else
 #define  LASERLIB_FEATURES_IMPORT
#endif

#endif //LASERLIB_FEATURES_EXPORT_H
