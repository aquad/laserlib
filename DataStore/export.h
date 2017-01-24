/* export.h
 *
 * \author     Hamdi Sahloul
 * \date       24-01-2017
*/

#ifndef LASERLIB_DATASTORE_EXPORT_H
#define LASERLIB_DATASTORE_EXPORT_H


#if defined(_WIN32) && defined(LASERLIB_EXPORT_API)
  #if defined(LaserDataStore_EXPORTS)
    #define  LASERLIB_DATASTORE_EXPORT __declspec(dllexport)
  #else
    #define  LASERLIB_DATASTORE_EXPORT __declspec(dllimport)
  #endif /* LaserDataStore_EXPORTS */
#else /* _WIN32 & LASERLIB_EXPORT_API */
 #define LASERLIB_DATASTORE_EXPORT
#endif

#if defined(_WIN32) && defined(LASERLIB_EXPORT_API) && defined(LaserDataStore_EXPORTS)
 #define  LASERLIB_DATASTORE_EXTERN extern
#else
 #define  LASERLIB_DATASTORE_EXTERN
#endif

#if defined(_WIN32) && defined(LASERLIB_EXPORT_API) && !defined(LaserDataStore_EXPORTS)
 #define  LASERLIB_DATASTORE_IMPORT __declspec(dllimport)
#else
 #define  LASERLIB_DATASTORE_IMPORT
#endif

#endif //LASERLIB_DATASTORE_EXPORT_H
