
#ifndef LIBWAVELETS_EXPORT_H
#define LIBWAVELETS_EXPORT_H

#ifdef LIBWAVELETS_STATIC_DEFINE
#  define LIBWAVELETS_EXPORT
#  define LIBWAVELETS_NO_EXPORT
#else
#  ifndef LIBWAVELETS_EXPORT
#    ifdef libwavelets_EXPORTS
        /* We are building this library */
#      define LIBWAVELETS_EXPORT __declspec(dllexport)
#    else
        /* We are using this library */
#      define LIBWAVELETS_EXPORT __declspec(dllimport)
#    endif
#  endif

#  ifndef LIBWAVELETS_NO_EXPORT
#    define LIBWAVELETS_NO_EXPORT 
#  endif
#endif

#ifndef LIBWAVELETS_DEPRECATED
#  define LIBWAVELETS_DEPRECATED __declspec(deprecated)
#endif

#ifndef LIBWAVELETS_DEPRECATED_EXPORT
#  define LIBWAVELETS_DEPRECATED_EXPORT LIBWAVELETS_EXPORT LIBWAVELETS_DEPRECATED
#endif

#ifndef LIBWAVELETS_DEPRECATED_NO_EXPORT
#  define LIBWAVELETS_DEPRECATED_NO_EXPORT LIBWAVELETS_NO_EXPORT LIBWAVELETS_DEPRECATED
#endif

#define DEFINE_NO_DEPRECATED 0
#if DEFINE_NO_DEPRECATED
# define LIBWAVELETS_NO_DEPRECATED
#endif

#endif
