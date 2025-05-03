#pragma once

#ifdef BACKENDS_EXPORTS
    #ifdef WIN32
        #define BACKENDS_EXPORT __declspec(dllexport)
    #else
        #define BACKENDS_EXPORT
    #endif
#else
    #ifdef WIN32
        #define BACKENDS_EXPORT __declspec(dllimport)
    #else
        #define BACKENDS_EXPORT
    #endif
#endif
