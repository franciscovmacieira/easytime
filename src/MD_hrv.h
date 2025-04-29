// File: MD_hrv.h
// Purpose: Declares functions implemented in MD_hrv.c

#ifndef MD_hrv_h
#define MD_hrv_h

#include <stddef.h> // Commonly included for size_t, though int is used here

// Define the export/import macro based on whether we are BUILDING the DLL
#ifdef _WIN32 // Only needed for Windows DLLs
    #ifdef BUILDING_C_METRICS_DLL // This macro will be defined when compiling the DLL sources
        #define C_METRICS_API __declspec(dllexport)
    #else // If not building the DLL, assume we are using (importing from) it
        #define C_METRICS_API __declspec(dllimport)
    #endif
#else // Not on Windows, API macro does nothing
    #define C_METRICS_API
#endif

// Add extern "C" guards for better C/C++ compatibility when linking
#ifdef __cplusplus
extern "C" {
#endif

// --- Function Declaration (Prototype) ---
// Use the API macro here
C_METRICS_API double MD_hrv_classic_pnn40(const double y[], const int size);

// Add declarations for any OTHER functions defined in MD_hrv.c here...
// Make sure to use C_METRICS_API for all functions you want to export


#ifdef __cplusplus
} // End extern "C"
#endif

#endif /* MD_hrv_h */