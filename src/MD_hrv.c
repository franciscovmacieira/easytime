// File: MD_hrv.c
// Purpose: Implements functions declared in MD_hrv.h

// Define this BEFORE including the header when building the DLL itself
#define BUILDING_C_METRICS_DLL

#include "MD_hrv.h"      // Include the corresponding header FIRST
#include "stats.h"       // Assuming 'diff' function is declared/defined via this
#include <math.h>        // For fabs, isnan
#include <stdlib.h>      // For malloc, free
#include <stdio.h>       // For perror

// Remove the old local DLLEXPORT macro definition - it's handled in the header now
/*
#ifdef _WIN32
    #define DLLEXPORT __declspec(dllexport)
#else
    #define DLLEXPORT // Define as empty for other platforms (like Linux/macOS)
#endif
*/

// --- Function Definition ---
// Use the C_METRICS_API macro from the header
C_METRICS_API double MD_hrv_classic_pnn40(const double y[], const int size){

    // --- Input Validation ---
    // NaN check
    for(int i = 0; i < size; i++) {
        // Ensure isnan is available (usually from math.h with C99+)
        if(isnan(y[i])) {
            fprintf(stderr, "Warning: NaN input to MD_hrv_classic_pnn40 for array of size %d\n", size);
            return -1.0; // Return specific code for NaN input
        }
    }

    // Need at least 2 points to calculate differences
    if (size < 2) {
        return 0.0; // Proportion is 0 if no differences possible
    }

    // --- Constants ---
    const double pNNx = 40.0;
    const double scale = 1000.0;

    // --- Calculate Differences ---
    // Allocate memory, check for failure
    double * Dy = malloc((size-1) * sizeof(double));
    if (Dy == NULL) {
        perror("Failed to allocate memory for Dy in MD_hrv_classic_pnn40");
        return -2.0; // Return specific code for memory error
    }

    // Call diff function (ensure it's implemented in linked files, e.g., stats.c)
    // Assuming signature: void diff(const double y[], int size, double dy[]);
    diff(y, size, Dy);

    // --- Count Exceeding Threshold ---
    double pnn40_count = 0.0; // Use double for count
    for(int i = 0; i < size-1; i++){
        if(fabs(Dy[i]) * scale > pNNx){
            pnn40_count += 1.0;
        }
    }

    // --- Cleanup and Return ---
    free(Dy); // Free allocated memory

    // Calculate and return proportion (ensuring float division)
    return pnn40_count / (double)(size-1);
}

// --- Implementations for other functions declared in MD_hrv.h would go here ---
// --- Make sure they also use C_METRICS_API if they need to be exported ---

// --- Crucially, make sure the 'diff' function used above is implemented in one ---
// --- of the other .c files you are compiling (e.g., stats.c or helper_functions.c) ---
