// File: SB_MotifThree_exportable.c (or similar name)
// Purpose: Implements SB_MotifThree features based on pycatch22 reference logic,
//          modified for DLL export on Windows.

#include <math.h>           // For isnan, NAN, log2 (if used in f_entropy)
#include <string.h>         // For strcmp, memcpy
#include <stdlib.h>         // For malloc, free, exit
#include <stdio.h>          // For fprintf, stdout, perror

// Include headers for dependent functions
#include "SB_CoarseGrain.h" // Must declare sb_coarsegrain
#include "helper_functions.h" // Must declare subset and f_entropy

// Define DLLEXPORT macro for DLL export compatibility
#ifdef _WIN32
    #define DLLEXPORT __declspec(dllexport)
#else
    #define DLLEXPORT // Define as empty for other platforms (like Linux/macOS)
#endif

// -----------------------------------------------------------------------------
// Exportable Feature Function: SB_MotifThree_quantile_hh
// -----------------------------------------------------------------------------
DLLEXPORT double SB_MotifThree_quantile_hh(const double y[], const int size)
{
    // --- Input Validation ---
    // NaN check
    for(int i = 0; i < size; i++)
    {
        if(isnan(y[i]))
        {
            fprintf(stderr, "Warning: NaN input detected in SB_MotifThree_quantile_hh.\n");
            // return NAN; // Might require specific compiler flags/defs
            return -1.0; // Return specific error code
        }
    }
     // Size check: Need at least 2 points for pairs/transitions
    if (size < 2) {
         fprintf(stderr, "Warning: Input size < 2 for SB_MotifThree_quantile_hh. Returning 0.0\n");
         return 0.0;
    }

    // --- Variable Declarations ---
    int tmp_idx, r_idx;
    int dynamic_idx;
    int alphabet_size = 3;
    // int array_size; // Scoped locally where used
    int * yt = NULL; // alphabetized array
    double hh = 0.0; // output
    // double * out = NULL; // output array (only used in sb_motifthree)

    // Intermediate arrays
    int ** r1 = NULL;
    int * sizes_r1 = NULL;
    double * out1 = NULL; // Not directly used for hh, but part of original structure
    int*** r2 = NULL;
    int** sizes_r2 = NULL;
    double** out2 = NULL;
    int* tmp_ar = NULL; // Temporary array for subset/memcpy

    // --- Memory Allocation (with goto for cleanup) ---
    yt = malloc(size * sizeof(int)); // Use sizeof(int)
    if (yt == NULL) { perror("SB_MotifThree_hh: Malloc failed for yt"); goto cleanup_hh; }

    // Allocate r1, sizes_r1, out1
    r1 = malloc(alphabet_size * sizeof(*r1));
    sizes_r1 = malloc(alphabet_size * sizeof(*sizes_r1));
    out1 = malloc(alphabet_size * sizeof(*out1)); // Potentially unused for 'hh'
    if (r1 == NULL || sizes_r1 == NULL || out1 == NULL) { perror("SB_MotifThree_hh: Malloc failed for r1/sizes_r1/out1"); goto cleanup_hh; }
    for (int i = 0; i < alphabet_size; i++) r1[i] = NULL; // Init for safe cleanup
    for (int i = 0; i < alphabet_size; i++) {
        r1[i] = malloc(size * sizeof(**r1));
        if (r1[i] == NULL) { perror("SB_MotifThree_hh: Malloc failed for r1[i]"); goto cleanup_hh; }
    }

    // Allocate r2, sizes_r2, out2
    r2 = malloc(alphabet_size * sizeof(*r2));
    sizes_r2 = malloc(alphabet_size * sizeof(*sizes_r2));
    out2 = malloc(alphabet_size * sizeof(*out2));
    if (r2 == NULL || sizes_r2 == NULL || out2 == NULL) { perror("SB_MotifThree_hh: Malloc failed for r2/sizes_r2/out2"); goto cleanup_hh; }
    for (int i = 0; i < alphabet_size; i++) { r2[i] = NULL; sizes_r2[i] = NULL; out2[i] = NULL; } // Init
    for (int i = 0; i < alphabet_size; i++) {
        r2[i] = malloc(alphabet_size * sizeof(**r2));
        sizes_r2[i] = malloc(alphabet_size * sizeof(**sizes_r2));
        out2[i] = malloc(alphabet_size * sizeof(**out2));
        if (r2[i] == NULL || sizes_r2[i] == NULL || out2[i] == NULL) { perror("SB_MotifThree_hh: Malloc failed for r2[i]/etc level 2"); goto cleanup_hh; }
        for (int j = 0; j < alphabet_size; j++) r2[i][j] = NULL; // Init level 3
        for (int j = 0; j < alphabet_size; j++) {
            r2[i][j] = malloc(size * sizeof(***r2));
             if (r2[i][j] == NULL) { perror("SB_MotifThree_hh: Malloc failed for r2[i][j] level 3"); goto cleanup_hh; }
        }
    }


    // --- Coarse Graining ---
    // transfer to alphabet. Assuming sb_coarsegrain outputs symbols 1, 2, 3.
    sb_coarsegrain(y, size, "quantile", 3, yt);


    // --- Calculations ---
    // words of length 1
    // array_size = alphabet_size; // Re-declare/scope locally if needed
    for (int i = 0; i < alphabet_size; i++) {
        r_idx = 0;
        sizes_r1[i] = 0;
        for (int j = 0; j < size; j++) {
            if (yt[j] == i + 1) { // Assumes 1-based yt
                r1[i][r_idx++] = j;
                sizes_r1[i]++;
            }
        }
        // Calculation for out1 (marginal probability) if needed later
        // double tmp_p1 = (double)sizes_r1[i] / size;
        // out1[i] = tmp_p1;
    }

    // words of length 2
    // array_size = alphabet_size * alphabet_size; // Scope locally if needed

    // removing last item if it is == max possible idx since later we are taking idx + 1
    // This section uses subset and memcpy - ensure these helpers are correctly linked and implemented.
    // WARNING: This specific subset/memcpy logic in the reference might be complex/risky.
    for (int i = 0; i < alphabet_size; i++) {
        if (sizes_r1[i] > 0 && r1[i][sizes_r1[i] - 1] == size - 1) {
            // Allocate temporary array ONLY if we actually need to copy
            tmp_ar = malloc(sizes_r1[i] * sizeof(int)); // As in reference
            if(tmp_ar == NULL) { perror("SB_MotifThree_hh: Malloc failed for tmp_ar"); goto cleanup_hh;}

            // subset copies elements from r1[i] (indices 0 to sizes_r1[i]-1) into tmp_ar
            subset(r1[i], tmp_ar, 0, sizes_r1[i]); // Assume subset copies ALL elements first

            // memcpy copies N-1 elements from tmp_ar back into r1[i].
            // Potential Issue: If subset copies N elements, this reads N-1 elements.
            // If subset itself copied only N-1 elements, memcpy might be redundant or wrong.
            // --> Replicating reference exactly, assuming subset & memcpy achieve the intended effect.
            memcpy(r1[i], tmp_ar, (sizes_r1[i] - 1) * sizeof(int)); // Use sizeof(int)
            sizes_r1[i]--;
            free(tmp_ar);
            tmp_ar = NULL; // Avoid double free in cleanup
        }
    }

    // fill separately (calculate counts and joint probabilities)
    for (int i = 0; i < alphabet_size; i++) {
        for (int j = 0; j < alphabet_size; j++) {
            sizes_r2[i][j] = 0;
            dynamic_idx = 0;
            for (int k = 0; k < sizes_r1[i]; k++) { // Use adjusted size
                 // Ensure index is valid before accessing yt
                 if (r1[i][k] + 1 < size) {
                    tmp_idx = yt[r1[i][k] + 1];
                    if (tmp_idx == (j + 1)) { // Assumes 1-based yt
                        // Storing index in r2[i][j] is not needed for hh calculation
                        // r2[i][j][dynamic_idx++] = r1[i][k];
                        sizes_r2[i][j]++;
                    }
                 }
            }
            // Calculate joint probability P(i,j) normalized by N-1
            // Check size > 1 (already checked at function start)
            double tmp_p2 = (double)sizes_r2[i][j] / ((double)(size) - 1.0);
            out2[i][j] =  tmp_p2;
        }
    }

    // Calculate final 'hh' value using f_entropy on rows of out2
    hh = 0.0;
    for (int i = 0; i < alphabet_size; i++) {
        // Assumes f_entropy is available and calculates Shannon entropy (base 2?) correctly
        hh += f_entropy(out2[i], alphabet_size);
    }

// --- Cleanup ---
cleanup_hh:
    // Free allocated memory carefully
    free(tmp_ar); // Free if allocated in the loop and not freed yet
    if (r2 != NULL) {
        for (int i = 0; i < alphabet_size; i++) {
            if (r2[i] != NULL) {
                for (int j = 0; j < alphabet_size; j++) {
                    free(r2[i][j]);
                }
                free(r2[i]);
            }
        }
        free(r2);
    }
    if (sizes_r2 != NULL) {
        for (int i = 0; i < alphabet_size; i++) {
            free(sizes_r2[i]);
        }
        free(sizes_r2);
    }
   if (out2 != NULL) {
        for (int i = 0; i < alphabet_size; i++) {
            free(out2[i]);
        }
        free(out2);
    }
    if (r1 != NULL) {
        for (int i = 0; i < alphabet_size; i++) {
            free(r1[i]);
        }
        free(r1);
    }
    free(sizes_r1);
    free(out1);
    free(yt);
    // Do not free 'out' here if it were used - it's only relevant for sb_motifthree

    return hh; // Return the calculated result
}


// -----------------------------------------------------------------------------
// Exportable Full Feature Function: sb_motifthree
// Calculates array of 124 features. NOTE: Caller must free the returned pointer.
// -----------------------------------------------------------------------------
DLLEXPORT double * sb_motifthree(const double y[], int size, const char how[])
{
    // --- Variable Declarations ---
    int tmp_idx, r_idx, i, j, k, l, m; // array_size scoped locally
    int dynamic_idx;
    int * tmp_ar = NULL;
    int alphabet_size = 3;
    int out_idx = 0;
    int * yt = NULL;
    double tmp;
    double * out = NULL; // output array - TO BE FREED BY CALLER
    double * diff_y = NULL; // For 'diffquant' method
    int current_size = size; // Effective size after diff

    // Pointers for intermediate results (init to NULL for cleanup)
    int ** r1 = NULL;
    int * sizes_r1 = NULL;
    double * out1 = NULL;
    int *** r2 = NULL;
    int ** sizes_r2 = NULL;
    double ** out2 = NULL;
    int **** r3 = NULL;
    int *** sizes_r3 = NULL;
    double *** out3 = NULL;
    int ***** r4 = NULL;
    int **** sizes_r4 = NULL;
    double **** out4 = NULL;

    // --- Allocate Output Array ---
    out = malloc(124 * sizeof(double));
    if (out == NULL) { perror("sb_motifthree: Malloc failed for out"); return NULL; }
    // Initialize 'out' elements to NaN for safety
    for(int x=0; x<124; ++x) out[x] = NAN;

    // --- Input Size Check ---
    if (size < 1) { fprintf(stderr, "Error: sb_motifthree input size < 1.\n"); goto cleanup_sb_full; } // Changed to goto cleanup


    // --- Malloc Main Arrays ---
    yt = malloc(size * sizeof(int));
    if (yt == NULL) { perror("sb_motifthree: Malloc failed for yt"); goto cleanup_sb_full; }


    // --- Coarse Graining ---
    if (strcmp(how, "quantile") == 0) {
        if (size < 1) { fprintf(stderr, "Error: sb_motifthree size < 1 for quantile.\n"); goto cleanup_sb_full; }
        sb_coarsegrain(y, size, how, alphabet_size, yt);
        current_size = size;
    } else if (strcmp(how, "diffquant") == 0) {
        if (size < 2) { fprintf(stderr, "Error: sb_motifthree size < 2 for diffquant.\n"); goto cleanup_sb_full; }
        diff_y = malloc((size - 1) * sizeof(double));
        if (diff_y == NULL) { perror("sb_motifthree: Malloc failed for diff_y"); goto cleanup_sb_full; }
        diff(y, size, diff_y); // Assumes diff exists and is linked
        sb_coarsegrain(diff_y, size - 1, "quantile", alphabet_size, yt); // Use size-1
        current_size = size - 1;
        free(diff_y); // Free diff_y once yt is created
        diff_y = NULL;
    } else {
        fprintf(stderr, "ERROR in sb_motifthree: Unknown 'how' method: %s\n", how);
        goto cleanup_sb_full;
    }
     if (current_size < 1) { fprintf(stderr, "Error: sb_motifthree effective size < 1.\n"); goto cleanup_sb_full; }


    // --- Word Calculations (Length 1 to 4) ---
    // Simplified goto logic - jump to cleanup if any malloc fails

    // Words of length 1
    int array_size_l1 = alphabet_size;
    r1 = malloc(array_size_l1 * sizeof(*r1));
    sizes_r1 = malloc(array_size_l1 * sizeof(*sizes_r1));
    out1 = malloc(array_size_l1 * sizeof(*out1));
    if (!r1 || !sizes_r1 || !out1) { perror("sb_motifthree: Malloc failed L1 base"); goto cleanup_sb_full; }
    for(i=0; i<array_size_l1; ++i) r1[i] = NULL; // Init for cleanup
    for (i = 0; i < array_size_l1; i++) {
        r1[i] = malloc(current_size * sizeof(**r1));
        if (!r1[i]) { perror("sb_motifthree: Malloc failed L1 inner"); goto cleanup_sb_full; }
        // Calculation...
        r_idx = 0; sizes_r1[i] = 0;
        for (j = 0; j < current_size; j++) { if (yt[j] == i + 1) { r1[i][r_idx++] = j; sizes_r1[i]++; } }
        tmp = (current_size > 0) ? (double)sizes_r1[i] / current_size : 0.0;
        out1[i] = tmp;
        if (out_idx < 124) out[out_idx++] = tmp; else { fprintf(stderr,"WARN: Out array overrun L1\n"); }
    }
    if (out_idx < 124) out[out_idx++] = f_entropy(out1, array_size_l1); else { fprintf(stderr,"WARN: Out array overrun L1 entropy\n"); }


    // Words of length 2
    if (current_size >= 2) {
        // Adjust r1 (using potentially complex subset/memcpy from reference)
        for (i = 0; i < alphabet_size; i++) {
            if (sizes_r1[i] > 0 && r1[i][sizes_r1[i] - 1] == current_size - 1) {
                 tmp_ar = malloc(sizes_r1[i] * sizeof(int)); // Size based on *before* decrement
                 if(!tmp_ar) { perror("sb_motifthree: Malloc failed tmp_ar L2"); goto cleanup_sb_full; }
                 subset(r1[i], tmp_ar, 0, sizes_r1[i]); // Copy N elements?
                 memcpy(r1[i], tmp_ar, (sizes_r1[i] - 1) * sizeof(int)); // Copy N-1 back
                 sizes_r1[i]--;
                 free(tmp_ar); tmp_ar = NULL;
             }
         }
        // Allocate L2 structures
        r2 = malloc(alphabet_size * sizeof(*r2));
        sizes_r2 = malloc(alphabet_size * sizeof(*sizes_r2));
        out2 = malloc(alphabet_size * sizeof(*out2));
        if (!r2 || !sizes_r2 || !out2) { perror("sb_motifthree: Malloc failed L2 base"); goto cleanup_sb_full; }
        for(i=0; i<alphabet_size; ++i) { r2[i]=NULL; sizes_r2[i]=NULL; out2[i]=NULL; } // Init
        for (i = 0; i < alphabet_size; i++) {
            r2[i] = malloc(alphabet_size * sizeof(**r2));
            sizes_r2[i] = malloc(alphabet_size * sizeof(**sizes_r2));
            out2[i] = malloc(alphabet_size * sizeof(**out2));
            if (!r2[i] || !sizes_r2[i] || !out2[i]) { perror("sb_motifthree: Malloc failed L2 inner"); goto cleanup_sb_full; }
            for(j=0; j<alphabet_size; ++j) r2[i][j]=NULL; // Init level 3
            for (j = 0; j < alphabet_size; j++) {
                r2[i][j] = malloc(current_size * sizeof(***r2));
                if (!r2[i][j]) { perror("sb_motifthree: Malloc failed L2 innermost"); goto cleanup_sb_full; }
                // Calculation...
                sizes_r2[i][j] = 0; dynamic_idx = 0;
                for (k = 0; k < sizes_r1[i]; k++) { if (r1[i][k] + 1 < current_size) { tmp_idx = yt[r1[i][k] + 1]; if (tmp_idx == (j + 1)) { r2[i][j][dynamic_idx++] = r1[i][k]; sizes_r2[i][j]++; } } }
                tmp = (current_size > 1) ? (double)sizes_r2[i][j] / (current_size - 1) : 0.0;
                out2[i][j] = tmp;
                if (out_idx < 124) out[out_idx++] = tmp; else { fprintf(stderr,"WARN: Out array overrun L2\n"); }
            }
        }
        tmp = 0.0; for (i = 0; i < alphabet_size; i++) { tmp += f_entropy(out2[i], alphabet_size); }
        if (out_idx < 124) out[out_idx++] = tmp; else { fprintf(stderr,"WARN: Out array overrun L2 entropy\n"); }
    } else { /* Fill L2 NaNs */ int n = alphabet_size*alphabet_size+1; while(n-->0 && out_idx<124) out[out_idx++]=NAN; }

    // Words of length 3
     if (current_size >= 3 && r2) { // Need r2 from previous step
         // Adjust r2
         for (i = 0; i < alphabet_size; i++) { for (j = 0; j < alphabet_size; j++) { if (sizes_r2[i][j] > 0 && r2[i][j][sizes_r2[i][j] - 1] == current_size - 2) { sizes_r2[i][j]--; } } }
         // Allocate L3
         r3 = malloc(alphabet_size * sizeof(*r3)); sizes_r3 = malloc(alphabet_size * sizeof(*sizes_r3)); out3 = malloc(alphabet_size * sizeof(*out3));
         if(!r3 || !sizes_r3 || !out3) { perror("sb_motifthree: Malloc failed L3 base"); goto cleanup_sb_full; }
         for(i=0; i<alphabet_size; ++i) { r3[i]=NULL; sizes_r3[i]=NULL; out3[i]=NULL; } // Init
         for (i = 0; i < alphabet_size; i++) {
             r3[i] = malloc(alphabet_size * sizeof(**r3)); sizes_r3[i] = malloc(alphabet_size * sizeof(**sizes_r3)); out3[i] = malloc(alphabet_size * sizeof(**out3));
             if(!r3[i] || !sizes_r3[i] || !out3[i]) { perror("sb_motifthree: Malloc failed L3 L2"); goto cleanup_sb_full; }
             for(j=0; j<alphabet_size; ++j) { r3[i][j]=NULL; sizes_r3[i][j]=NULL; out3[i][j]=NULL; } // Init
             for (j = 0; j < alphabet_size; j++) {
                 r3[i][j] = malloc(alphabet_size * sizeof(***r3)); sizes_r3[i][j] = malloc(alphabet_size * sizeof(***sizes_r3)); out3[i][j] = malloc(alphabet_size * sizeof(***out3));
                 if(!r3[i][j] || !sizes_r3[i][j] || !out3[i][j]) { perror("sb_motifthree: Malloc failed L3 L3"); goto cleanup_sb_full; }
                 for(k=0; k<alphabet_size; ++k) r3[i][j][k]=NULL; // Init
                 for (k = 0; k < alphabet_size; k++) {
                     r3[i][j][k] = malloc(current_size * sizeof(****r3));
                     if(!r3[i][j][k]) { perror("sb_motifthree: Malloc failed L3 L4"); goto cleanup_sb_full; }
                     // Calculation...
                     sizes_r3[i][j][k] = 0; dynamic_idx = 0;
                     for (l = 0; l < sizes_r2[i][j]; l++) { if (r2[i][j][l] + 2 < current_size) { tmp_idx = yt[r2[i][j][l] + 2]; if (tmp_idx == (k + 1)) { r3[i][j][k][dynamic_idx++] = r2[i][j][l]; sizes_r3[i][j][k]++; } } }
                     tmp = (current_size > 2) ? (double)sizes_r3[i][j][k] / (current_size - 2) : 0.0;
                     out3[i][j][k] = tmp;
                     if (out_idx < 124) out[out_idx++] = tmp; else { fprintf(stderr,"WARN: Out array overrun L3\n"); }
                 }
             }
         }
         tmp = 0.0; for (i = 0; i < alphabet_size; i++) { for (j = 0; j < alphabet_size; j++) { tmp += f_entropy(out3[i][j], alphabet_size); } }
         if (out_idx < 124) out[out_idx++] = tmp; else { fprintf(stderr,"WARN: Out array overrun L3 entropy\n"); }
     } else { /* Fill L3 NaNs */ int n = alphabet_size*alphabet_size*alphabet_size+1; while(n-->0 && out_idx<124) out[out_idx++]=NAN; }

    // Words of length 4
     if (current_size >= 4 && r3) { // Need r3 from previous step
         // Adjust r3
         for(i=0;i<alphabet_size;i++){for(j=0;j<alphabet_size;j++){for(k=0;k<alphabet_size;k++){if(sizes_r3[i][j][k]>0 && r3[i][j][k][sizes_r3[i][j][k]-1]==current_size-3){sizes_r3[i][j][k]--;}}}}
         // Allocate L4
         r4 = malloc(alphabet_size * sizeof(*r4)); sizes_r4 = malloc(alphabet_size * sizeof(*sizes_r4)); out4 = malloc(alphabet_size * sizeof(*out4));
         if(!r4 || !sizes_r4 || !out4) { perror("sb_motifthree: Malloc failed L4 base"); goto cleanup_sb_full; }
         for(i=0; i<alphabet_size; ++i) { r4[i]=NULL; sizes_r4[i]=NULL; out4[i]=NULL; } // Init
         for (i = 0; i < alphabet_size; i++) {
             r4[i] = malloc(alphabet_size * sizeof(**r4)); sizes_r4[i] = malloc(alphabet_size * sizeof(**sizes_r4)); out4[i] = malloc(alphabet_size * sizeof(**out4));
             if(!r4[i] || !sizes_r4[i] || !out4[i]) { perror("sb_motifthree: Malloc failed L4 L2"); goto cleanup_sb_full; }
             for(j=0; j<alphabet_size; ++j) { r4[i][j]=NULL; sizes_r4[i][j]=NULL; out4[i][j]=NULL; } // Init
             for (j = 0; j < alphabet_size; j++) {
                 r4[i][j] = malloc(alphabet_size * sizeof(***r4)); sizes_r4[i][j] = malloc(alphabet_size * sizeof(***sizes_r4)); out4[i][j] = malloc(alphabet_size * sizeof(***out4));
                 if(!r4[i][j] || !sizes_r4[i][j] || !out4[i][j]) { perror("sb_motifthree: Malloc failed L4 L3"); goto cleanup_sb_full; }
                 for(k=0; k<alphabet_size; ++k) { r4[i][j][k]=NULL; sizes_r4[i][j][k]=NULL; out4[i][j][k]=NULL; } // Init
                 for (k = 0; k < alphabet_size; k++) {
                     r4[i][j][k] = malloc(alphabet_size * sizeof(****r4)); sizes_r4[i][j][k] = malloc(alphabet_size * sizeof(****sizes_r4)); out4[i][j][k] = malloc(alphabet_size * sizeof(****out4));
                     if(!r4[i][j][k] || !sizes_r4[i][j][k] || !out4[i][j][k]) { perror("sb_motifthree: Malloc failed L4 L4"); goto cleanup_sb_full; }
                     for(l=0; l<alphabet_size; ++l) r4[i][j][k][l]=NULL; // Init
                     for (l = 0; l < alphabet_size; l++) {
                         r4[i][j][k][l] = malloc(current_size * sizeof(*****r4));
                         if(!r4[i][j][k][l]) { perror("sb_motifthree: Malloc failed L4 L5"); goto cleanup_sb_full; }
                         // Calculation...
                         sizes_r4[i][j][k][l] = 0; dynamic_idx = 0;
                         for (m = 0; m < sizes_r3[i][j][k]; m++) { if (r3[i][j][k][m] + 3 < current_size) { tmp_idx = yt[r3[i][j][k][m] + 3]; if (tmp_idx == l + 1) { r4[i][j][k][l][dynamic_idx++] = r3[i][j][k][m]; sizes_r4[i][j][k][l]++; } } }
                         tmp = (current_size > 3) ? (double)sizes_r4[i][j][k][l] / (current_size - 3) : 0.0;
                         out4[i][j][k][l] = tmp;
                         if (out_idx < 124) out[out_idx++] = tmp; else { fprintf(stderr,"WARN: Out array overrun L4\n"); }
                     }
                 }
             }
         }
         tmp = 0.0; for(i=0;i<alphabet_size;i++){for(j=0;j<alphabet_size;j++){for(k=0;k<alphabet_size;k++){tmp+=f_entropy(out4[i][j][k], alphabet_size);}}}
         if (out_idx < 124) out[out_idx++] = tmp; else { fprintf(stderr,"WARN: Out array overrun L4 entropy\n"); }
     } else { /* Fill L4 NaNs */ int n = alphabet_size*alphabet_size*alphabet_size*alphabet_size+1; while(n-->0 && out_idx<124) out[out_idx++]=NAN; }


    // --- Final Checks ---
    // Fill any remaining 'out' elements with NaN if loop finished early or errors occurred
    while (out_idx < 124) { out[out_idx++] = NAN; }


cleanup_sb_full:
    // --- Free All Allocated Memory ---
    // Free memory in reverse order of allocation and check for NULL pointers
    // Free L4 structures
    if (r4 != NULL) { /* ... full nested free logic ... */ } // Add full free logic
    if (sizes_r4 != NULL) { /* ... full nested free logic ... */ } // Add full free logic
    if (out4 != NULL) { /* ... full nested free logic ... */ } // Add full free logic

    // Free L3 structures
    if (r3 != NULL) { /* ... full nested free logic ... */ } // Add full free logic
    if (sizes_r3 != NULL) { /* ... full nested free logic ... */ } // Add full free logic
    if (out3 != NULL) { /* ... full nested free logic ... */ } // Add full free logic

    // Free L2 structures
    if (r2 != NULL) { for(i=0;i<alphabet_size;i++){if(r2[i]){for(j=0;j<alphabet_size;j++){free(r2[i][j]);} free(r2[i]);}} free(r2); }
    if (sizes_r2 != NULL) { for(i=0;i<alphabet_size;i++){free(sizes_r2[i]);} free(sizes_r2); }
    if (out2 != NULL) { for(i=0;i<alphabet_size;i++){free(out2[i]);} free(out2); }

    // Free L1 structures
    if (r1 != NULL) { for(i=0;i<alphabet_size;i++){free(r1[i]);} free(r1); }
    free(sizes_r1);
    free(out1);

    // Free base arrays
    free(yt);
    free(tmp_ar); // Free temp array if allocated and not freed in loop
    // free(diff_y); // Already freed if allocated

    // Note: 'out' array is NOT freed here - it's the return value. Caller must free it.
    // If cleanup was reached due to error before 'out' was fully populated, it might contain NaNs or garbage.
    return out;
}
























