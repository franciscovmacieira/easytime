import pandas as pd
import numpy as np
import time
import json
import matplotlib.pyplot as plt
import os # Make sure os is imported
import pprint # For printing sys.path if needed
import sys

# --- Assume necessary imports and function definitions are done above ---
# Like: CUSTOM_SCRIPT_MODULE_NAME, CUSTOM_FUNCTION_NAME, etc.
# Like: custom_analysis_func, library_analysis_func definitions

# --- Imports (Copied from your snippet for completeness) ---
print("--- Importing Analysis Functions (Will crash on error) ---")

CUSTOM_SCRIPT_MODULE_NAME = "easytest_metrics"
CUSTOM_FUNCTION_NAME = "test_easytest_metrics"
LIBRARY_SCRIPT_MODULE_NAME = "original_metrics" # Make sure this file exists
LIBRARY_FUNCTION_NAME = "test_original_metrics" # Make sure this function exists

try:
    custom_module = __import__(CUSTOM_SCRIPT_MODULE_NAME)
    custom_analysis_func = getattr(custom_module, CUSTOM_FUNCTION_NAME)
    print(f"Successfully imported: {CUSTOM_SCRIPT_MODULE_NAME}.{CUSTOM_FUNCTION_NAME}")

    library_module = __import__(LIBRARY_SCRIPT_MODULE_NAME)
    library_analysis_func = getattr(library_module, LIBRARY_FUNCTION_NAME)
    print(f"Successfully imported: {LIBRARY_SCRIPT_MODULE_NAME}.{LIBRARY_FUNCTION_NAME}")
except ModuleNotFoundError as e:
     print(f"ERROR: Import failed ({e}). Check script names and sys.path.")
     print("Current sys.path:")
     pprint.pprint(sys.path)
     exit()
except AttributeError as e:
     print(f"ERROR: Import failed ({e}). Check function names within scripts.")
     exit()
except Exception as e:
     print(f"ERROR: An unexpected error occurred during import: {e}")
     exit()


print("--- End Imports ---")


def compare_results():
    """
    Runs both analyses, compares extreme values, displays the comparison
    table as a matplotlib image, AND saves it to the current directory.
    """

    custom_results = None
    library_results = None

    # --- Run Custom Analysis ---
    print("\n--- Running Analysis with Custom Metrics (NO error handling) ---")
    start_time = time.time()
    # Add basic check if function exists before calling
    if custom_analysis_func:
        custom_results = custom_analysis_func()
    else:
        print("ERROR: Custom analysis function not loaded.")
    end_time = time.time()
    print(f"--- Custom Analysis Duration: {end_time - start_time:.2f} seconds ---")


    # --- Run Library Analysis ---
    print("\n--- Running Analysis with Library Functions (NO error handling) ---")
    start_time = time.time()
    if library_analysis_func:
        library_results = library_analysis_func()
    else:
         print("ERROR: Library analysis function not loaded.")
    end_time = time.time()
    print(f"--- Library Analysis Duration: {end_time - start_time:.2f} seconds ---")

    # --- Comparison ---
    if custom_results is None or library_results is None:
         print("\nOne or both analyses failed to produce results dictionary. Cannot compare.")
         return # Stop comparison if results are missing

    print("\n--- Preparing Comparison Table ---")

    # ... (keep the comparison_data preparation logic exactly as before) ...
    custom_keys = set(custom_results.keys())
    library_keys = set(library_results.keys())
    all_features = sorted(list(custom_keys | library_keys))
    comparison_data = []
    for feature in all_features:
        custom_low = custom_results.get(feature, {}).get('lowest', (None, np.nan))
        custom_high = custom_results.get(feature, {}).get('highest', (None, np.nan))
        library_low = library_results.get(feature, {}).get('lowest', (None, np.nan))
        library_high = library_results.get(feature, {}).get('highest', (None, np.nan))
        custom_low_id, custom_low_val = custom_low if isinstance(custom_low, tuple) and len(custom_low) == 2 else (None, np.nan)
        custom_high_id, custom_high_val = custom_high if isinstance(custom_high, tuple) and len(custom_high) == 2 else (None, np.nan)
        library_low_id, library_low_val = library_low if isinstance(library_low, tuple) and len(library_low) == 2 else (None, np.nan)
        library_high_id, library_high_val = library_high if isinstance(library_high, tuple) and len(library_high) == 2 else (None, np.nan)
        comparison_data.append({ 'Feature': feature, 'Custom Lowest ID': custom_low_id, 'Custom Lowest Value': custom_low_val, 'Custom Highest ID': custom_high_id, 'Custom Highest Value': custom_high_val, 'Library Lowest ID': library_low_id, 'Library Lowest Value': library_low_val, 'Library Highest ID': library_high_id, 'Library Highest Value': library_high_val })
    comparison_df = pd.DataFrame(comparison_data)
    if 'Feature' in comparison_df.columns: comparison_df.set_index('Feature', inplace=True)


    print("--- Generating Comparison Table Image ---")

    # --- Prepare data for display ---
    df_display = comparison_df.copy()
    float_format = "{:.4f}".format; na_rep = 'N/A'
    for col in df_display.columns:
        if 'Value' in col: df_display[col] = df_display[col].apply(lambda x: float_format(x) if pd.notna(x) else na_rep)
        elif 'ID' in col : df_display[col] = df_display[col].apply(lambda x: str(x) if pd.notna(x) else na_rep)
    cell_text = df_display.values.tolist() if not df_display.empty else []
    col_labels = df_display.columns.tolist(); row_labels = df_display.index.tolist() if not df_display.empty else []

    # --- Create Figure ---
    base_width = 14; col_width_factor = 1.8; fig_width = max(base_width, len(col_labels) * col_width_factor); fig_height = max(4, len(row_labels) * 0.5 + 1.5)
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    ax.axis('tight'); ax.axis('off')

    # --- Create Table ---
    if cell_text:
         the_table = ax.table(cellText=cell_text, colLabels=col_labels, rowLabels=row_labels, loc='center', cellLoc='left', rowLoc='right')
         the_table.auto_set_font_size(False); the_table.set_fontsize(10)
         the_table.auto_set_column_width(col=list(range(len(col_labels))))
    else: ax.text(0.5, 0.5, "No comparison data available.", ha='center', va='center')

    # --- Set Title and Layout ---
    plt.title('Comparison of Extreme Values (Custom vs. Library)', fontsize=14, y=1.02)
    fig.tight_layout() # Adjust layout BEFORE saving is often better

    plt.title('Comparison of Extreme Values (Custom vs. Library)', fontsize=14, y=1.02) # Adjust title Y pos
    fig.tight_layout() # Adjust layout BEFORE saving is often better

    # --- *** UPDATED CODE FOR SAVING *** ---
    # Define filename
    output_filename = "comparison_table.png"

    try:
        # Determine the directory containing the script file itself
        script_directory = os.path.dirname(os.path.abspath(__file__))
        # Create the full path to save the file in that directory
        save_path = os.path.join(script_directory, output_filename)

        print(f"\nAttempting to save table image to script directory: {save_path}")
        # Save BEFORE showing
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"--- Save command executed. Checking file existence... ---")

        # Immediately check if the file exists after saving
        if os.path.exists(save_path):
             print(f"--- SUCCESS: Comparison table saved to: {save_path} ---")
        else:
             # This case suggests savefig completed without error but the file isn't there.
             # Could be permissions, a quirk in the backend, or filesystem delay.
             print(f"--- WARNING: File NOT found at {save_path} immediately after save command. Please verify manually. ---")

    except NameError:
        # This happens if __file__ is not defined (e.g., running interactively)
        print("\n--- WARNING: Cannot determine script directory automatically (likely running interactively). ---")
        print(f"--- Image '{output_filename}' was NOT saved automatically. ---")
        print(f"--- You may need to save it manually from the plot window or run this code as a script. ---")
    except Exception as e:
        # Catch other potential saving errors (permissions, invalid path chars etc.)
        print(f"--- ERROR: Failed to save comparison table image: {e} ---")

    plt.show() # Display the plot window AFTER saving attempt or warning


# NpEncoder class definition (keep as is)
class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer): return int(obj)
        if isinstance(obj, np.floating): return float(obj)
        if isinstance(obj, np.ndarray): return obj.tolist()
        if isinstance(obj, float) and np.isnan(obj): return None
        return super(NpEncoder, self).default(obj)

if __name__ == "__main__":
    compare_results()
    print("\nComparison script finished.")