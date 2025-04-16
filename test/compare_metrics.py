import pandas as pd
import numpy as np
import time
import sys
import json
import matplotlib.pyplot as plt 

CUSTOM_SCRIPT_MODULE_NAME = "easytest_metrics"
CUSTOM_FUNCTION_NAME = "test_easytest_metrics" # Assuming this is correct now

LIBRARY_SCRIPT_MODULE_NAME = "original_metrics"
LIBRARY_FUNCTION_NAME = "test_original_metrics"

print("--- Importing Analysis Functions (Will crash on error) ---")


custom_module = __import__(CUSTOM_SCRIPT_MODULE_NAME)
custom_analysis_func = getattr(custom_module, CUSTOM_FUNCTION_NAME) 
print(f"Successfully imported: {CUSTOM_SCRIPT_MODULE_NAME}.{CUSTOM_FUNCTION_NAME}")

library_module = __import__(LIBRARY_SCRIPT_MODULE_NAME)
library_analysis_func = getattr(library_module, LIBRARY_FUNCTION_NAME)
print(f"Successfully imported: {LIBRARY_SCRIPT_MODULE_NAME}.{LIBRARY_FUNCTION_NAME}")

print("--- End Imports ---")


def compare_results():
    """
    Runs both analyses and compares extreme values.
    Displays the comparison table as a matplotlib image.
    """

    custom_results = None
    library_results = None

    # --- Run Custom Analysis (Will crash if function fails) ---
    print("\n--- Running Analysis with Custom Metrics (NO error handling) ---")
    start_time = time.time()
    custom_results = custom_analysis_func() # Direct call
    end_time = time.time()
    print(f"--- Custom Analysis Duration: {end_time - start_time:.2f} seconds ---")

    # --- Run Library Analysis (Will crash if function fails) ---
    print("\n--- Running Analysis with Library Functions (NO error handling) ---")
    start_time = time.time()
    library_results = library_analysis_func() # Direct call
    end_time = time.time()
    print(f"--- Library Analysis Duration: {end_time - start_time:.2f} seconds ---")

    # --- Comparison (Will crash if results are None or unexpected format) ---
    if custom_results is None or library_results is None:
         print("\nOne or both analyses failed to produce results dictionary.")
         return

    print("\n--- Preparing Comparison Table ---") # Changed print message

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

        comparison_data.append({
            'Feature': feature,
            'Custom Lowest ID': custom_low_id,
            'Custom Lowest Value': custom_low_val,
            'Custom Highest ID': custom_high_id,
            'Custom Highest Value': custom_high_val,
            'Library Lowest ID': library_low_id,
            'Library Lowest Value': library_low_val,
            'Library Highest ID': library_high_id,
            'Library Highest Value': library_high_val,
        })

    comparison_df = pd.DataFrame(comparison_data)
    comparison_df.set_index('Feature', inplace=True)

    print("--- Displaying Comparison Table as Image ---")

    # Prepare data for table - format numbers and handle NaNs
    df_display = comparison_df.copy()
    float_format = "{:.4f}".format
    na_rep = 'N/A'

    for col in df_display.columns:
        if 'Value' in col:
            # Format numeric columns, converting NaN to string representation
             df_display[col] = df_display[col].apply(lambda x: float_format(x) if pd.notna(x) else na_rep)
        else:
             # Format ID columns (handle None or other types)
             df_display[col] = df_display[col].apply(lambda x: str(x) if pd.notna(x) else na_rep)

    cell_text = df_display.values.tolist()
    col_labels = df_display.columns.tolist()
    row_labels = df_display.index.tolist()

    # Create figure - adjust size dynamically (especially height)
    # Increase width factor if columns are wide
    fig_width = 14
    fig_height = max(3, len(row_labels) * 0.5 + 1) # Base height + per row
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    ax.axis('tight')
    ax.axis('off')

    # Create table
    the_table = ax.table(cellText=cell_text,
                         colLabels=col_labels,
                         rowLabels=row_labels,
                         loc='center',
                         cellLoc='left', # Align text left within cells
                         rowLoc='right') # Align row labels right

    # Adjust font size if needed
    the_table.auto_set_font_size(False)
    the_table.set_fontsize(10)

    # Adjust column widths automatically
    the_table.auto_set_column_width(col=list(range(len(col_labels))))

    plt.title('Comparison of Extreme Values (Custom vs. Library)', fontsize=14, y=1.05) # Adjust title position
    fig.tight_layout() # Adjust layout
    plt.show() # Display the plot window
    # --- *** END OF CODE CHANGE *** ---


class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer): return int(obj)
        if isinstance(obj, np.floating): return float(obj)
        if isinstance(obj, np.ndarray): return obj.tolist()
        if np.isnan(obj): return None
        return super(NpEncoder, self).default(obj)

if __name__ == "__main__":
    # Direct function call - will crash on any unhandled error inside
    compare_results() # Call the main comparison function

    # Code below only runs if *nothing* crashed
    print("\nComparison script finished.")