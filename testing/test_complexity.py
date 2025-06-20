import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datasetsforecast.m3 import M3
from easytime import complexity
import sys
import os
import random

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
    print(f"Added to sys.path: {PROJECT_ROOT}")


def test_complexity_analysis():
    """
    Analyzes 10 randomly concatenated M3 monthly series using the Metrics class,
    focusing on the complexity estimate, and plots each original and concatenated series.
    Returns the DataFrame of computed complexities.
    """

    data_dir = os.path.join(PROJECT_ROOT, 'data', 'm3_download')

    os.makedirs(data_dir, exist_ok=True)
    loaded_data = M3.load(directory=data_dir, group='Monthly') 

    Y_df = None
    if isinstance(loaded_data, (list, tuple)) and len(loaded_data) > 0:
        Y_df = loaded_data[0]
    elif isinstance(loaded_data, pd.DataFrame):
        Y_df = loaded_data

    if Y_df is None: 
        print("Failed to extract DataFrame from loaded data.")
        return None

    print("Data loaded successfully.")
    required_cols = ['unique_id', 'ds', 'y']
    if not all(col in Y_df.columns for col in required_cols):
        raise ValueError(f"Error: Missing required columns: {[c for c in required_cols if c not in Y_df.columns]}")
    
    all_m3_ids = Y_df['unique_id'].unique()
    if len(all_m3_ids) < 2:
        print("Not enough unique series in M3 data to form pairs.")
        return None

    results_data = []
    num_concatenated_series_to_test = 10
    min_len_for_complexity = 2

    print(f"\nAnalyzing complexity for {num_concatenated_series_to_test} randomly selected and concatenated monthly time series...")

    processed_pairs_count = 0
    attempts = 0
    max_attempts = num_concatenated_series_to_test * 5 

    while processed_pairs_count < num_concatenated_series_to_test and attempts < max_attempts:
        attempts += 1
        print(f"\n--- Preparing Test Case {processed_pairs_count + 1}/{num_concatenated_series_to_test} (Attempt {attempts}) ---")
        
        id1, id2 = random.sample(list(all_m3_ids), 2)
        
        series1_pd = Y_df[Y_df['unique_id'] == id1]['y'].reset_index(drop=True)
        series2_pd = Y_df[Y_df['unique_id'] == id2]['y'].reset_index(drop=True)

        if series1_pd.empty or series2_pd.empty:
            print(f"Could not load series {id1} or {id2}. Skipping this pair.")
            continue
            
        series1_np = series1_pd.to_numpy()
        series2_np = series2_pd.to_numpy()

        if len(series1_np) < min_len_for_complexity:
            print(f"Series {id1} is too short ({len(series1_np)} points). Skipping.")
            continue
        if len(series2_np) < min_len_for_complexity:
            print(f"Series {id2} is too short ({len(series2_np)} points). Skipping.")
            continue

        complexity_s1 = complexity(series1_np)
        print(f"Complexity for {id1}: {complexity_s1:.4f}")

        complexity_s2 = complexity(series2_np)
        print(f"Complexity for {id2}: {complexity_s2:.4f}")
        
        concatenated_series_np = np.concatenate([series1_np, series2_np])
        concatenation_point = len(series1_np)
        series_pair_id = f"{id1}_plus_{id2}"
        
        print(f"Concatenated: {id1} ({len(series1_np)}pts) + {id2} ({len(series2_np)}pts) = Total {len(concatenated_series_np)}pts. CP at {concatenation_point}")

        if len(concatenated_series_np) < min_len_for_complexity:
            print(f"Concatenated series {series_pair_id} is too short ({len(concatenated_series_np)} points) for complexity calculation. Skipping complexity for concatenated.")
            complexity_concat = np.nan
        else:
            complexity_concat = complexity(concatenated_series_np)
            print(f"Complexity for concatenated series {series_pair_id}: {complexity_concat:.4f}")

        current_features = {
            "unique_id_pair": series_pair_id,
            "ID1": id1,
            "Length1": len(series1_np),
            "Complexity_S1": complexity_s1,
            "ID2": id2,
            "Length2": len(series2_np),
            "Complexity_S2": complexity_s2,
            "Length_Concatenated": len(concatenated_series_np),
            "Complexity_Concatenated": complexity_concat,
            "Actual_Concatenation_Point": concatenation_point
        }
        results_data.append(current_features)
        
        processed_pairs_count += 1

    if processed_pairs_count < num_concatenated_series_to_test:
        print(f"Warning: Only processed {processed_pairs_count} pairs after {max_attempts} attempts due to length constraints.")

    print(f"\nAnalysis loop complete. Features computed for {len(results_data)} pairs.")

    if not results_data:
        print("No data was processed successfully.")
        return None

    results_df = pd.DataFrame(results_data)

    print("\n--- Calculation Summary for Series Complexities ---")
    if not results_df.empty:
        print(results_df.info()) 
        print("\n--- Sample of Results DataFrame (Complexity features) ---")
        columns_to_display = [
            "unique_id_pair", "Complexity_S1", "Complexity_S2", "Complexity_Concatenated"
        ]
        columns_to_display = [col for col in columns_to_display if col in results_df.columns]
        if columns_to_display:
             print(results_df[columns_to_display].head(num_concatenated_series_to_test))
        else:
            print("No complexity columns found in the results DataFrame to display.")
    else:
        print("Results DataFrame is empty.")

    return results_df

if __name__ == "__main__":
    plt.ioff() 
    
    all_metrics_df = test_complexity_analysis()

    print("\n--- Script Finished ---")
    if all_metrics_df is not None and not all_metrics_df.empty:
        print("Metrics DataFrame for series complexities was generated.")
        
        # --- Save results DataFrame as an image ---
        table_image_output_path = os.path.join(PROJECT_ROOT, "examples + more info", "series_complexity_results_table.png")
        
        cols_for_image_display = [
            "unique_id_pair", "ID1", "Length1", "Complexity_S1", 
            "ID2", "Length2", "Complexity_S2", 
            "Length_Concatenated", "Complexity_Concatenated"
        ]

        actual_cols_for_image = [col for col in cols_for_image_display if col in all_metrics_df.columns]
        
        if actual_cols_for_image:
            df_for_image = all_metrics_df[actual_cols_for_image].copy()

            complexity_cols_in_df = [col for col in ["Complexity_S1", "Complexity_S2", "Complexity_Concatenated"] if col in df_for_image.columns]
            for col in complexity_cols_in_df:
                df_for_image[col] = pd.to_numeric(df_for_image[col], errors='coerce').round(4)

            if not df_for_image.empty:
                num_rows = len(df_for_image)
                num_cols = len(df_for_image.columns)
                fig_width = max(10, num_cols * 1.8) 
                fig_height = max(3, num_rows * 0.3 + 1.5)

                fig, ax = plt.subplots(figsize=(fig_width, fig_height)) 
                ax.axis('tight')
                ax.axis('off')
                
                the_table = ax.table(cellText=df_for_image.values,
                                     colLabels=df_for_image.columns,
                                     cellLoc='center', 
                                     loc='center',
                                     colColours=['#DDDDDD']*len(df_for_image.columns))

                the_table.auto_set_font_size(False)
                the_table.set_fontsize(8)
                the_table.scale(1.1, 1.1)

                plt.title("Series Complexity Analysis Results", fontsize=12, y=1.05)
                fig.tight_layout(pad=1.0)

                plt.savefig(table_image_output_path, dpi=200, bbox_inches='tight')
                print(f"Results table saved as image to: {table_image_output_path}")
                plt.close(fig)
            else:
                print("DataFrame selected for image is empty or relevant columns are missing.")
        else:
            print("No suitable columns found in DataFrame to create a table image.")
            
    else:
        print("Analysis did not generate a DataFrame or failed.")