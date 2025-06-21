import numpy as np
import pandas as pd
import os
import sys
from datasetsforecast.m3 import M3
from ete_ts import (
    trend_strength,
    trend_changes,
    linear_regression_slope,
    linear_regression_r2,
    forecastability,
    fluctuation,
    ac_relevance,
    seasonal_strength,
    window_fluctuation,
    st_variation,
    diff_series,
    complexity,
    rec_concentration,
    centroid,
    info
)

# --- Path Setup ---
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
    print(f"Added to sys.path: {PROJECT_ROOT}")

def analyze_feature_correlations():
    """
    Loads the M3 monthly dataset, calculates a suite of time series features for each
    series, computes the correlation matrix for these features, ranks the
    correlations, and saves the ranked list to a text file.
    """

    # --- 1. Load Data and Calculate Features ---
    data_dir = os.path.join(PROJECT_ROOT, 'data', 'm3_download')
    print(f"\nLoading M3 Monthly data (will download to '{data_dir}' if needed)...")
    os.makedirs(data_dir, exist_ok=True)

    loaded_data = M3.load(directory=data_dir, group='Monthly')
    Y_df = loaded_data[0] if isinstance(loaded_data, (list, tuple)) and len(loaded_data) > 0 else loaded_data

    if Y_df is None:
        raise ValueError("Failed to extract DataFrame from loaded data.")

    print("Data loaded successfully.")
    Y_df['ds'] = pd.to_datetime(Y_df['ds'])

    all_dataset_ids = Y_df['unique_id'].unique()
    results_data = []

    print(f"\nAnalyzing {len(all_dataset_ids)} monthly time series to generate features...")

    processed_count = 0
    skipped_count = 0

    for i, dataset_id in enumerate(all_dataset_ids):
        df_series = Y_df[Y_df['unique_id'] == dataset_id].sort_values('ds')
        series_pd = df_series['y'].reset_index(drop=True)

        # Skip series with issues
        if series_pd.isnull().any() or not np.isfinite(series_pd).all() or series_pd.nunique() <= 1 or len(series_pd) < 2:
            skipped_count += 1
            continue

        series_np = series_pd.to_numpy()
        processed_count += 1
        
        # Calculate all features using the Metrics class
        current_features = {
            "unique_id": dataset_id,
            "Trend Changes": trend_changes(series_np),
            "Trend Strength": trend_strength(series_np),
            "Linear Regression Slope": linear_regression_slope(series_np),
            "Linear Regression R2": linear_regression_r2(series_np),
            "Series Forecastability": forecastability(series_np, sf=1),
            "Series Fluctuation": fluctuation(series_np),
            "Autocorrelation Relevance": ac_relevance(series_np),
            "Seasonal Strength": seasonal_strength(series_np),
            "Window Fluctuation": window_fluctuation(series_np),
            "Short-Term Variation": st_variation(series_np),
            "Differenced Series": diff_series(series_np),
            "Series Complexity": complexity(series_np),
            "Records Concentration": rec_concentration(series_np),
            "Series Centroid": centroid(series_np, fs=12),
        }
        results_data.append(current_features)

        if (processed_count) % 200 == 0:
            print(f"Processed {processed_count}/{len(all_dataset_ids)-skipped_count} series...")

    print(f"\nFeature calculation complete. Computed features for {processed_count} series.")

    if not results_data:
        print("No data was processed, cannot perform correlation analysis.")
        return

    results_df = pd.DataFrame(results_data)

    results_df.dropna(axis=1, how='all', inplace=True)

    # --- 2. Calculate, Rank, and Save Correlations ---
    print("\nCalculating feature correlations...")
    
    # Select only numeric columns for correlation calculation
    numeric_features_df = results_df.select_dtypes(include=np.number)

    # Calculate the correlation matrix
    corr_matrix = numeric_features_df.corr()

    # Unstack the matrix to create a Series of all pairs
    corr_pairs = corr_matrix.unstack()

    # Remove self-correlations (e.g., ('Trend Strength', 'Trend Strength'))
    corr_pairs = corr_pairs[corr_pairs.index.get_level_values(0) != corr_pairs.index.get_level_values(1)]
    
    # Create a canonical key for each pair 
    corr_pairs.index = corr_pairs.index.map(lambda x: tuple(sorted(x)))
    
    # Drop the duplicate pairs
    ranked_correlations = corr_pairs.drop_duplicates()
    
    # Sort the pairs by the absolute value of their correlation, in descending order
    ranked_correlations = ranked_correlations.abs().sort_values(ascending=False)

    print(f"Found {len(ranked_correlations)} unique feature pairs.")

    # Prepare the output file path
    output_filename = "feature_correlation_new_rank.txt"
    full_output_path = os.path.join(os.path.dirname(__file__), output_filename)

    print(f"Saving ranked correlations to: {full_output_path}")

    try:
        with open(full_output_path, 'w') as f:
            f.write("Ranked Feature Correlations (Absolute Values)\n")
            f.write("="*45 + "\n\n")
            
            for i, ((feat1, feat2), corr_val) in enumerate(ranked_correlations.items()):
                f.write(f"Rank {i+1}:\n")
                f.write(f"  - Feature 1: {feat1}\n")
                f.write(f"  - Feature 2: {feat2}\n")
                f.write(f"  - Correlation: {corr_val:.6f}\n\n")
        
        print("Successfully saved the correlation rankings.")

    except Exception as e:
        print(f"Error saving file: {e}")
        
    # --- 3. Display Top 10 Correlations Pairs ---
    print("\n--- Top 10 Most Correlated Feature Pairs ---")
    print(ranked_correlations.head(10).to_string())
    print("-" * 45)


if __name__ == "__main__":
    analyze_feature_correlations()
    print("\n--- Script Finished ---")
