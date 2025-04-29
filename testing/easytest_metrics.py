import sys
import os

# --- Path Setup ---
try:
    PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
except NameError:
    # Fallback for interactive environments (like Jupyter basic run)
    # Assumes the notebook CWD is the project root
    print("Warning: __file__ not defined. Assuming CWD is project root for path setup.")
    PROJECT_ROOT = os.getcwd()

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
    print(f"Added to sys.path: {PROJECT_ROOT}")

# --- Imports ---
import numpy as np
import pandas as pd
import warnings
import json
from datasetsforecast.m3 import M3
from src.metrics import (
    Pelt, STLFeatures, ACF_Features, CrossingPoints, LinearRegression,
    EntropyPairs, SpectralEntropy, HighFluctuation
)


# Suppress warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

def test_easytest_metrics():
    """
    Analyzes M3 monthly data using custom classes from src/metrics.py.
    Includes Pelt, STL, ACF, CrossingPoints, LinearRegression,
    EntropyPairs, SpectralEntropy, HighFluctuation features.
    Returns a dictionary of extreme values found for each feature.
    """
    # --- Load Data ---
    # Using a relative path to the project root for data directory
    data_dir = os.path.join(PROJECT_ROOT, 'data', 'm3_download') # Store downloads here
    print(f"\nLoading M3 Monthly data (will download to '{data_dir}' if needed)...")
    os.makedirs(data_dir, exist_ok=True) # Ensure data dir exists

    try:
        loaded_data = M3.load(directory=data_dir, group='Monthly')
        Y_df = None # Initialize Y_df
        if isinstance(loaded_data, (list, tuple)) and len(loaded_data) > 0:
            Y_df = loaded_data[0]
        elif isinstance(loaded_data, pd.DataFrame):
            Y_df = loaded_data
        else:
             raise TypeError(f"Loaded data not DataFrame or list/tuple, got {type(loaded_data)}")

        if Y_df is None: raise ValueError("Failed to extract DataFrame from loaded data.")

        print("Data loaded successfully.")
        # Basic validation
        required_cols = ['unique_id', 'ds', 'y']
        if not all(col in Y_df.columns for col in required_cols):
            raise ValueError(f"Error: Missing required columns: {[c for c in required_cols if c not in Y_df.columns]}")
        Y_df['ds'] = pd.to_datetime(Y_df['ds']) # Ensure datetime conversion

    except Exception as e:
         print(f"FATAL ERROR during data loading: {e}")
         return {} # Return empty if data loading fails


    # --- End Load Data ---

    all_dataset_ids = Y_df['unique_id'].unique()
    results_data = []

    # --- Instantiate custom metric classes ---
    print("Instantiating custom metric classes...")
    try:
        pelt_instance = Pelt()
        stl_instance = STLFeatures(freq=12) # Assuming monthly data
        acf_instance = ACF_Features(nlags=10)
        crossing_points_instance = CrossingPoints()
        linear_regression_instance = LinearRegression() # Instantiated but fit per series     
        entropy_pairs_instance = EntropyPairs()
        spectral_entropy_instance = SpectralEntropy(sf=1) # Use sf=1 for monthly data (unit cycles/month)
        high_fluctuation_instance = HighFluctuation()

        print("Metric classes instantiated.")
    except NameError as e:
         print(f"FATAL ERROR during metric instantiation: Class not found. Did you import correctly? Details: {e}")
         return {}
    except Exception as e:
         print(f"FATAL ERROR during metric instantiation: {e}")
         return {}

    # --- End Instantiation ---

    print(f"\nAnalyzing {len(all_dataset_ids)} monthly time series using CUSTOM metrics...")

    processed_count = 0
    skipped_count = 0
    for i, dataset_id in enumerate(all_dataset_ids):
        df_series = Y_df[Y_df['unique_id'] == dataset_id].sort_values('ds')
        series_pd = df_series['y'].reset_index(drop=True)

        # --- Pre-computation Checks ---
        if series_pd.isnull().any() or not np.isfinite(series_pd).all():
            skipped_count += 1
            continue
        if series_pd.nunique() <= 1: # Skip constant series
             skipped_count += 1
             continue

        series_np = series_pd.to_numpy()
        series_len = len(series_np)

        # Check minimum lengths required by calculators <<< UPDATED HERE
        min_len_overall = 2 # Most features need at least 2 points
        min_len_pelt = getattr(pelt_instance, 'min_size', 2)
        stl_freq = getattr(stl_instance, 'freq', 1); min_len_stl = (2 * stl_freq + 1) if stl_freq > 1 else 2
        acf_nlags = getattr(acf_instance, 'nlags', 1); min_len_acf = acf_nlags + 1
        # New features generally need min length 2 based on their internal checks
        min_len_entropy_pairs = 2
        min_len_spectral = 2
        min_len_fluctuation = 2

        # Ensure series is long enough for ALL calculations intended
        required_len = max(min_len_overall, min_len_pelt, min_len_stl, min_len_acf,
                           min_len_entropy_pairs, min_len_spectral, min_len_fluctuation)

        if series_len < required_len:
             skipped_count += 1
             continue
        # --- End Pre-computation Checks ---

        processed_count += 1
        # Initialize features dict
        current_features = {
            "unique_id": dataset_id,
            "Pelt_Num_Breakpoints": np.nan,
            "STL_Trend_Strength": np.nan,
            "ACF_FirstLag": np.nan,
            "CrossingPoints": np.nan,
            "LinearRegression_Slope": np.nan,
            "LinearRegression_R2": np.nan,
            "EntropyPairs_Value": np.nan,     
            "SpectralEntropy_Value": np.nan,
            "HighFluctuation_Value": np.nan   
        }

        # --- Feature Extraction ---
        try:
            # 1. Pelt
            pen_value = np.log(series_len) if series_len > 1 else 0
            pelt_instance.fit(series_np)
            bkps = pelt_instance.predict(pen=pen_value)
            num_changepoints = len(bkps) -1 if bkps else 0
            current_features["Pelt_Num_Breakpoints"] = max(0, num_changepoints)

            # 2. STL
            stl_result = stl_instance.get_features(x=series_np)
            current_features["STL_Trend_Strength"] = stl_result.get('trend', np.nan)

            # 3. ACF
            acf_result = acf_instance.get_features(x=series_np)
            current_features["ACF_FirstLag"] = acf_result.get('x_acf1', np.nan)

            # 4. Crossing Points
            cp_result = crossing_points_instance.get_features(x=series_np.astype(float))
            current_features["CrossingPoints"] = cp_result.get('crossing_points', np.nan)

            # 5. Linear Regression
            lr_instance = LinearRegression() # Re-instantiate per series
            lr_instance.fit(time_series=series_np)
            current_features["LinearRegression_Slope"] = lr_instance.coef_
            current_features["LinearRegression_R2"] = lr_instance.score()

            # 6. Entropy Pairs
            entropy_pairs_result = entropy_pairs_instance.get_features(x=series_np)
            current_features["EntropyPairs_Value"] = entropy_pairs_result.get('entropy_pairs', np.nan)

            # 7. Spectral Entropy 
            spectral_entropy_result = spectral_entropy_instance.get_features(x=series_np)
            current_features["SpectralEntropy_Value"] = spectral_entropy_result.get('spectral_entropy', np.nan)

            # 8. High Fluctuation
            high_fluctuation_result = high_fluctuation_instance.get_features(x=series_np)
            current_features["HighFluctuation_Value"] = high_fluctuation_result.get('high_fluctuation', np.nan)

        except Exception as e:
             print(f"ERROR calculating features for {dataset_id}: {e}")
             # Keep features as NaN on error

        results_data.append(current_features)

        # --- Progress Update ---
        if (processed_count) % 100 == 0 or (i + 1) == len(all_dataset_ids):
            print(f"Processed {processed_count}/{len(all_dataset_ids)-skipped_count} series... (Checked {i + 1}, Skipped {skipped_count})")

    # --- AFTER THE LOOP ---
    print(f"\nAnalysis loop complete. Features computed for {processed_count} series. {skipped_count} series skipped.")

    if not results_data:
        print("No data was processed successfully.")
        return {}

    # --- Convert results to DataFrame ---
    try:
        results_df = pd.DataFrame(results_data)
        if 'unique_id' in results_df.columns:
             results_df.set_index('unique_id', inplace=True)
        else:
             print("Warning: 'unique_id' column missing after DataFrame creation.")
    except Exception as e:
         print(f"FATAL ERROR converting results to DataFrame: {e}")
         return {}

    # --- Calculate and Display Results ---
    print("\n--- Custom Feature Calculation Summary ---")
    print(results_df.info()) 
    print("\n--- Custom Descriptive Statistics ---")
    with pd.option_context('display.float_format', '{:,.4f}'.format, 'display.max_rows', None, 'display.max_columns', None, 'display.width', 1000):
        print(results_df.describe()) 
    print("\n--- Sample of Custom Results DataFrame ---")
    print(results_df.head()) 
    print("\n--- Custom Extreme Values per Feature ---")
    extreme_results = {}
    try:
        for feature_name in results_df.select_dtypes(include=np.number).columns:
            valid_series = results_df[feature_name].dropna()
            if valid_series.empty or not np.all(np.isfinite(valid_series)):
                print(f"\nFeature: {feature_name}\n   No valid finite data found.")
                extreme_results[feature_name] = {'lowest': (None, np.nan), 'highest': (None, np.nan)}
                continue
            idx_min, val_min = valid_series.idxmin(), valid_series.min()
            idx_max, val_max = valid_series.idxmax(), valid_series.max()
            print(f"\nFeature: {feature_name}")
            print(f"   Lowest:  ID = {idx_min}, Value = {val_min:.4f}")
            print(f"   Highest: ID = {idx_max}, Value = {val_max:.4f}")
            extreme_results[feature_name] = {'lowest': (idx_min, val_min), 'highest': (idx_max, val_max)}
    except Exception as e:
         print(f"FATAL ERROR calculating extreme values: {e}")
         return {}

    return extreme_results

# NpEncoder class definition
class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer): return int(obj)
        if isinstance(obj, np.floating): return float(obj)
        if isinstance(obj, np.ndarray): return obj.tolist()
        if isinstance(obj, float) and np.isnan(obj): return None
        return super(NpEncoder, self).default(obj)

# --- Main execution block ---
if __name__ == "__main__":
    print(f"--- Running Custom Metrics Analysis Script ---")
    analysis_results = {}
    try:
        analysis_results = test_easytest_metrics()
    except Exception as e:
        print(f"\n--- SCRIPT EXECUTION FAILED ---")
        print(f"An unexpected error occurred: {e}")

    print("\n--- Script Finished ---")
    if analysis_results:
        print("Results dictionary was generated (extreme values shown above).")
    else:
        print("Analysis did not generate results or failed.")