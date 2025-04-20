import sys
import os

# --- Path Setup ---
# Assumes this script file is located one directory below the project root
# (e.g., in a 'testing' folder adjacent to 'src')
try:
    # __file__ is defined when running as a script
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
# This import should now work correctly if path setup is right
from src.metrics import Pelt, STLFeatures, ACF_Features, CrossingPoints, LinearRegression

# Suppress warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

def test_easytest_metrics():
    """
    Analyzes M3 monthly data using custom classes from src/metrics.py.
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

    # --- Instantiate custom metric classes --- # <<< CORRECTIONS HERE
    print("Instantiating custom metric classes...")
    try:
        pelt_instance = Pelt() # OK - Takes no args or uses defaults
        # OK - STLFeatures still takes freq. Using defaults for new 'seasonal' and 'robust' params.
        stl_instance = STLFeatures(freq=12)
        # CORRECTED: ACF_Features now takes 'nlags', not 'freq'. Using default nlags=10.
        acf_instance = ACF_Features(nlags=10)
        # CORRECTED: CrossingPoints now takes no arguments.
        crossing_points_instance = CrossingPoints()
        # OK - LinearRegression takes no arguments.
        linear_regression_instance = LinearRegression()
        print("Metric classes instantiated.")
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

        # Check minimum lengths required by calculators
        min_len_overall = 2
        # Get actual min_size used by this instance
        min_len_pelt = getattr(pelt_instance, 'min_size', 2)
         # Get actual freq used by this instance
        stl_freq = getattr(stl_instance, 'freq', 1)
        min_len_stl = (2 * stl_freq + 1) if stl_freq > 1 else 2
        # Get actual nlags used by this instance
        acf_nlags = getattr(acf_instance, 'nlags', 1)
        min_len_acf = acf_nlags + 1 # Need at least nlags+1 points for ACF

        # Ensure series is long enough for ALL calculations intended
        # (Add more checks if other features have stricter length needs)
        required_len = max(min_len_overall, min_len_pelt, min_len_stl, min_len_acf)
        if series_len < required_len:
             # Optional: print which check failed
             # print(f"Skipping {dataset_id}: len {series_len} < required {required_len}")
             skipped_count += 1
             continue
        # --- End Pre-computation Checks ---

        processed_count += 1
        current_features = {
            "unique_id": dataset_id,
            "Pelt_Num_Breakpoints": np.nan,
            "STL_Trend_Strength": np.nan,
            "ACF_FirstLag": np.nan,
            "CrossingPoints": np.nan,
            "LinearRegression_Slope": np.nan,
            "LinearRegression_R2": np.nan,
        }

        # --- Feature Extraction (with basic try-except for safety) ---
        try:
            # 1. Pelt
            pen_value = np.log(series_len) if series_len > 1 else 0 # Avoid log(1)
            pelt_instance.fit(series_np)
            bkps = pelt_instance.predict(pen=pen_value)
            # Predict returns end points, number of changes is len(bkps)-1 if bkps includes series end
            # A single segment (0, N) gives bkps=[N], len=1 -> 0 changes.
            # Segments (0, t1), (t1, N) gives bkps=[t1, N], len=2 -> 1 change.
            # Handle empty bkps case (might happen on error or very short series)
            num_changepoints = len(bkps) -1 if bkps else 0
            current_features["Pelt_Num_Breakpoints"] = max(0, num_changepoints)

            # 2. STL
            stl_result = stl_instance.get_features(x=series_np) # Already checked length
            current_features["STL_Trend_Strength"] = stl_result.get('trend', np.nan)

            # 3. ACF
            acf_result = acf_instance.get_features(x=series_np) # Already checked length
            current_features["ACF_FirstLag"] = acf_result.get('x_acf1', np.nan)

            # 4. Crossing Points
            # Pass float array, handles potential NaNs inside get_features
            cp_result = crossing_points_instance.get_features(x=series_np.astype(float))
            current_features["CrossingPoints"] = cp_result.get('crossing_points', np.nan)

            # 5. Linear Regression
            # Use a new instance per fit or ensure fit resets state if reusing
            lr_instance = LinearRegression()
            lr_instance.fit(time_series=series_np)
            current_features["LinearRegression_Slope"] = lr_instance.coef_
            current_features["LinearRegression_R2"] = lr_instance.score()

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
    # Use context manager for float formatting
    with pd.option_context('display.float_format', '{:,.4f}'.format, 'display.max_rows', None, 'display.max_columns', None, 'display.width', 1000):
        print(results_df.describe())
    print("\n--- Sample of Custom Results DataFrame ---")
    print(results_df.head())
    print("\n--- Custom Extreme Values per Feature ---")
    extreme_results = {}
    try:
        for feature_name in results_df.select_dtypes(include=np.number).columns:
            valid_series = results_df[feature_name].dropna()
            if valid_series.empty or not np.all(np.isfinite(valid_series)): # Check for finite values too
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
         return {} # Return empty if extremes calculation fails


    return extreme_results

# NpEncoder class definition (keep as is)
class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer): return int(obj)
        if isinstance(obj, np.floating): return float(obj) # Handle numpy floats
        if isinstance(obj, np.ndarray): return obj.tolist()
        # Check for NaN specifically BEFORE standard JSON encoding
        if isinstance(obj, float) and np.isnan(obj): return None # Represent NaN as null in JSON
        return super(NpEncoder, self).default(obj)

# --- Main execution block ---
if __name__ == "__main__":
    print(f"--- Running Custom Metrics Analysis Script ---")
    # Add try-except around the main function call for robustness
    analysis_results = {}
    try:
        analysis_results = test_easytest_metrics()
    except Exception as e:
        print(f"\n--- SCRIPT EXECUTION FAILED ---")
        print(f"An unexpected error occurred: {e}")
        # Optionally re-raise or exit
        # raise e

    print("\n--- Script Finished ---")
    if analysis_results:
        print("Results dictionary was generated (extreme values shown above).")
        # Example: Optionally save results to JSON
        # try:
        #     output_path = os.path.join(PROJECT_ROOT, 'results.json')
        #     with open(output_path, 'w') as f:
        #         json.dump(analysis_results, f, cls=NpEncoder, indent=4)
        #     print(f"Results saved to {output_path}")
        # except Exception as e:
        #     print(f"Error saving results to JSON: {e}")
    else:
        print("Analysis did not generate results or failed.")