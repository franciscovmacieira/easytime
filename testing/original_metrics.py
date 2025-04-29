import numpy as np
import pandas as pd
import warnings
import json
import os
import sys
# --- Library Imports ---
from datasetsforecast.m3 import M3
import ruptures as rpt
from ruptures.detection import Pelt 
from tsfeatures import acf_features
from sklearn.linear_model import LinearRegression as SkLearnLinearRegression
from statsmodels.tsa.seasonal import STL
import antropy as ant
import pycatch22


# Suppress warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

def test_original_metrics():
    """
    Analyzes M3 monthly data using original library functions/implementations.
    Includes: Pelt, STL Trend Strength, ACF Lag 1, Crossing Points (Manual),
              Linear Regression (Slope, R2), Spectral Entropy (antropy),
              EntropyPairs (catch22), HighFluctuation (catch22).
    Returns dictionary of extreme values found.
    """
 # --- Load Data (Will crash if fails) ---
    data_dir = './data'
    print(f"\nLoading M3 Monthly data (will download to '{data_dir}' if needed)...")

    loaded_data = M3.load(directory=data_dir, group='Monthly')
    if isinstance(loaded_data, (list, tuple)): Y_df = loaded_data[0]
    elif isinstance(loaded_data, pd.DataFrame): Y_df = loaded_data
    else: raise TypeError("Loaded data not DataFrame")
    print("Data loaded successfully.")

    # --- Data Validation (Will crash if columns missing) ---
    required_cols = ['unique_id', 'ds', 'y']
    if not all(col in Y_df.columns for col in required_cols):
        raise ValueError(f"Error: Missing required columns: {[c for c in required_cols if c not in Y_df.columns]}")
    # --- End Load Data ---

    all_dataset_ids = Y_df['unique_id'].unique()
    results_data = []

    # --- Instantiate classes if needed ---
    print("Instantiating necessary library models (Pelt, Linear Regression)...")
    lr_model_instance = SkLearnLinearRegression() # Can reuse instance
    pelt_instance = Pelt()
    print("Using library Pelt with default parameters.")


    print(f"\nAnalyzing {len(all_dataset_ids)} monthly time series using LIBRARY functions (NO error handling)...")

    processed_count = 0
    skipped_count = 0 # Add skipped counter
    for i, dataset_id in enumerate(all_dataset_ids):
        df_series = Y_df[Y_df['unique_id'] == dataset_id].sort_values('ds')
        series_pd = df_series['y'].reset_index(drop=True) # Keep as pandas Series for tsfeatures

        # --- Pre-computation Checks ---
        if series_pd.isnull().any() or not np.isfinite(series_pd).all():
            skipped_count += 1
            continue
        if series_pd.nunique() <= 1: # Skip constant series
             skipped_count += 1
             continue

        series_np = series_pd.to_numpy() # Convert to numpy for libraries needing it
        series_len = len(series_np)

        min_len_overall = 2 # Most features need at least 2 points
        min_len_stl = 2 * 12 + 1 # Specific to monthly STL
        min_len_pelt = getattr(pelt_instance, 'min_size', 2) # Get default min size

        required_len = max(min_len_overall, min_len_pelt, min_len_stl) # Check against main known limits
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
        # Wrap individual library calls for safety, although script aims for no handling

        # 1. Pelt Breakpoints (Library)
        try:
            pen_value = np.log(series_len) if series_len > 1 else 0
            algo = pelt_instance.fit(series_np)
            bkps = algo.predict(pen=pen_value)
            num_changepoints = len(bkps) - 1 if bkps else 0
            current_features["Pelt_Num_Breakpoints"] = max(0, num_changepoints)
        except Exception as e: print(f"WARN: Pelt failed for {dataset_id}: {e}")

        # 2. STL Trend Strength (Statsmodels)
        try:
            stl_sm = STL(series_np, period=12) # Defaults: seasonal=7, robust=False
            res_sm = stl_sm.fit()
            remainder = res_sm.resid; deseason = series_np - res_sm.seasonal
            vare = np.nanvar(remainder, ddof=1); vardeseason = np.nanvar(deseason, ddof=1)
            if vardeseason <= 1e-10: trend_strength = 0.0
            else: trend_strength = max(0., min(1., 1. - (vare / vardeseason if vardeseason > 1e-10 else 1e-10))) # Avoid div by zero
            current_features["STL_Trend_Strength"] = trend_strength
        except Exception as e: print(f"WARN: STL failed for {dataset_id}: {e}")


        # 3. ACF Features (tsfeatures)
        try:
            acf_result = acf_features(series_pd)
            current_features["ACF_FirstLag"] = acf_result.get('x_acf1', np.nan)
        except Exception as e: print(f"WARN: ACF (tsfeatures) failed for {dataset_id}: {e}")


        # 4. Crossing Points (Manual Median)
        try:
            if series_len > 1:
                midline = np.median(series_np)
                ab = series_np <= midline
                p1 = ab[:-1]; p2 = ab[1:]
                cross = (p1 & (~p2)) | (p2 & (~p1))
                current_features["CrossingPoints"] = cross.sum()
            else: current_features["CrossingPoints"] = 0
        except Exception as e: print(f"WARN: CrossingPoints (manual) failed for {dataset_id}: {e}")


        # 5. Linear Regression (Scikit-learn)
        try:
            time_steps = np.arange(series_len).reshape(-1, 1)
            lr_model_instance.fit(time_steps, series_np)
            slope = lr_model_instance.coef_[0] if lr_model_instance.coef_.size > 0 else np.nan
            r_squared = lr_model_instance.score(time_steps, series_np)
            current_features["LinearRegression_Slope"] = slope
            current_features["LinearRegression_R2"] = r_squared
        except Exception as e: print(f"WARN: LinearRegression (sklearn) failed for {dataset_id}: {e}")


        # 6. Spectral Entropy
        try:
            # Using sf=1 (samples per month), welch method default, no normalization
            spec_entropy = ant.spectral_entropy(series_np, sf=1, method='welch', normalize=False)
            current_features["SpectralEntropy_Value"] = spec_entropy
        except Exception as e: print(f"WARN: SpectralEntropy (antropy) failed for {dataset_id}: {e}")


        # 7. & 8. EntropyPairs HighFluctuation
        try:
            series_list = series_np.tolist() # Use list as per recommendation
            # Get the raw results which contain 'names' and 'values' lists
            catch22_raw_results = pycatch22.catch22_all(series_list, catch24=False)

            # --- Create a name-to-value dictionary from the results ---
            if 'names' in catch22_raw_results and 'values' in catch22_raw_results:
                # Use zip to pair names with values and create a dictionary
                feature_dict = dict(zip(catch22_raw_results['names'], catch22_raw_results['values']))
            else:
                # Handle case where pycatch22 might return unexpected format
                print(f"WARN: Unexpected format from pycatch22 for {dataset_id}. Keys: {catch22_raw_results.keys()}")
                feature_dict = {} # Use empty dict to avoid errors below
            # --- End dictionary creation ---

            current_features["EntropyPairs_Value"] = feature_dict.get('SB_MotifThree_quantile_hh', np.nan)
            current_features["HighFluctuation_Value"] = feature_dict.get('MD_hrv_classic_pnn40', np.nan)

        except Exception as e:
            print(f"WARN: Catch22 calculation failed for {dataset_id}: {e}")
            current_features["EntropyPairs_Value"] = np.nan
            current_features["HighFluctuation_Value"] = np.nan


        # --- Append results for this series ---
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
    # (Keep this section as is)
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
    print("\n--- Library Feature Calculation Summary ---")
    print(results_df.info()) # Info will show new columns
    print("\n--- Library Descriptive Statistics ---")
    with pd.option_context('display.float_format', '{:,.4f}'.format, 'display.max_rows', None, 'display.max_columns', None, 'display.width', 1000):
        print(results_df.describe()) # Describe will include new columns
    print("\n--- Sample Library Results DataFrame ---")
    print(results_df.head()) # Head will show new columns
    print("\n--- Library Extreme Values per Feature ---")
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
    try:
        PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
        if PROJECT_ROOT not in sys.path: sys.path.insert(0, PROJECT_ROOT)
    except NameError: # If __file__ is not defined (e.g. interactive)
        PROJECT_ROOT = os.getcwd()
        if PROJECT_ROOT not in sys.path: sys.path.insert(0, PROJECT_ROOT)

    print(f"--- Running Library Metrics Analysis Script ---")
    analysis_results = {}
    try:
        analysis_results = test_original_metrics()
    except Exception as e:
        print(f"\n--- SCRIPT EXECUTION FAILED ---")
        print(f"An unexpected error occurred: {e}")
        import traceback
        traceback.print_exc() # Print full traceback on error

    print("\n--- Script Finished ---")
    if analysis_results:
        print("Results dictionary was generated (extreme values shown above).")
    else:
        print("Analysis did not generate results or failed.")