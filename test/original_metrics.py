import numpy as np
import pandas as pd
import warnings
import json

from datasetsforecast.m3 import M3
import ruptures as rpt 
from ruptures.detection import Pelt 
from tsfeatures import acf_features 
from sklearn.linear_model import LinearRegression as SkLearnLinearRegression
from statsmodels.tsa.seasonal import STL

# Suppress warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

def test_original_metrics():
    """
    Analyzes M3 monthly data using original library functions.
    Pelt uses library defaults. STL uses statsmodels directly (period=12).
    CrossingPoints calculated manually (median). ACF uses tsfeatures workaround.
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
    pelt_instance = Pelt() # Use library default parameters model='l2', min_size=2, jump=5
    print("Using library Pelt with default parameters.")


    print(f"\nAnalyzing {len(all_dataset_ids)} monthly time series using LIBRARY functions (NO error handling)...")

    processed_count = 0
    for i, dataset_id in enumerate(all_dataset_ids):
        df_series = Y_df[Y_df['unique_id'] == dataset_id].sort_values('ds')
        series_pd = df_series['y'].reset_index(drop=True)

        # --- Pre-computation Checks ---
        if series_pd.isnull().any() or not np.isfinite(series_pd).all(): continue
        if series_pd.nunique() <= 1: continue
        series_np = series_pd.to_numpy()
        min_len_overall = 2
        min_len_stl = 2 * 12 + 1
        min_len_pelt = pelt_instance.min_size
        if len(series_np) < max(min_len_overall, min_len_pelt): continue
        # --- End Pre-computation Checks ---

        processed_count += 1
        # Initialize features dict with CONSISTENT keys
        current_features = {
            "unique_id": dataset_id,
            "Pelt_Num_Breakpoints": np.nan,
            "STL_Trend_Strength": np.nan, # Consistent Key
            "ACF_FirstLag": np.nan,
            "CrossingPoints": np.nan,     # Consistent Key
            "LinearRegression_Slope": np.nan,
            "LinearRegression_R2": np.nan,
        }


        # --- Feature Extraction ---

        # 1. Pelt Breakpoints - Use LIBRARY Pelt with DEFAULTS, log(n) penalty
        pen_value = np.log(len(series_np))
        algo = pelt_instance.fit(series_np) # fit returns self
        bkps = algo.predict(pen=pen_value)
        # Calculate number of CHANGEPOINTS
        num_changepoints = len(bkps) - 1 if bkps else 0
        current_features["Pelt_Num_Breakpoints"] = max(0, num_changepoints)

        # 2. STL Trend Strength - USING STATSMODELS DIRECTLY FOR FAIR COMPARISON
        if len(series_np) >= min_len_stl:
            stl_sm = STL(series_np, period=12, robust=True) # Using robust=True example
            res_sm = stl_sm.fit()
            remainder = res_sm.resid
            deseason = series_np - res_sm.seasonal
            vare = np.nanvar(remainder, ddof=1)
            vardeseason = np.nanvar(deseason, ddof=1)
            if vardeseason <= 1e-10:
                 trend_strength = 0.0
            else:
                 trend_strength = max(0., min(1., 1. - vare / vardeseason))
            current_features["STL_Trend_Strength"] = trend_strength
        # else: STL_Trend_Strength remains NaN

        # 3. ACF Features - Use tsfeatures library function (workaround call)
        acf_result = acf_features(series_pd) # Simplified call for v0.4.5
        current_features["ACF_FirstLag"] = acf_result.get('x_acf1', np.nan)

        # 4. Crossing Points - *** MANUAL CALCULATION (Median) ***
        # Bypassing potentially unreliable tsfeatures.crossing_points from v0.4.5
        midline = np.median(series_np) # Calculate median
        ab = series_np <= midline
        lenx = len(series_np)
        if lenx > 1:
            p1 = ab[:(lenx - 1)]
            p2 = ab[1:]
            cross = (p1 & (~p2)) | (p2 & (~p1)) # Find crossings
            current_features["CrossingPoints"] = cross.sum() # Store the count
        else:
             current_features["CrossingPoints"] = 0 # Assign 0 if length is 1

        # 5. Linear Regression - Use scikit-learn library function
        time_steps = np.arange(len(series_np)).reshape(-1, 1)
        lr_model_instance.fit(time_steps, series_np) # Use the single instance
        slope = lr_model_instance.coef_[0] if lr_model_instance.coef_.size > 0 else np.nan
        r_squared = lr_model_instance.score(time_steps, series_np)
        current_features["LinearRegression_Slope"] = slope
        current_features["LinearRegression_R2"] = r_squared

        # --- Append results for this series ---
        results_data.append(current_features)

        # --- Progress Update ---
        if (processed_count) % 100 == 0 or (i + 1) == len(all_dataset_ids):
            print(f"Processed {i + 1}/{len(all_dataset_ids)} potential series... ({processed_count} completed)")

    # --- AFTER THE LOOP ---
    print(f"\nAnalysis loop complete. Features computed for {processed_count} series.")

    if not results_data:
        print("No data was processed successfully.")
        return {}

    # --- Convert results to DataFrame (Will crash if fails) ---
    print("Converting results to DataFrame...")
    results_df = pd.DataFrame(results_data)
    if 'unique_id' in results_df.columns:
         results_df.set_index('unique_id', inplace=True)
    else:
        print("Warning: 'unique_id' column missing after DataFrame creation.")

    # --- Display Results (Will crash if DataFrame is invalid) ---
    print("\n--- Library Feature Calculation Summary (Pelt Defaults, Direct STL, Manual CP) ---")
    print(results_df.info())
    print("\n--- Library Descriptive Statistics (Pelt Defaults, Direct STL, Manual CP) ---")
    with pd.option_context('display.float_format', '{:,.2f}'.format): print(results_df.describe())
    print("\n--- Sample Library Results DataFrame (Pelt Defaults, Direct STL, Manual CP) ---")
    print(results_df.head())
    print("\n--- Library Extreme Values per Feature (Pelt Defaults, Direct STL, Manual CP) ---")
    extreme_results = {}
    # Direct calculation (Will crash on errors)
    for feature_name in results_df.select_dtypes(include=np.number).columns:
        valid_series = results_df[feature_name].dropna()
        if valid_series.empty:
            print(f"\nFeature: {feature_name}\n  No valid finite data found.")
            extreme_results[feature_name] = {'lowest': (None, None), 'highest': (None, None)}
            continue

        idx_min, val_min = valid_series.idxmin(), valid_series.min()
        idx_max, val_max = valid_series.idxmax(), valid_series.max()
        print(f"\nFeature: {feature_name}")
        print(f"  Lowest:  ID = {idx_min}, Value = {val_min:.4f}")
        print(f"  Highest: ID = {idx_max}, Value = {val_max:.4f}")
        extreme_results[feature_name] = {'lowest': (idx_min, val_min), 'highest': (idx_max, val_max)}

    return extreme_results

class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer): return int(obj)
        if isinstance(obj, np.floating): return float(obj)
        if isinstance(obj, np.ndarray): return obj.tolist()
        if np.isnan(obj): return None
        return super(NpEncoder, self).default(obj)

if __name__ == "__main__":
    print(f"--- Running Library Metrics Analysis Script: {__file__} ---")
    # Direct function call - will crash on any unhandled error inside
    analysis_results = test_original_metrics()

    print("\nAnalysis function completed without crashing.")
    if analysis_results:
        print("Results dictionary was generated (showing extremes above).")
    else:
        print("Analysis generated no results.")