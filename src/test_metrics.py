import numpy as np
from datasetsforecast.m3 import M3
import pandas as pd
# Assuming your custom metrics classes are in a file named 'metrics.py'
# If not, adjust the import accordingly.
from metrics import Pelt, STLFeatures, ACF_Features, CrossingPoints, LinearRegression
# Placeholder classes if metrics.py is not available for testing the loading part
class BaseMetric:
    def get_features(self, signal, **kwargs):
        # Simple placeholder logic - replace with your actual implementations
        if isinstance(signal, pd.Series):
            val = signal.mean() if not signal.empty else 0
        else: # Assuming signal is numpy array for Pelt
             val = np.mean(signal) if signal.size > 0 else 0

        if isinstance(self, Pelt):
            return {'breakpoints': [1, 2] if len(signal)>1 else []} # Dummy breakpoints
        if isinstance(self, STLFeatures):
            return {'trend': val * 0.9} # Dummy trend
        if isinstance(self, ACF_Features):
            return {'x_acf1': val * 0.1} # Dummy ACF
        if isinstance(self, CrossingPoints):
            return {'crossing_points': int(val / 10)} # Dummy crossing points
        if isinstance(self, LinearRegression):
             # Dummy slope and R-squared
            return {'slope': val * 0.01, 'r_squared': 0.5 if val != 0 else 0}
        return {}

class Pelt(BaseMetric): pass
class STLFeatures(BaseMetric): pass
class ACF_Features(BaseMetric): pass
class CrossingPoints(BaseMetric): pass
class LinearRegression(BaseMetric): pass
# --- End of Placeholder Classes ---


def analyze_m3_monthly_data():
    """
    Analyzes the M3 monthly datasets to find the datasets with the lowest
    and highest values for each of the features (Pelt, STL, ACF, Crossing Points, Linear Regression).
    """
    # Load the M3 monthly datasets
    data_dir = './data'
    print(f"Loading M3 Monthly data (will download to '{data_dir}' if needed)...")

    # --- CORRECTION HERE ---
    # M3.load in your environment seems to return only Y_df.
    # Assign the result to a single variable.
    try:
        Y_df = M3.load(directory=data_dir, group='Monthly')
        # Check if X_df exists (newer versions might return it)
        # If M3.load returns a tuple/list:
        if isinstance(Y_df, (list, tuple)):
             if len(Y_df) == 2:
                 Y_df, _ = Y_df # Unpack if it returns two items
             elif len(Y_df) == 1:
                 Y_df = Y_df[0] # Get the first item if it's a single-item tuple/list
             else:
                 # If it returns more than 2, take the first one as Y_df (best guess)
                 print("Warning: M3.load returned more than 2 values. Assuming the first is Y_df.")
                 Y_df = Y_df[0]
        # If it's not a tuple/list, assume it's the DataFrame directly (older versions)
        elif not isinstance(Y_df, pd.DataFrame):
             raise TypeError(f"M3.load returned an unexpected type: {type(Y_df)}")

    except Exception as e:
        print(f"Error during M3.load: {e}")
        print("Please check your datasetsforecast library version and installation.")
        return None # Exit the function if loading fails

    print("Data loaded successfully.")
    # --- END CORRECTION ---

    # Ensure Y_df is actually a DataFrame before proceeding
    if not isinstance(Y_df, pd.DataFrame) or 'unique_id' not in Y_df.columns:
         print("Error: Loaded data is not the expected DataFrame.")
         return None

    all_dataset_ids = Y_df['unique_id'].unique()

    results = {}
    feature_values = {
        "Pelt_Breakpoints": [],
        "STL_Trend": [],
        "ACF_FirstLag": [],
        "CrossingPoints": [],
        "LinearRegression_slope": [],
        "LinearRegression_r_squared": [],
    }
    processed_dataset_ids = []

    pelt_instance = Pelt()
    stl_instance = STLFeatures()
    acf_instance = ACF_Features()
    crossing_points_instance = CrossingPoints()
    linear_regression_instance = LinearRegression()

    print(f"Analyzing {len(all_dataset_ids)} monthly time series...")
    for i, dataset_id in enumerate(all_dataset_ids):
        df_series = Y_df[Y_df['unique_id'] == dataset_id]
        # Added check for 'ds' column existence
        if 'ds' not in df_series.columns:
             print(f"Skipping dataset {dataset_id}: 'ds' column missing.")
             continue
        series = df_series.sort_values('ds')['y']

        if len(series) < 2:
             #print(f"Skipping dataset {dataset_id}: Too short (length {len(series)})") # Less verbose
             continue

        processed_dataset_ids.append(dataset_id)

        try:
            series_np = series.to_numpy()
            pelt_features = pelt_instance.get_features(signal=series_np, pen=10)['breakpoints']
            feature_values["Pelt_Breakpoints"].append(len(pelt_features))

            stl_features_result = stl_instance.get_features(series)
            feature_values["STL_Trend"].append(stl_features_result['trend'])

            acf_features_result = acf_instance.get_features(series)
            feature_values["ACF_FirstLag"].append(acf_features_result['x_acf1'])

            crossing_points_result = crossing_points_instance.get_features(series)
            feature_values["CrossingPoints"].append(crossing_points_result['crossing_points'])

            linear_regression_result = linear_regression_instance.get_features(series)
            feature_values["LinearRegression_slope"].append(linear_regression_result['slope'])
            feature_values["LinearRegression_r_squared"].append(linear_regression_result['r_squared'])

        except Exception as e:
            print(f"Error processing dataset {dataset_id}: {e}")
            feature_values["Pelt_Breakpoints"].append(None)
            feature_values["STL_Trend"].append(None)
            feature_values["ACF_FirstLag"].append(None)
            feature_values["CrossingPoints"].append(None)
            feature_values["LinearRegression_slope"].append(None)
            feature_values["LinearRegression_r_squared"].append(None)

        if (i + 1) % 100 == 0:
             print(f"Processed {i + 1}/{len(all_dataset_ids)} series...")

    print("Analysis complete. Finding extreme values...")

    for feature_name, values in feature_values.items():
        valid_values_with_ids = [
            (processed_dataset_ids[i], v) for i, v in enumerate(values) if v is not None and np.isfinite(v)
        ]

        if not valid_values_with_ids:
            print(f"\nNo valid finite data for feature: {feature_name}")
            results[feature_name] = {'lowest': (None, None), 'highest': (None, None)}
            continue

        lowest_dataset_id, lowest_value = min(valid_values_with_ids, key=lambda item: item[1])
        highest_dataset_id, highest_value = max(valid_values_with_ids, key=lambda item: item[1])

        print(f"\nFeature: {feature_name}")
        if isinstance(lowest_value, (int, float)):
             print(f"  Lowest:  Dataset = {lowest_dataset_id}, Value = {lowest_value:.4f}")
        else:
             print(f"  Lowest:  Dataset = {lowest_dataset_id}, Value = {lowest_value}")
        if isinstance(highest_value, (int, float)):
             print(f"  Highest: Dataset = {highest_dataset_id}, Value = {highest_value:.4f}")
        else:
              print(f"  Highest: Dataset = {highest_dataset_id}, Value = {highest_value}")

        results[feature_name] = {
            'lowest': (lowest_dataset_id, lowest_value),
            'highest': (highest_dataset_id, highest_value),
        }
    return results


if __name__ == "__main__":
    analysis_results = analyze_m3_monthly_data()
    # Optional: print results
    # if analysis_results:
    #    print("\nFull Results Dictionary:")
    #    print(analysis_results)