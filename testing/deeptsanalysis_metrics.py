import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import json
from datasetsforecast.m3 import M3
from src.metrics import Metrics
import sys
import os

# --- Path Setup ---
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
    print(f"Added to sys.path: {PROJECT_ROOT}")

def test_deeptsanalysis_metrics():
    """
    Analyzes M3 monthly data using the Metrics class.
    Includes Trend Changes, Trend Strength, Median Crosses,
        Linear Regression Slope, Linear Regression R2,
        Entropy Pairs, Series Forecastability and Series Fluctuation.
    Returns a dictionary of extreme values found for each feature.
    """

    data_dir = os.path.join(PROJECT_ROOT, 'data', 'm3_download')
    print(f"\nLoading M3 Monthly data (will download to '{data_dir}' if needed)...")
    os.makedirs(data_dir, exist_ok=True)

    loaded_data = M3.load(directory=data_dir, group='Monthly')
    Y_df = None
    if isinstance(loaded_data, (list, tuple)) and len(loaded_data) > 0:
        Y_df = loaded_data[0]
    elif isinstance(loaded_data, pd.DataFrame):
        Y_df = loaded_data

    if Y_df is None: raise ValueError("Failed to extract DataFrame from loaded data.")

    print("Data loaded successfully.")
    required_cols = ['unique_id', 'ds', 'y']
    if not all(col in Y_df.columns for col in required_cols):
        raise ValueError(f"Error: Missing required columns: {[c for c in required_cols if c not in Y_df.columns]}")
    Y_df['ds'] = pd.to_datetime(Y_df['ds'])


    all_dataset_ids = Y_df['unique_id'].unique()
    results_data = []

    print(f"\nAnalyzing {len(all_dataset_ids)} monthly time series using the Metrics class...")

    processed_count = 0
    skipped_count = 0

    for i, dataset_id in enumerate(all_dataset_ids):
        df_series = Y_df[Y_df['unique_id'] == dataset_id].sort_values('ds')
        series_pd = df_series['y'].reset_index(drop=True)

        if series_pd.isnull().any() or not np.isfinite(series_pd).all():
            skipped_count += 1
            continue
        if series_pd.nunique() <= 1:
            skipped_count += 1
            continue

        series_np = series_pd.to_numpy()

        processed_count += 1
        current_features = {
            "unique_id": dataset_id,
            "Trend Changes": np.nan,
            "Trend Strength": np.nan,
            "Median Crosses": np.nan,
            "Linear Regression Slope": np.nan,
            "Linear Regression R2": np.nan,
            "Entropy Pairs": np.nan,
            "Series Forecastability": np.nan,
            "Series Fluctuation": np.nan,
            "Autocorrelation Relevance": np.nan,
            "Seasonal Strength": np.nan,
            "Window Fluctuation": np.nan,
            "Short-Term Variation": np.nan,
            "Autocorrelation": np.nan,
            "Differenced Series": np.nan,
            "Series Complexity": np.nan,
            "Records Concentration": np.nan,
            "Series Centroid": np.nan
        }

        metrics_instance = Metrics(series_np)
        current_features["Trend Changes"] = metrics_instance.trend_changes()
        current_features["Trend Strength"] = metrics_instance.trend_strength()
        current_features["Median Crosses"] = metrics_instance.median_crosses()
        current_features["Linear Regression Slope"] = metrics_instance.linear_regression_slope()
        current_features["Linear Regression R2"] = metrics_instance.linear_regression_r2()
        current_features["Entropy Pairs"] = metrics_instance.entropy_pairs()
        current_features["Series Forecastability"] = metrics_instance.forecastability(sf=1)
        current_features["Series Fluctuation"] = metrics_instance.fluctuation()
        current_features["Autocorrelation Relevance"] = metrics_instance.ac_relevance()
        current_features["Seasonal Strength"] = metrics_instance.seasonal_strength()
        current_features["Window Fluctuation"] = metrics_instance.window_fluctuation()
        current_features["Short-Term Variation"] = metrics_instance.st_variation()
        current_features["Autocorrelation"] = metrics_instance.ac()
        current_features["Differenced Series"] = metrics_instance.diff_series()
        current_features["Series Complexity"] = metrics_instance.complexity()
        current_features["Records Concentration"] = metrics_instance.rec_concentration()
        current_features["Series Centroid"] = metrics_instance.centroid(fs=12)

        results_data.append(current_features)

        if (processed_count) % 100 == 0 or (i + 1) == len(all_dataset_ids) or processed_count == 1:
            print(f"Processed {processed_count}/{len(all_dataset_ids)-skipped_count} series... (Checked {i + 1}, Skipped {skipped_count})")

    print(f"\nAnalysis loop complete. Features computed for {processed_count} series. {skipped_count} series skipped.")

    if not results_data:
        print("No data was processed successfully.")
        return {}

    results_df = pd.DataFrame(results_data)

    print("\n--- Calculation Summary ---")
    if not results_df.empty:
        print(results_df.info())
        print("\n--- Metrics Descriptive Statistics ---")
        with pd.option_context('display.float_format', '{:,.4f}'.format, 'display.max_rows', None, 'display.max_columns', None, 'display.width', 1000):
            print(results_df.describe())
        print("\n--- Sample of Results DataFrame ---")
        print(results_df.head())
    else:
        print("Results DataFrame is empty.")


    print("\n--- Metrics Extreme Values per Feature ---")
    extreme_results = {}
    if not results_df.empty:
        for feature_name in results_df.select_dtypes(include=np.number).columns:
            valid_series = results_df[feature_name].dropna()
            if valid_series.empty or not np.all(np.isfinite(valid_series)):
                print(f"\nFeature: {feature_name}\n    No valid finite data found.")
                extreme_results[feature_name] = {'lowest': (None, np.nan), 'highest': (None, np.nan)}
                continue
            idx_min, val_min = valid_series.idxmin(), valid_series.min()
            idx_max, val_max = valid_series.idxmax(), valid_series.max()
            actual_low_id = results_df.loc[idx_min, 'unique_id']
            actual_high_id = results_df.loc[idx_max, 'unique_id']
            print(f"\nFeature: {feature_name}")
            print(f"    Lowest:  ID = {actual_low_id}, Value = {val_min:.4f}")
            print(f"    Highest: ID = {actual_high_id}, Value = {val_max:.4f}")
            extreme_results[feature_name] = {'lowest': (actual_low_id, val_min), 'highest': (actual_high_id, val_max)}

    else:
        print("Cannot calculate extreme values as results DataFrame is empty.")


    return extreme_results

def save_extreme_values_table_as_image(extreme_results, output_filename="extreme_values_summary.png"):
    """
    Creates a table image from the extreme_results dictionary and saves it.

    """
    if not extreme_results:
        print("No extreme results to generate table image.")
        return

    data_for_df = []
    for feature, extremes in extreme_results.items():
        id_min, val_min = extremes.get('lowest', (None, np.nan))
        id_max, val_max = extremes.get('highest', (None, np.nan))
        data_for_df.append({
            "Feature Name": feature,
            "Lowest Value": f"{val_min:.4f}" if pd.notna(val_min) else "N/A",
            "Lowest ID": str(id_min) if pd.notna(id_min) else "N/A",
            "Highest Value": f"{val_max:.4f}" if pd.notna(val_max) else "N/A",
            "Highest ID": str(id_max) if pd.notna(id_max) else "N/A",
        })

    if not data_for_df:
        print("No data processed for the table image.")
        return

    df = pd.DataFrame(data_for_df)

    fig, ax = plt.subplots(figsize=(12, max(4, len(df) * 0.5)))
    ax.axis('tight')
    ax.axis('off')

    fig.suptitle("Extreme Feature Values Summary", fontsize=16, y=0.95)


    table = ax.table(cellText=df.values,
                     colLabels=df.columns,
                     cellLoc='center',
                     loc='center',
                     colColours=["#f2f2f2"] * len(df.columns))

    table.set_fontsize(10)
    table.scale(1.2, 1.2) 

    for (i, j), cell in table.get_celld().items():
        if i == 0: 
            cell.set_text_props(weight='bold')

    plt.tight_layout(pad=1.5)

    if os.path.isabs(output_filename):
        full_output_path = output_filename
    else:
        full_output_path = os.path.join(PROJECT_ROOT, output_filename)

    try:
        plt.savefig(full_output_path, bbox_inches='tight', dpi=200)
        print(f"\nTable image saved to: {full_output_path}")
    except Exception as e:
        print(f"Error saving table image: {e}")
    finally:
        plt.close(fig)

class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer): return int(obj)
        if isinstance(obj, np.floating): return float(obj)
        if isinstance(obj, np.ndarray): return obj.tolist()
        if isinstance(obj, float) and np.isnan(obj): return None
        if pd.isna(obj): return None
        return super(NpEncoder, self).default(obj)

if __name__ == "__main__":
    analysis_results = {}
    analysis_results = test_deeptsanalysis_metrics()


    print("\n--- Script Finished ---")
    if analysis_results:
        print("Results dictionary was generated (extreme values shown above).")
        image_output_name = "metrics_table.png"
        save_extreme_values_table_as_image(analysis_results, output_filename=image_output_name)
    else:
        print("Analysis did not generate results or failed.")