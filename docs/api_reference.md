# API Reference

## Introduction

This document provides a detailed reference for the `ete_ts` library API. It covers installation, basic usage, and a complete guide to all available functions for time-series feature extraction.

## Installation

Install the library from PyPI using pip:

```bash
pip install ete_ts
```

## Basic Usage

To get started, import the library and a numerical package like NumPy. Then you can call any feature-extraction function with your time-series data.

```python
import ete_ts
import numpy as np

# Create a sample time-series
my_series = np.array([1, 2, 4, 8, 12, 18, 25, 33, 45, 58])

# Calculate a feature
slope = ete_ts.linear_regression_slope(my_series)

print(f"The slope of the series is: {slope}")
```

## Exceptions

The library uses standard Python exceptions to report errors. The most common are:
* `ValueError`: Raised when a function receives an argument of the correct type but an inappropriate value (e.g., a time-series that is too short for a given calculation).
* `TypeError`: Raised when a function receives an argument of the wrong type (e.g., a list where a NumPy array is expected).

---

## Functions Reference

#### trend_strength(series, period=1, seasonal=7, robust=False)
```python
strength = ete_ts.trend_strength(my_series)
```

#### trend_changes(series, model="l2", min_size=2, jump=5, params=None, custom_cost=None)
```python
changes = ete_ts.trend_changes(my_series)
```

#### linear_regression_slope(series)
```python
slope = ete_ts.linear_regression_slope(my_series)
```

#### linear_regression_r2(series)
```python
r2_score = ete_ts.linear_regression_r2(my_series)
```

#### forecastability(series, sf, method="welch", nperseg=None, normalize=False)
```python
entropy = ete_ts.forecastability(my_series, sf=1.0)
```

#### fluctuation(series)
```python
fluc = ete_ts.fluctuation(my_series)
```

#### window_fluctuation(series)
```python
win_fluc = ete_ts.window_fluctuation(my_series)
```

#### seasonal_strength(series, period=1, seasonal=7, robust=False)
```python
season_str = ete_ts.seasonal_strength(my_series)
```

#### ac_relevance(series)
```python
relevance = ete_ts.ac_relevance(my_series)
```

#### st_variation(series)
```python
variation = ete_ts.st_variation(my_series)
```

#### diff_series(series)
```python
diff_acf = ete_ts.diff_series(my_series)
```

#### complexity(series)
```python
comp = ete_ts.complexity(my_series)
```

#### rec_concentration(series)
```python
concentration = ete_ts.rec_concentration(my_series)
```

#### centroid(series, fs=1)
```python
spec_centroid = ete_ts.centroid(my_series, fs=1)
```

#### info()
```python
ete_ts.info()
```

#### all_metrics(series, fs=1)
```python
outputs = ete_ts.all_metrics(my_series, fs=1)
```
    