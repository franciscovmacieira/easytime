# API Reference

## Introduction

This document provides a detailed reference for the `easytime` library API. It covers installation, basic usage, and a complete guide to all available functions for time-series feature extraction.

## Installation

Install the library from PyPI using pip:

```bash
pip install easytime
```

## Basic Usage

To get started, import the library and a numerical package like NumPy. Then you can call any feature-extraction function with your time-series data.

```python
import easytime
import numpy as np

# Create a sample time-series
my_series = np.array([1, 2, 4, 8, 12, 18, 25, 33, 45, 58])

# Calculate a feature
slope = easytime.linear_regression_slope(my_series)

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
strength = easytime.trend_strength(my_series)
```

#### trend_changes(series, model="l2", min_size=2, jump=5, params=None, custom_cost=None)
```python
changes = easytime.trend_changes(my_series)
```

#### linear_regression_slope(series)
```python
slope = easytime.linear_regression_slope(my_series)
```

#### linear_regression_r2(series)
```python
r2_score = easytime.linear_regression_r2(my_series)
```

#### forecastability(series, sf, method="welch", nperseg=None, normalize=False)
```python
entropy = easytime.forecastability(my_series, sf=1.0)
```

#### fluctuation(series)
```python
fluc = easytime.fluctuation(my_series)
```

#### window_fluctuation(series)
```python
win_fluc = easytime.window_fluctuation(my_series)
```

#### seasonal_strength(series, period=1, seasonal=7, robust=False)
```python
season_str = easytime.seasonal_strength(my_series)
```

#### ac_relevance(series)
```python
relevance = easytime.ac_relevance(my_series)
```

#### st_variation(series)
```python
variation = easytime.st_variation(my_series)
```

#### diff_series(series)
```python
diff_acf = easytime.diff_series(my_series)
```

#### complexity(series)
```python
comp = easytime.complexity(my_series)
```

#### rec_concentration(series)
```python
concentration = easytime.rec_concentration(my_series)
```

#### centroid(series, fs=1)
```python
spec_centroid = easytime.centroid(my_series, fs=1)
```

#### info()
```python
easytime.info()
```

#### all_metrics(series, fs=1)
```python
outputs = easytime.all_metrics(my_series, fs=1)
```
    