from statsmodels.tsa.seasonal import STL
from statsmodels.nonparametric import _smoothers_lowess
from statsmodels.regression.linear_model import OLS
from statsmodels.tools.tools import add_constant
import statsmodels.api as sm
from typing import Dict, List
from math import floor
from ruptures.base import BaseEstimator
from ruptures.costs import cost_factory
import numpy as np
import matplotlib.pyplot as plt
from math import floor

from ruptures.costs import cost_factory
from ruptures.base import BaseCost, BaseEstimator


class Pelt(BaseEstimator):

    """Penalized change point detection.

    For a given model and penalty level, computes the segmentation which minimizes the constrained
    sum of approximation errors.

    """

    def __init__(self, model="l2", custom_cost=None, min_size=2, jump=5, params=None):
        """Initialize a Pelt instance.

        Args:
            model (str, optional): segment model, ["l1", "l2", "rbf"]. Not used if ``'custom_cost'`` is not None.
            custom_cost (BaseCost, optional): custom cost function. Defaults to None.
            min_size (int, optional): minimum segment length.
            jump (int, optional): subsample (one every *jump* points).
            params (dict, optional): a dictionary of parameters for the cost instance.

        Returns:
            self
        """
        if custom_cost is not None and isinstance(custom_cost, BaseCost):
            self.cost = custom_cost
        else:
            if params is None:
                self.cost = cost_factory(model=model)
            else:
                self.cost = cost_factory(model=model, **params)
        self.min_size = max(min_size, self.cost.min_size)
        self.jump = jump
        self.n_samples = None


    def _seg(self, pen):
        """Computes the segmentation for a given penalty using PELT (or a list
        of penalties).

        Args:
            penalty (float): penalty value

        Returns:
            dict: partition dict {(start, end): cost value,...}
        """

        # initialization
        # partitions[t] contains the optimal partition of signal[0:t]
        partitions = dict()  # this dict will be recursively filled
        partitions[0] = {(0, 0): 0}
        admissible = []

        # Recursion
        ind = [
            k for k in range(0, self.n_samples, self.jump) if k >= self.min_size]
        ind += [self.n_samples]
        for bkp in ind:
            # adding a point to the admissible set from the previous loop.
            new_adm_pt = floor((bkp - self.min_size) / self.jump)
            new_adm_pt *= self.jump
            admissible.append(new_adm_pt)

            subproblems = list()
            for t in admissible:
                # left partition
                try:
                    tmp_partition = partitions[t].copy()
                except KeyError:  # no partition of 0:t exists
                    continue
                # we update with the right partition
                tmp_partition.update({(t, bkp): self.cost.error(t, bkp) + pen})
                subproblems.append(tmp_partition)

            # finding the optimal partition
            partitions[bkp] = min(
                subproblems, key=lambda d: sum(d.values()))
            # trimming the admissible set
            admissible = [t for t, partition in
                          zip(admissible, subproblems) if
                          sum(partition.values()) <=
                          sum(partitions[bkp].values()) + pen]

        best_partition = partitions[self.n_samples]
        del best_partition[(0, 0)]
        return best_partition

    def fit(self, signal):
        """Set params.

        Args:
            signal (array): signal to segment. Shape (n_samples, n_features) or (n_samples,).

        Returns:
            self
        """
        # update params
        self.cost.fit(signal)
        if signal.ndim == 1:
            n_samples, = signal.shape
        else:
            n_samples, _ = signal.shape
        self.n_samples = n_samples
        return self


    def predict(self, pen):
        """Return the optimal breakpoints.

        Must be called after the fit method. The breakpoints are associated with the signal passed
        to fit().

        Args:
            pen (float): penalty value (>0)

        Returns:
            list: sorted list of breakpoints
        """
        partition = self._seg(pen)
        bkps = sorted(e for s, e in partition.keys())
        return bkps


    def fit_predict(self, signal, pen):
        """Fit to the signal and return the optimal breakpoints.

        Helper method to call fit and predict once

        Args:
            signal (array): signal. Shape (n_samples, n_features) or (n_samples,).
            pen (float): penalty value (>0)

        Returns:
            list: sorted list of breakpoints
        """
        self.fit(signal)
        return self.predict(pen)



class STLFeatures:
    """
    Calculates seasonal trend using loess decomposition.
    """
    def __init__(self, freq: int = 1):
        """
        Initialize the STLFeatures class.

        Args:
            freq (int, optional): Frequency of the time series. Defaults to 1.
        """
        self.freq = freq

    def get_features(self, x: np.ndarray) -> Dict[str, float]:
        """
        Calculates STL features.

        Args:
            x (numpy array): The time series.

        Returns:
        dict: A dictionary containing the STL features.
        """
        m = self.freq
        nperiods = int(m > 1)
        # STL fits
        if m > 1:
            try:
                stlfit = STL(x, period=m).fit()
            except:
                output = {
                    'nperiods': nperiods,
                    'seasonal_period': m,
                    'trend': np.nan,
                    'spike': np.nan,
                    'linearity': np.nan,
                    'curvature': np.nan,
                    'e_acf1': np.nan,
                    'e_acf10': np.nan,
                    'seasonal_strength': np.nan,
                    'peak': np.nan,
                    'trough': np.nan
                }

                return output

            trend0 = stlfit.trend
            remainder = stlfit.resid
            seasonal = stlfit.seasonal
        else:
            deseas = x
            t = np.arange(len(x)) + 1
            try:
                trend0 = _smoothers_lowess().fit(t, deseas).predict(t)
            except:
                output = {
                    'nperiods': nperiods,
                    'seasonal_period': m,
                    'trend': np.nan,
                    'spike': np.nan,
                    'linearity': np.nan,
                    'curvature': np.nan,
                    'e_acf1': np.nan,
                    'e_acf10': np.nan
                }

                return output

            remainder = deseas - trend0
            seasonal = np.zeros(len(x))
        # De-trended and de-seasonalized data
        detrend = x - trend0
        deseason = x - seasonal
        fits = x - remainder
        # Summay stats
        n = len(x)
        varx = np.nanvar(x, ddof=1)
        vare = np.nanvar(remainder, ddof=1)
        vardetrend = np.nanvar(detrend, ddof=1)
        vardeseason = np.nanvar(deseason, ddof=1)
        # Measure of trend strength
        if varx < np.finfo(float).eps:
            trend = 0
        elif (vardeseason / varx < 1e-10):
            trend = 0
        else:
            trend = max(0, min(1, 1 - vare / vardeseason))
        # Measure of seasonal strength
        if m > 1:
            if varx < np.finfo(float).eps:
                season = 0
            elif np.nanvar(remainder + seasonal, ddof=1) < np.finfo(float).eps:
                season = 0
            else:
                season = max(0, min(1, 1 - vare / np.nanvar(remainder + seasonal, ddof=1)))

            peak = (np.argmax(seasonal) + 1) % m
            peak = m if peak == 0 else peak

            trough = (np.argmin(seasonal) + 1) % m
            trough = m if trough == 0 else trough
        # Compute measure of spikiness
        d = (remainder - np.nanmean(remainder)) ** 2
        varloo = (vare * (n - 1) - d) / (n - 2)
        spike = np.nanvar(varloo, ddof=1)
        # Compute measures of linearity and curvature
        time = np.arange(n) + 1
        poly_m = poly(time, 2)
        time_x = add_constant(poly_m)
        coefs = OLS(trend0, time_x).fit().params

        try:
            linearity = coefs[1]
        except:
            linearity = np.nan
        try:
            curvature = -coefs[2]
        except:
            curvature = np.nan
        # ACF features
        acf_obj = ACF_Features(freq=m) #instantiate the ACF_Features class
        acfremainder = acf_obj.get_features(remainder)
        # Assemble features
        output = {
            'nperiods': nperiods,
            'seasonal_period': m,
            'trend': trend,
            'spike': spike,
            'linearity': linearity,
            'curvature': curvature,
            'e_acf1': acfremainder['x_acf1'],
            'e_acf10': acfremainder['x_acf10']
        }

        if m > 1:
            output['seasonal_strength'] = season
            output['peak'] = peak
            output['trough'] = trough

        return output

def poly(t: np.ndarray, degree: int) -> np.ndarray:
    """Computes a polynomial of a given degree.

    Parameters
    ----------
    t: numpy array
        The time series.
    degree: int
        Degree of the polynomial

    Returns
    -------
    numpy array
        Polynomial of given degree
    """
    n = len(t)
    T = np.zeros((n, degree + 1))
    for i in range(degree + 1):
        T[:, i] = t ** i
    return T



class ACF_Features:
    """
    Calculates the autocorrelation coefficients
    """
    def __init__(self, freq: int = 1):
        """
        Initialize the ACF_Features class.

        Args:
            freq (int, optional): Frequency of the time series. Defaults to 1.
        """
        self.freq = freq

    def get_features(self, x: np.ndarray) -> Dict[str, float]:
        """
        Calculates the autocorrelation coefficients

        Parameters
        ----------
        x: numpy array
            The time series.

        Returns
        -------
        dict
            'x_acf1': First autocorrelation coefficient
            'x_acf10': Sum of the first 10 autocorrelation coefficients
        """
        from statsmodels.tsa.stattools import acf
        acf_values = acf(x, nlags=10)
        acf1 = acf_values[1]
        acf10 = acf_values[1:11].sum()
        return {'x_acf1': acf1, 'x_acf10': acf10}



class CrossingPoints:
    """
    Calculates the number of times a time series crosses its median.
    """
    def __init__(self, freq: int = 1):
        """
        Initialize the CrossingPoints class.

        Args:
            freq (int, optional): Frequency of the time series. Defaults to 1.
        """
        self.freq = freq
    def get_features(self, x: np.ndarray) -> Dict[str, float]:
        """Crossing points.

        Parameters
        ----------
        x: numpy array
            The time series.

        Returns
        -------
        dict
            'crossing_points': Number of times that x crosses the median.
        """
        midline = np.median(x)
        ab = x <= midline
        lenx = len(x)
        p1 = ab[:(lenx - 1)]
        p2 = ab[1:]
        cross = (p1 & (~p2)) | (p2 & (~p1))

        return {'crossing_points': cross.sum()}

class LinearRegression:
    """
    Performs linear regression of a time series against time using statsmodels OLS,
    mimicking the attribute structure of scikit-learn's LinearRegression.

    Fits a model y = intercept + coef * t, where t is the time step index.

    Attributes:
        coef_ (float): The coefficient (slope) of the time variable.
        intercept_ (float): The intercept of the regression line.
        results_ (statsmodels.regression.linear_model.RegressionResults):
            The full results object from statsmodels OLS fit. Contains
            detailed statistics (R-squared, p-values, standard errors, etc.).
        n_features_in_ (int): Number of features seen during fit (always 1 for time).
        _time_steps (np.ndarray): Internal storage of time steps used for fitting.
        _time_series (np.ndarray): Internal storage of the time series used for fitting.
    """

    def __init__(self):
        """Initializes the TimeLinearRegression model."""
        self.coef_ = None
        self.intercept_ = None
        self.results_ = None
        self.n_features_in_ = 1 # Regression against time (1 feature)
        self._time_steps = None
        self._time_series = None

    def fit(self, time_series: np.ndarray) -> 'TimeLinearRegression':
        """
        Fits the linear regression model to the time series.

        Args:
            time_series (np.ndarray): The input time series data (1-dimensional).

        Returns:
            self: The fitted TimeLinearRegression instance.

        Raises:
            ValueError: If the input time_series is not convertible to a 1D numpy array
                        or has less than 2 data points.
        """
        # --- Input validation ---
        if not isinstance(time_series, np.ndarray):
            try:
                time_series = np.array(time_series, dtype=float)
            except Exception as e:
                raise ValueError(f"Input time_series could not be converted to a numpy array: {e}")

        if time_series.ndim != 1:
            raise ValueError("Input time_series must be a 1D array.")
        if len(time_series) < 2:
            raise ValueError("Input time_series must have at least two data points for linear regression.")
        # --- End Validation ---

        self._time_series = time_series # Store original series
        self._time_steps = np.arange(len(time_series)) # Create time variable X

        # Add a constant (intercept) term to the independent variable matrix
        # Reshape time_steps to be a column vector for add_constant if needed,
        # though add_constant usually handles 1D correctly.
        time_with_constant = sm.add_constant(self._time_steps)

        # Perform the linear regression using Ordinary Least Squares (OLS)
        model = sm.OLS(self._time_series, time_with_constant)
        self.results_: RegressionResults = model.fit() # Store full results

        # Extract intercept and slope (coefficient for time)
        self.intercept_ = self.results_.params[0]
        self.coef_ = self.results_.params[1] # Coef for the time variable

        return self # Return the fitted instance

    def predict(self, time_steps: np.ndarray = None) -> np.ndarray:
        """
        Predicts time series values for given time steps using the fitted model.

        Args:
            time_steps (np.ndarray, optional): Array of time steps (indices) to predict for.
                If None, predicts for the time steps used during fitting. Defaults to None.

        Returns:
            np.ndarray: The predicted values.

        Raises:
            RuntimeError: If the model has not been fitted yet.
            ValueError: If time_steps is not a 1D array or convertible to one.
        """
        if self.results_ is None:
            raise RuntimeError("Model has not been fitted yet. Call fit() first.")

        if time_steps is None:
            # Predict on the original time steps used for fitting
            time_input = self._time_steps
        else:
             # --- Input validation ---
            if not isinstance(time_steps, np.ndarray):
                try:
                    time_steps = np.array(time_steps, dtype=float)
                except Exception as e:
                    raise ValueError(f"Input time_steps could not be converted to a numpy array: {e}")
            if time_steps.ndim != 1:
                raise ValueError("Input time_steps must be a 1D array.")
            time_input = time_steps
             # --- End Validation ---


        # Add constant for prediction
        time_input_with_constant = sm.add_constant(time_input)
        return self.results_.predict(time_input_with_constant)

    def score(self) -> float:
        """
        Returns the R-squared ($R^2$) coefficient of determination for the fit
        on the training data.

        Returns:
            float: The R-squared value.

        Raises:
            RuntimeError: If the model has not been fitted yet.
        """
        if self.results_ is None:
            raise RuntimeError("Model has not been fitted yet. Call fit() first.")

        # R-squared is directly available from statsmodels results
        return self.results_.rsquared

    def plot_fit(self, figsize=(10, 6)):
        """
        Generates a plot showing the original data and the fitted line.

        Args:
            figsize (tuple, optional): Figure size for the plot. Defaults to (10, 6).

        Raises:
            RuntimeError: If the model has not been fitted yet.
        """
        if self.results_ is None:
            raise RuntimeError("Model has not been fitted yet. Call fit() first.")

        predicted_values = self.predict() # Predict on original time steps

        plt.figure(figsize=figsize)
        plt.scatter(self._time_steps, self._time_series, label='Original Data Points', marker='o', s=20, alpha=0.7)
        plt.plot(self._time_steps, predicted_values, color='red', linewidth=2, label=f'Fitted Line (Slope={self.coef_:.4f})')

        plt.xlabel("Time Step")
        plt.ylabel("Time Series Value")
        plt.title("Linear Regression of Time Series vs. Time")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    def summary(self):
        """Prints the detailed summary from statsmodels OLS results."""
        if self.results_ is None:
            raise RuntimeError("Model has not been fitted yet. Call fit() first.")
        print(self.results_.summary())