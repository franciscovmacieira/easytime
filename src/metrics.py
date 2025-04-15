from statsmodels.tsa.seasonal import STL
from statsmodels.nonparametric import _smoothers_lowess
from statsmodels.regression.linear_model import OLS
from statsmodels.tools.tools import add_constant
from typing import Dict, List
from math import floor
from ruptures.base import BaseEstimator
from ruptures.costs import cost_factory
import numpy as np


class Pelt(BaseEstimator):
    """
    Penalized change point detection.

    For a given model and penalty level, computes the segmentation which minimizes the constrained
    sum of approximation errors.
    """

    def __init__(self, model="l2", custom_cost=None, min_size=2, jump=5, params=None):
        """
        Initialize a Pelt instance.

        Args:
            model (str, optional): segment model, ["l1", "l2", "rbf"].
                Not used if ``'custom_cost'`` is not None.
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

    def _seg(self, pen: float) -> Dict[str, List[int]]:
        """
        Computes the segmentation for a given penalty using PELT.

        Args:
            pen (float): penalty value

        Returns:
        dict: A dictionary containing the breakpoints.  For consistency,
                we return a dictionary.
        """
        # initialization
        partitions = dict()
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
        del best_partition[(0, 0)]  # Remove the initial (0,0) "breakpoint"
        bkps = sorted(e for s, e in best_partition.keys())

        return {'breakpoints': bkps}  # Changed to return a dict

    def fit(self, signal: np.ndarray) -> 'Pelt':
        """
        Set params.

        Args:
            signal (array): signal to segment.
                Shape (n_samples, n_features) or (n_samples,).

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

    def predict(self, pen: float) -> List[int]:
        """
        Return the optimal breakpoints.

        Must be called after the fit method.
        The breakpoints are associated with the signal passed to fit().

        Args:
            pen (float): penalty value (>0)

        Returns:
            list: sorted list of breakpoints
        """
        partition = self._seg(pen)
        return partition['breakpoints']  # changed to return the breakpoints list

    def fit_predict(self, signal: np.ndarray, pen: float) -> List[int]:
        """
        Fit to the signal and return the optimal breakpoints.

        Helper method to call fit and predict once

        Args:
            signal (array): signal. Shape (n_samples, n_features) or (n_samples,).
            pen (float): penalty value (>0)

        Returns:
            list: sorted list of breakpoints
        """
        self.fit(signal)
        return self.predict(pen)

    def get_features(self, signal: np.ndarray, pen: float) -> Dict[str, List[int]]:
        """
        Calculate Pelt features (breakpoints).  This is the key change
        to make Pelt consistent with other feature extractors.

        Args:
            signal (np.ndarray): The input time series signal.
            pen (float): Penalty parameter for PELT.

        Returns:
            Dict[str, List[int]]: A dictionary with a single key
            'breakpoints' and the corresponding list of breakpoints.
        """
        self.fit(signal)  # Ensure the model is fitted.
        breakpoints = self.predict(pen)  # Get the breakpoints
        return {'breakpoints': breakpoints}  # Return in the desired format



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
                stlfit = STL(x, m, 13).fit()
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
    Performs linear regression on time series data.

    This class calculates the linear regression of a time series against time
    (i.e., it fits a line to the data) and returns the slope, intercept,
    and R-squared value.
    """

    def get_features(self, time_series: np.ndarray) -> Dict[str, float]:
        """
        Calculates linear regression features for a time series.

        Args:
            time_series (np.ndarray): The input time series data.

        Returns:
            Dict[str, float]: A dictionary containing the slope, intercept,
                and R-squared value of the regression.
        """
        # Create the time variable (independent variable)
        time = np.arange(len(time_series))
        time_with_constant = add_constant(time)  # Add a constant for the intercept

        # Perform the linear regression
        model = OLS(time_series, time_with_constant).fit()

        # Extract the parameters
        slope = model.params[1]
        intercept = model.params[0]
        r_squared = model.rsquared

        return {
            "slope": slope,
            "intercept": intercept,
            "r_squared": r_squared,
        }