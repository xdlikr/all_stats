import numpy as np
import pandas as pd
from scipy import stats
from typing import Dict, List, Any, Optional
import warnings

def fit_distribution(data: pd.Series, distribution_name: str, **kwargs) -> Dict[str, Any]:
    """
    Fit a specific distribution to data.
    
    Args:
        data: Input data series
        distribution_name: Name of distribution to fit
        **kwargs: Additional parameters for distribution fitting
        
    Returns:
        Dictionary with distribution fit results
    """
    clean_data = data.dropna()
    n = len(clean_data)
    
    if n < 2:
        return {
            "success": False,
            "message": "Insufficient data for distribution fitting",
            "n": n
        }
    
    result = {
        "distribution": distribution_name,
        "n": n,
        "success": False,
        "params": {},
        "log_likelihood": None,
        "aic": None,
        "bic": None,
        "ks_statistic": None,
        "ks_pvalue": None,
        "ad_statistic": None,
        "ad_pvalue": None
    }
    
    try:
        if distribution_name == "normal":
            # Fit normal distribution
            params = stats.norm.fit(clean_data)
            result["params"] = {"loc": params[0], "scale": params[1]}
            
        elif distribution_name == "lognormal":
            # Fit lognormal distribution (log transform first)
            log_data = np.log(clean_data[clean_data > 0])
            if len(log_data) < 2:
                result["message"] = "Insufficient positive values for lognormal fit"
                return result
            
            params = stats.norm.fit(log_data)
            result["params"] = {"mu": params[0], "sigma": params[1]}
            
        elif distribution_name == "weibull":
            # Fit Weibull distribution
            params = stats.weibull_min.fit(clean_data, floc=0)  # Fix location at 0
            result["params"] = {"c": params[0], "scale": params[2]}
            
        elif distribution_name == "gamma":
            # Fit Gamma distribution
            params = stats.gamma.fit(clean_data, floc=0)  # Fix location at 0
            result["params"] = {"a": params[0], "scale": params[2]}
            
        elif distribution_name == "logistic":
            # Fit Logistic distribution
            params = stats.logistic.fit(clean_data)
            result["params"] = {"loc": params[0], "scale": params[1]}
            
        elif distribution_name == "exponential":
            # Fit Exponential distribution
            params = stats.expon.fit(clean_data, floc=0)  # Fix location at 0
            result["params"] = {"scale": params[1]}
            
        else:
            result["message"] = f"Unsupported distribution: {distribution_name}"
            return result
        
        # Calculate log-likelihood
        if distribution_name == "lognormal":
            log_likelihood = stats.norm.logpdf(log_data, *params).sum()
        else:
            # Map distribution names to scipy.stats names
            dist_mapping = {
                "normal": stats.norm,
                "weibull": stats.weibull_min,
                "gamma": stats.gamma,
                "logistic": stats.logistic,
                "exponential": stats.expon
            }
            dist = dist_mapping.get(distribution_name)
            if dist is None:
                raise ValueError(f"Unknown distribution: {distribution_name}")
            log_likelihood = dist.logpdf(clean_data, *params).sum()
        
        result["log_likelihood"] = log_likelihood
        
        # Calculate AIC and BIC
        k = len(result["params"])
        result["aic"] = 2 * k - 2 * log_likelihood
        result["bic"] = k * np.log(n) - 2 * log_likelihood
        
        # Goodness-of-fit tests
        if distribution_name == "lognormal":
            fitted_dist = stats.norm(*params)
            test_data = log_data
        else:
            fitted_dist = dist(*params)
            test_data = clean_data
        
        # Kolmogorov-Smirnov test
        ks_result = stats.kstest(test_data, fitted_dist.cdf)
        result["ks_statistic"] = ks_result.statistic
        result["ks_pvalue"] = ks_result.pvalue
        
        # Anderson-Darling test (for normal distribution)
        if distribution_name == "normal":
            ad_result = stats.anderson(test_data, dist='norm')
            result["ad_statistic"] = ad_result.statistic
        
        result["success"] = True
        result["message"] = "Distribution fitted successfully"
        
    except Exception as e:
        result["message"] = f"Error fitting distribution: {str(e)}"
    
    return result

def compare_distributions(data: pd.Series, distributions: List[str] = None) -> Dict[str, Any]:
    """
    Compare multiple distribution fits to data.
    
    Args:
        data: Input data series
        distributions: List of distribution names to compare
        
    Returns:
        Dictionary with comparison results
    """
    if distributions is None:
        distributions = ["normal", "lognormal", "weibull", "gamma", "logistic"]
    
    results = {}
    comparison_table = []
    
    for dist_name in distributions:
        fit_result = fit_distribution(data, dist_name)
        results[dist_name] = fit_result
        
        if fit_result["success"]:
            comparison_table.append({
                "Distribution": dist_name,
                "AIC": fit_result["aic"],
                "BIC": fit_result["bic"],
                "Log-Likelihood": fit_result["log_likelihood"],
                "KS Statistic": fit_result["ks_statistic"],
                "KS p-value": fit_result["ks_pvalue"],
                "Parameters": fit_result["params"]
            })
    
    # Sort by AIC (lower is better)
    if comparison_table:
        comparison_table.sort(key=lambda x: x["AIC"] if x["AIC"] is not None else float('inf'))
        best_dist = comparison_table[0]["Distribution"]
    else:
        best_dist = None
    
    return {
        "results": results,
        "comparison_table": comparison_table,
        "best_distribution": best_dist,
        "n_distributions": len(comparison_table)
    }

def calculate_quantile(distribution_name: str, params: Dict[str, float], p: float) -> float:
    """
    Calculate quantile for a fitted distribution.
    
    Args:
        distribution_name: Name of distribution
        params: Distribution parameters
        p: Probability (0-1)
        
    Returns:
        Quantile value
    """
    if distribution_name == "normal":
        return stats.norm.ppf(p, loc=params["loc"], scale=params["scale"])
    
    elif distribution_name == "lognormal":
        return np.exp(stats.norm.ppf(p, loc=params["mu"], scale=params["sigma"]))
    
    elif distribution_name == "weibull":
        return stats.weibull_min.ppf(p, c=params["c"], scale=params["scale"])
    
    elif distribution_name == "gamma":
        return stats.gamma.ppf(p, a=params["a"], scale=params["scale"])
    
    elif distribution_name == "logistic":
        return stats.logistic.ppf(p, loc=params["loc"], scale=params["scale"])
    
    elif distribution_name == "exponential":
        return stats.expon.ppf(p, scale=params["scale"])
    
    else:
        raise ValueError(f"Unsupported distribution: {distribution_name}")

def calculate_cdf(distribution_name: str, params: Dict[str, float], x: float) -> float:
    """
    Calculate CDF for a fitted distribution.
    
    Args:
        distribution_name: Name of distribution
        params: Distribution parameters
        x: Value to evaluate
        
    Returns:
        Cumulative probability
    """
    if distribution_name == "normal":
        return stats.norm.cdf(x, loc=params["loc"], scale=params["scale"])
    
    elif distribution_name == "lognormal":
        if x <= 0:
            return 0.0
        return stats.norm.cdf(np.log(x), loc=params["mu"], scale=params["sigma"])
    
    elif distribution_name == "weibull":
        return stats.weibull_min.cdf(x, c=params["c"], scale=params["scale"])
    
    elif distribution_name == "gamma":
        return stats.gamma.cdf(x, a=params["a"], scale=params["scale"])
    
    elif distribution_name == "logistic":
        return stats.logistic.cdf(x, loc=params["loc"], scale=params["scale"])
    
    elif distribution_name == "exponential":
        return stats.expon.cdf(x, scale=params["scale"])
    
    else:
        raise ValueError(f"Unsupported distribution: {distribution_name}")

def get_distribution_fit_quality(aic: float, ks_pvalue: float) -> str:
    """
    Assess quality of distribution fit based on AIC and KS test.
    
    Args:
        aic: Akaike Information Criterion
        ks_pvalue: Kolmogorov-Smirnov test p-value
        
    Returns:
        Quality assessment: 'excellent', 'good', 'fair', 'poor'
    """
    if ks_pvalue is None:
        return "unknown"
    
    if ks_pvalue > 0.1:
        if aic is not None and aic < 0:  # Lower AIC is better
            return "excellent"
        else:
            return "good"
    elif ks_pvalue > 0.05:
        return "fair"
    else:
        return "poor"