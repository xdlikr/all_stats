import numpy as np
from scipy import stats, optimize, integrate
from typing import Dict, Any, Optional, Tuple
import warnings
from services.distributions import calculate_quantile

def tolerance_factor_method3_one_sided(n: int, p: float, conf: float = 0.95) -> float:
    """
    Calculate one-sided tolerance factor using METHOD=3 (exact method).
    
    Args:
        n: Sample size
        p: Proportion of population to cover
        conf: Confidence level
        
    Returns:
        Tolerance factor g
    """
    if n < 2:
        raise ValueError("Sample size must be at least 2")
    if not (0 < p < 1):
        raise ValueError("Proportion p must be between 0 and 1")
    if not (0 < conf < 1):
        raise ValueError("Confidence level must be between 0 and 1")
    
    # Non-central t distribution parameter
    df = n - 1
    nc = stats.norm.ppf(p) * np.sqrt(n)
    
    # Calculate tolerance factor g
    g = stats.nct.ppf(conf, df, nc) / np.sqrt(n)
    
    return g

def tolerance_factor_method3_two_sided(n: int, p: float, conf: float = 0.95, 
                                      method: str = "howe") -> float:
    """
    Calculate two-sided tolerance factor using METHOD=3.
    
    Args:
        n: Sample size
        p: Proportion of population to cover
        conf: Confidence level
        method: Method to use ('howe' for approximation, 'exact' for numerical solution)
        
    Returns:
        Tolerance factor k
    """
    if n < 2:
        raise ValueError("Sample size must be at least 2")
    if not (0 < p < 1):
        raise ValueError("Proportion p must be between 0 and 1")
    if not (0 < conf < 1):
        raise ValueError("Confidence level must be between 0 and 1")
    
    if method == "howe":
        # Howe's approximation (commonly used in engineering)
        df = n - 1
        z_p = stats.norm.ppf((1 + p) / 2)
        chi2_alpha = stats.chi2.ppf(1 - conf, df)
        
        k = z_p * np.sqrt(df * (1 + 1/n) / chi2_alpha)
        
    elif method == "exact":
        # Exact numerical solution (more computationally intensive)
        k = _solve_exact_tolerance_factor(n, p, conf)
        
    else:
        raise ValueError("Method must be 'howe' or 'exact'")
    
    return k

def _solve_exact_tolerance_factor(n: int, p: float, conf: float = 0.95) -> float:
    """
    Solve for exact two-sided tolerance factor using numerical methods.
    
    Args:
        n: Sample size
        p: Proportion of population to cover
        conf: Confidence level
        
    Returns:
        Exact tolerance factor k
    """
    df = n - 1
    
    def equation(k):
        # Integral equation for exact tolerance factor
        integrand = lambda u: stats.norm.cdf(u + k) - stats.norm.cdf(u - k)
        
        # Numerical integration over chi-square distribution
        result, _ = integrate.quad(
            lambda u: integrand(u) * stats.chi2.pdf(u * u * n / df, df) * 2 * u * n / df,
            0, np.inf
        )
        
        return result - p
    
    # Find k that satisfies the equation
    try:
        solution = optimize.root_scalar(
            equation,
            bracket=[1.0, 10.0],  # Reasonable range for k
            method='brentq'
        )
        return solution.root
    except:
        # Fallback to Howe's approximation if numerical solution fails
        warnings.warn("Exact solution failed, using Howe's approximation")
        return tolerance_factor_method3_two_sided(n, p, conf, method="howe")

def calculate_distribution_based_tolerance_intervals(
    distribution_name: str,
    distribution_params: Dict[str, float],
    p: float = 0.95,
    conf: float = 0.95,
    n: int = None,
    bootstrap_iterations: int = 1000
) -> Dict[str, Any]:
    """
    Calculate tolerance intervals based on fitted distribution using bootstrap for confidence.
    
    Args:
        distribution_name: Name of fitted distribution
        distribution_params: Parameters of fitted distribution
        p: Proportion of population to cover
        conf: Confidence level
        n: Sample size (for bootstrap)
        bootstrap_iterations: Number of bootstrap iterations
        
    Returns:
        Dictionary with tolerance interval results
    """
    result = {
        "type": "distribution-based",
        "distribution": distribution_name,
        "params": distribution_params,
        "p": p,
        "conf": conf,
        "success": False
    }
    
    try:
        # Calculate point estimates using distribution quantiles
        alpha = 1 - p
        lower_quantile = alpha / 2
        upper_quantile = 1 - alpha / 2
        
        point_lower = calculate_quantile(distribution_name, distribution_params, lower_quantile)
        point_upper = calculate_quantile(distribution_name, distribution_params, upper_quantile)
        
        result.update({
            "point_interval": (point_lower, point_upper),
            "point_lower": point_lower,
            "point_upper": point_upper
        })
        
        # If sample size provided, calculate confidence intervals using bootstrap
        if n is not None and n > 5:
            # Bootstrap confidence intervals for the tolerance interval
            bootstrap_results = _bootstrap_distribution_tolerance(
                distribution_name, distribution_params, p, n, bootstrap_iterations
            )
            
            if bootstrap_results["success"]:
                # Calculate confidence intervals for the bounds
                conf_alpha = 1 - conf
                ci_lower_idx = int(conf_alpha / 2 * bootstrap_iterations)
                ci_upper_idx = int((1 - conf_alpha / 2) * bootstrap_iterations)
                
                lower_bounds_sorted = np.sort(bootstrap_results["lower_bounds"])
                upper_bounds_sorted = np.sort(bootstrap_results["upper_bounds"])
                
                ci_lower_bound = (lower_bounds_sorted[ci_lower_idx], lower_bounds_sorted[ci_upper_idx])
                ci_upper_bound = (upper_bounds_sorted[ci_lower_idx], upper_bounds_sorted[ci_upper_idx])
                
                # Conservative interval: use worst-case bounds
                conservative_lower = lower_bounds_sorted[ci_lower_idx]
                conservative_upper = upper_bounds_sorted[ci_upper_idx]
                
                result.update({
                    "interval": (conservative_lower, conservative_upper),
                    "confidence_intervals": {
                        "lower_bound_ci": ci_lower_bound,
                        "upper_bound_ci": ci_upper_bound
                    },
                    "bootstrap_stats": bootstrap_results,
                    "method_used": f"Distribution-based with Bootstrap CI (n={bootstrap_iterations})"
                })
            else:
                # Fallback to point estimates
                result.update({
                    "interval": (point_lower, point_upper),
                    "method_used": "Distribution-based (Point Estimates Only)"
                })
        else:
            # No confidence interval calculation
            result.update({
                "interval": (point_lower, point_upper),
                "method_used": "Distribution-based (Point Estimates Only)"
            })
        
        result["success"] = True
        result["message"] = f"Distribution-based tolerance interval calculated for {distribution_name}"
        
    except Exception as e:
        result["message"] = f"Error calculating distribution-based tolerance interval: {str(e)}"
    
    return result

def _bootstrap_distribution_tolerance(
    distribution_name: str,
    distribution_params: Dict[str, float], 
    p: float,
    n: int,
    iterations: int = 1000
) -> Dict[str, Any]:
    """
    Bootstrap confidence intervals for distribution-based tolerance intervals.
    """
    try:
        lower_bounds = []
        upper_bounds = []
        
        alpha = 1 - p
        lower_quantile = alpha / 2
        upper_quantile = 1 - alpha / 2
        
        # Generate bootstrap samples
        np.random.seed(42)  # For reproducibility
        
        for _ in range(iterations):
            # Generate a bootstrap sample from the distribution
            if distribution_name == "normal":
                sample = np.random.normal(
                    distribution_params["loc"], 
                    distribution_params["scale"], 
                    n
                )
            elif distribution_name == "lognormal":
                sample = np.random.lognormal(
                    distribution_params["mu"], 
                    distribution_params["sigma"], 
                    n
                )
            elif distribution_name == "weibull":
                sample = np.random.weibull(
                    distribution_params["c"], 
                    n
                ) * distribution_params["scale"]
            elif distribution_name == "gamma":
                sample = np.random.gamma(
                    distribution_params["a"], 
                    distribution_params["scale"], 
                    n
                )
            elif distribution_name == "logistic":
                sample = np.random.logistic(
                    distribution_params["loc"], 
                    distribution_params["scale"], 
                    n
                )
            else:
                # Fallback: use empirical bootstrap
                sample = np.random.choice(np.random.normal(0, 1, 1000), n, replace=True)
            
            # Calculate tolerance bounds for this bootstrap sample
            sample_lower = np.percentile(sample, lower_quantile * 100)
            sample_upper = np.percentile(sample, upper_quantile * 100)
            
            lower_bounds.append(sample_lower)
            upper_bounds.append(sample_upper)
        
        return {
            "success": True,
            "lower_bounds": lower_bounds,
            "upper_bounds": upper_bounds,
            "iterations": iterations
        }
        
    except Exception as e:
        return {
            "success": False,
            "message": f"Bootstrap failed: {str(e)}"
        }

def calculate_tolerance_intervals(
    data: np.ndarray,
    p: float = 0.95,
    conf: float = 0.95,
    method: str = "auto",
    two_sided_method: str = "howe"
) -> Dict[str, Any]:
    """
    Calculate tolerance intervals for data.
    
    Args:
        data: Input data array
        p: Proportion of population to cover
        conf: Confidence level
        method: Method to use ('auto', 'one-sided', 'two-sided')
        two_sided_method: For two-sided intervals ('howe', 'exact')
        
    Returns:
        Dictionary with tolerance interval results
    """
    clean_data = data[~np.isnan(data)]
    n = len(clean_data)
    
    if n < 2:
        return {
            "success": False,
            "message": "Insufficient data for tolerance interval calculation",
            "n": n
        }
    
    mean = np.mean(clean_data)
    std_dev = np.std(clean_data, ddof=1)
    
    result = {
        "n": n,
        "mean": mean,
        "std_dev": std_dev,
        "p": p,
        "conf": conf,
        "method": method,
        "success": False
    }
    
    try:
        if method == "one-sided" or (method == "auto" and p >= 0.9):
            # One-sided tolerance interval
            g = tolerance_factor_method3_one_sided(n, p, conf)
            tolerance_lower = mean - g * std_dev
            tolerance_upper = mean + g * std_dev
            
            result.update({
                "type": "one-sided",
                "method_used": "METHOD=3 Exact (One-sided)",
                "factor": g,
                "interval": (tolerance_lower, tolerance_upper),
                "success": True,
                "message": "One-sided tolerance interval calculated using exact method"
            })
            
        else:
            # Two-sided tolerance interval
            k = tolerance_factor_method3_two_sided(n, p, conf, method=two_sided_method)
            tolerance_lower = mean - k * std_dev
            tolerance_upper = mean + k * std_dev
            
            method_description = "Howe Approximation" if two_sided_method == "howe" else "Exact Numerical"
            result.update({
                "type": "two-sided",
                "method_used": f"METHOD=3 {method_description}",
                "factor": k,
                "interval": (tolerance_lower, tolerance_upper),
                "success": True,
                "message": f"Two-sided tolerance interval calculated using {method_description}"
            })
            
    except Exception as e:
        result["message"] = f"Error calculating tolerance interval: {str(e)}"
    
    return result

def coverage_scan(
    data: np.ndarray,
    lsl: Optional[float] = None,
    usl: Optional[float] = None,
    p_min: float = 0.80,
    p_max: float = 0.995,
    p_step: float = 0.005,
    conf: float = 0.95,
    two_sided_method: str = "howe"
) -> Dict[str, Any]:
    """
    Perform coverage scan to find maximum p where tolerance interval is within specifications.
    
    Args:
        data: Input data array
        lsl: Lower specification limit
        usl: Upper specification limit
        p_min: Minimum proportion to scan
        p_max: Maximum proportion to scan
        p_step: Step size for proportion
        conf: Confidence level
        
    Returns:
        Dictionary with coverage scan results
    """
    clean_data = data[~np.isnan(data)]
    n = len(clean_data)
    
    if n < 2:
        return {
            "success": False,
            "message": "Insufficient data for coverage scan",
            "n": n
        }
    
    if lsl is None and usl is None:
        return {
            "success": False,
            "message": "No specification limits provided",
            "n": n
        }
    
    # Determine interval type based on specifications
    if lsl is not None and usl is not None:
        interval_type = "two-sided"
    else:
        interval_type = "one-sided"
    
    results = []
    p_star = p_min
    oos_ti = 1 - p_min
    
    # Scan through p values
    p_values = np.arange(p_min, p_max + p_step, p_step)
    
    for p_val in p_values:
        try:
            if interval_type == "two-sided":
                k = tolerance_factor_method3_two_sided(n, p_val, conf, method=two_sided_method)
                lower_bound = np.mean(clean_data) - k * np.std(clean_data, ddof=1)
                upper_bound = np.mean(clean_data) + k * np.std(clean_data, ddof=1)
                
                # Check if tolerance interval is within specifications
                within_specs = (lower_bound >= lsl) and (upper_bound <= usl)
                
                results.append({
                    "p": p_val,
                    "coverage_%": p_val * 100,
                    "lower_bound": lower_bound,
                    "upper_bound": upper_bound,
                    "within_specs": within_specs,
                    "oos_ti": 1 - p_val
                })
                
            else:
                g = tolerance_factor_method3_one_sided(n, p_val, conf)
                
                if usl is not None:
                    # Upper one-sided
                    upper_bound = np.mean(clean_data) + g * np.std(clean_data, ddof=1)
                    within_specs = upper_bound <= usl
                    
                    results.append({
                        "p": p_val,
                        "coverage_%": p_val * 100,
                        "upper_bound": upper_bound,
                        "within_specs": within_specs,
                        "oos_ti": 1 - p_val
                    })
                else:
                    # Lower one-sided
                    lower_bound = np.mean(clean_data) - g * np.std(clean_data, ddof=1)
                    within_specs = lower_bound >= lsl
                    
                    results.append({
                        "p": p_val,
                        "coverage_%": p_val * 100,
                        "lower_bound": lower_bound,
                        "within_specs": within_specs,
                        "oos_ti": 1 - p_val
                    })
            
            if within_specs and p_val > p_star:
                p_star = p_val
                oos_ti = 1 - p_val
                
        except Exception as e:
            warnings.warn(f"Error calculating for p={p_val}: {str(e)}")
            continue
    
    return {
        "success": True,
        "message": "Coverage scan completed",
        "n": n,
        "interval_type": interval_type,
        "p_star": p_star,
        "oos_ti": oos_ti,
        "results": results,
        "scan_range": (p_min, p_max, p_step)
    }