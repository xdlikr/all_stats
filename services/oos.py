import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, List
from scipy import stats
from services.distributions import calculate_cdf
from services.tolerance import coverage_scan

def calculate_oos(
    data: pd.Series,
    method: str = "fitted",
    distribution_name: Optional[str] = None,
    distribution_params: Optional[Dict[str, float]] = None,
    lsl: Optional[float] = None,
    usl: Optional[float] = None,
    conf: float = 0.95
) -> Dict[str, Any]:
    """
    Calculate out-of-specification percentage using different methods.
    
    Args:
        data: Input data series
        method: Calculation method ('fitted', 'empirical', 'ti_method3')
        distribution_name: Name of fitted distribution (required for 'fitted' method)
        distribution_params: Parameters of fitted distribution (required for 'fitted' method)
        lsl: Lower specification limit
        usl: Upper specification limit
        conf: Confidence level (for 'ti_method3')
        
    Returns:
        Dictionary with %OOS results
    """
    clean_data = data.dropna()
    n = len(clean_data)
    
    result = {
        "n": n,
        "method": method,
        "lsl": lsl,
        "usl": usl,
        "success": False,
        "oos_percentage": None,
        "message": ""
    }
    
    if n == 0:
        result["message"] = "No data available"
        return result
    
    if lsl is None and usl is None:
        result["message"] = "No specification limits provided"
        return result
    
    try:
        if method == "normal":
            # Calculate %OOS assuming normal distribution with sample mean/std
            oos_percentage = _calculate_normal_oos(clean_data, lsl, usl)
            result["oos_percentage"] = oos_percentage
            result["message"] = "%OOS calculated assuming normal distribution"
            
        elif method == "fitted":
            if distribution_name is None or distribution_params is None:
                result["message"] = "Distribution information required for fitted method"
                return result
            
            # Calculate %OOS from fitted distribution
            oos_percentage = _calculate_fitted_oos(distribution_name, distribution_params, lsl, usl)
            result["oos_percentage"] = oos_percentage
            result["message"] = "%OOS calculated from fitted distribution"
            
        elif method == "empirical":
            # Calculate empirical %OOS
            oos_percentage = _calculate_empirical_oos(clean_data, lsl, usl)
            result["oos_percentage"] = oos_percentage
            result["message"] = "%OOS calculated empirically from data"
            
        elif method == "ti_method3":
            # Calculate %OOS using tolerance interval method
            oos_percentage = _calculate_ti_oos(clean_data, lsl, usl, conf)
            result["oos_percentage"] = oos_percentage
            result["message"] = "%OOS calculated using tolerance interval method"
            
        else:
            result["message"] = f"Unknown method: {method}"
            return result
        
        result["success"] = True
        
    except Exception as e:
        result["message"] = f"Error calculating %OOS: {str(e)}"
    
    return result

def _calculate_normal_oos(data: np.ndarray, lsl: Optional[float], usl: Optional[float]) -> float:
    """
    Calculate %OOS assuming normal distribution with sample mean and standard deviation.
    """
    mean = np.mean(data)
    std_dev = np.std(data, ddof=1)
    
    oos_percentage = 0.0
    
    if lsl is not None:
        # Probability below LSL using normal distribution
        prob_below_lsl = stats.norm.cdf(lsl, loc=mean, scale=std_dev)
        oos_percentage += prob_below_lsl * 100
    
    if usl is not None:
        # Probability above USL using normal distribution  
        prob_above_usl = 1 - stats.norm.cdf(usl, loc=mean, scale=std_dev)
        oos_percentage += prob_above_usl * 100
    
    return oos_percentage

def _calculate_fitted_oos(
    distribution_name: str,
    distribution_params: Dict[str, float],
    lsl: Optional[float],
    usl: Optional[float]
) -> float:
    """
    Calculate %OOS from fitted distribution.
    """
    oos_percentage = 0.0
    
    if lsl is not None:
        # Probability below LSL
        prob_below_lsl = calculate_cdf(distribution_name, distribution_params, lsl)
        oos_percentage += prob_below_lsl * 100
    
    if usl is not None:
        # Probability above USL
        prob_above_usl = 1 - calculate_cdf(distribution_name, distribution_params, usl)
        oos_percentage += prob_above_usl * 100
    
    return oos_percentage

def _calculate_empirical_oos(data: np.ndarray, lsl: Optional[float], usl: Optional[float]) -> float:
    """
    Calculate empirical %OOS from data.
    """
    n = len(data)
    oos_count = 0
    
    if lsl is not None:
        oos_count += np.sum(data < lsl)
    
    if usl is not None:
        oos_count += np.sum(data > usl)
    
    return (oos_count / n) * 100

def _calculate_ti_oos(data: np.ndarray, lsl: Optional[float], usl: Optional[float], conf: float = 0.95) -> float:
    """
    Calculate %OOS using tolerance interval method.
    """
    # Perform coverage scan to find maximum p where TI is within specs
    coverage_results = coverage_scan(data, lsl, usl, p_min=0.80, p_max=0.995, p_step=0.005, conf=conf)
    
    if coverage_results["success"]:
        # %OOS_TI = 1 - p_star
        return coverage_results["oos_ti"] * 100
    else:
        raise ValueError(f"Coverage scan failed: {coverage_results['message']}")

def compare_oos_methods(
    data: pd.Series,
    distribution_name: Optional[str] = None,
    distribution_params: Optional[Dict[str, float]] = None,
    lsl: Optional[float] = None,
    usl: Optional[float] = None,
    conf: float = 0.95,
    methods: List[str] = None
) -> Dict[str, Any]:
    """
    Compare %OOS results from different calculation methods.
    
    Args:
        data: Input data series
        distribution_name: Name of fitted distribution
        distribution_params: Parameters of fitted distribution
        lsl: Lower specification limit
        usl: Upper specification limit
        conf: Confidence level
        methods: List of methods to compare
        
    Returns:
        Dictionary with comparison results
    """
    if methods is None:
        methods = ["normal", "empirical", "fitted"]
    
    results = {}
    comparison_table = []
    
    for method in methods:
        oos_result = calculate_oos(
            data, method, distribution_name, distribution_params, lsl, usl, conf
        )
        
        results[method] = oos_result
        
        if oos_result["success"]:
            comparison_table.append({
                "Method": method,
                "%OOS": oos_result["oos_percentage"],
                "Success": True,
                "Message": oos_result["message"]
            })
        else:
            comparison_table.append({
                "Method": method,
                "%OOS": None,
                "Success": False,
                "Message": oos_result["message"]
            })
    
    return {
        "results": results,
        "comparison_table": comparison_table,
        "n_methods": len([x for x in comparison_table if x["Success"]])
    }

def calculate_oos_vs_ppk_curve(
    distribution_name: str,
    distribution_params: Dict[str, float],
    lsl: Optional[float] = None,
    usl: Optional[float] = None,
    ppk_range: tuple = (0.5, 2.0),
    ppk_steps: int = 50
) -> Dict[str, Any]:
    """
    Calculate %OOS vs Ppk curve for a given distribution.
    
    Args:
        distribution_name: Name of distribution
        distribution_params: Distribution parameters
        lsl: Lower specification limit
        usl: Upper specification limit
        ppk_range: Range of Ppk values to evaluate
        ppk_steps: Number of steps in the range
        
    Returns:
        Dictionary with curve data
    """
    ppk_values = np.linspace(ppk_range[0], ppk_range[1], ppk_steps)
    oos_values = []
    
    for ppk in ppk_values:
        try:
            # For normal distribution with centered mean
            if distribution_name == "normal" and lsl is not None and usl is not None:
                # Assume centered process (mean = (LSL + USL)/2)
                mean = (lsl + usl) / 2
                # Calculate required standard deviation for given Ppk
                # Ppk = min(USL - mean, mean - LSL) / (3σ)
                sigma = min(usl - mean, mean - lsl) / (3 * ppk)
                
                # Update distribution parameters
                updated_params = {"loc": mean, "scale": sigma}
                
                # Calculate %OOS
                oos = _calculate_fitted_oos(distribution_name, updated_params, lsl, usl)
                oos_values.append(oos)
                
            else:
                # For non-normal distributions, use a simplified approach
                # This is an approximation
                if lsl is not None and usl is not None:
                    # Two-sided specifications
                    oos = 2 * (1 - stats.norm.cdf(3 * ppk)) * 100
                else:
                    # One-sided specification
                    oos = (1 - stats.norm.cdf(3 * ppk)) * 100
                oos_values.append(oos)
                
        except:
            oos_values.append(None)
    
    return {
        "ppk_values": ppk_values,
        "oos_values": oos_values,
        "distribution": distribution_name,
        "spec_type": "two_sided" if (lsl and usl) else "one_sided"
    }