import numpy as np
import pandas as pd
from scipy import stats
from typing import Dict, Any, Optional
from services.distributions import calculate_quantile, calculate_cdf

def calculate_capability_metrics(
    data: pd.Series,
    distribution_name: str,
    distribution_params: Dict[str, float],
    lsl: Optional[float] = None,
    usl: Optional[float] = None,
    target: Optional[float] = None,
    ci_method: str = "analytic_normal",
    confidence: float = 0.95
) -> Dict[str, Any]:
    """
    Calculate process capability indices for a given distribution.
    
    Args:
        data: Input data series
        distribution_name: Name of fitted distribution
        distribution_params: Parameters of fitted distribution
        lsl: Lower specification limit
        usl: Upper specification limit
        target: Target value
        ci_method: Confidence interval method
        confidence: Confidence level
        
    Returns:
        Dictionary with capability indices and confidence intervals
    """
    clean_data = data.dropna()
    n = len(clean_data)
    
    result = {
        "n": n,
        "distribution": distribution_name,
        "params": distribution_params,
        "specs": {"LSL": lsl, "USL": usl, "target": target},
        "ci_method": ci_method,
        "confidence": confidence,
        "success": False
    }
    
    if n < 2:
        result["message"] = "Insufficient data for capability analysis"
        return result
    
    if lsl is None and usl is None:
        result["message"] = "No specification limits provided"
        return result
    
    try:
        # Calculate basic statistics
        mean = np.mean(clean_data)
        std_dev = np.std(clean_data, ddof=1)
        
        # Calculate quantiles for non-normal distributions
        if distribution_name == "normal":
            # For normal distribution, use traditional formulas
            if lsl is not None and usl is not None:
                # Two-sided specifications
                pp = (usl - lsl) / (6 * std_dev) if std_dev > 0 else float('inf')
                ppu = (usl - mean) / (3 * std_dev) if std_dev > 0 else float('inf')
                ppl = (mean - lsl) / (3 * std_dev) if std_dev > 0 else float('inf')
                ppk = min(ppu, ppl)
                
                result.update({
                    "Pp": pp,
                    "Ppk": ppk,
                    "Ppu": ppu,
                    "Ppl": ppl
                })
                
            elif usl is not None:
                # Upper specification only
                ppu = (usl - mean) / (3 * std_dev) if std_dev > 0 else float('inf')
                result.update({
                    "Ppk": ppu,
                    "Ppu": ppu
                })
                
            elif lsl is not None:
                # Lower specification only
                ppl = (mean - lsl) / (3 * std_dev) if std_dev > 0 else float('inf')
                result.update({
                    "Ppk": ppl,
                    "Ppl": ppl
                })
        
        else:
            # For non-normal distributions, use quantile-based approach
            # Calculate process spread (6σ equivalent)
            p_0_00135 = calculate_quantile(distribution_name, distribution_params, 0.00135)
            p_0_5 = calculate_quantile(distribution_name, distribution_params, 0.5)
            p_0_99865 = calculate_quantile(distribution_name, distribution_params, 0.99865)
            
            process_spread = p_0_99865 - p_0_00135
            
            if lsl is not None and usl is not None:
                # Two-sided specifications
                spec_spread = usl - lsl
                pp = spec_spread / process_spread if process_spread > 0 else float('inf')
                
                # Ppk equivalent (min of upper and lower capability)
                upper_cap = (usl - p_0_5) / (p_0_99865 - p_0_5) * 3 if (p_0_99865 - p_0_5) > 0 else float('inf')
                lower_cap = (p_0_5 - lsl) / (p_0_5 - p_0_00135) * 3 if (p_0_5 - p_0_00135) > 0 else float('inf')
                ppk = min(upper_cap, lower_cap)
                
                result.update({
                    "Pp": pp,
                    "Ppk": ppk,
                    "Ppu": upper_cap,
                    "Ppl": lower_cap
                })
                
            elif usl is not None:
                # Upper specification only
                ppu = (usl - p_0_5) / (p_0_99865 - p_0_5) * 3 if (p_0_99865 - p_0_5) > 0 else float('inf')
                result.update({
                    "Ppk": ppu,
                    "Ppu": ppu
                })
                
            elif lsl is not None:
                # Lower specification only
                ppl = (p_0_5 - lsl) / (p_0_5 - p_0_00135) * 3 if (p_0_5 - p_0_00135) > 0 else float('inf')
                result.update({
                    "Ppk": ppl,
                    "Ppl": ppl
                })
        
        # Calculate confidence intervals if requested
        if ci_method == "analytic_normal" and distribution_name == "normal":
            ci_results = calculate_analytic_ci_normal(
                result, clean_data, lsl, usl, confidence
            )
            result.update(ci_results)
        elif ci_method == "bootstrap":
            ci_results = calculate_bootstrap_ci(
                clean_data, distribution_name, distribution_params, lsl, usl, confidence
            )
            result.update(ci_results)
        elif ci_method == "gof_adjusted":
            ci_results = calculate_gof_adjusted_ci(
                result, clean_data, distribution_name, distribution_params, lsl, usl, confidence
            )
            result.update(ci_results)
        
        result["success"] = True
        result["message"] = "Capability indices calculated successfully"
        
    except Exception as e:
        result["message"] = f"Error calculating capability indices: {str(e)}"
    
    return result

def calculate_analytic_ci_normal(
    capability_results: Dict[str, Any],
    data: pd.Series,
    lsl: Optional[float],
    usl: Optional[float],
    confidence: float = 0.95
) -> Dict[str, Any]:
    """
    Calculate analytic confidence intervals for normal distribution capability indices.
    
    Args:
        capability_results: Dictionary with capability indices
        data: Input data series
        lsl: Lower specification limit
        usl: Upper specification limit
        confidence: Confidence level
        
    Returns:
        Dictionary with confidence intervals
    """
    n = len(data)
    mean = np.mean(data)
    std_dev = np.std(data, ddof=1)
    
    ci_results = {}
    
    
    # Confidence interval for Ppk (approximate method)
    if "Ppk" in capability_results and capability_results["Ppk"] != float('inf'):
        ppk = capability_results["Ppk"]
        
        # Bissell's approximation
        z_alpha = stats.norm.ppf((1 + confidence) / 2)
        ppk_se = ppk * np.sqrt(1 / (9 * n * ppk**2) + 1 / (2 * (n - 1)))
        
        ppk_ci_lower = max(0, ppk - z_alpha * ppk_se)
        ppk_ci_upper = ppk + z_alpha * ppk_se
        
        ci_results["Ppk_ci_lower"] = ppk_ci_lower
        ci_results["Ppk_ci_upper"] = ppk_ci_upper
    
    return ci_results

def calculate_oos_from_distribution(
    distribution_name: str,
    distribution_params: Dict[str, float],
    lsl: Optional[float] = None,
    usl: Optional[float] = None
) -> float:
    """
    Calculate out-of-specification percentage from fitted distribution.
    
    Args:
        distribution_name: Name of fitted distribution
        distribution_params: Parameters of fitted distribution
        lsl: Lower specification limit
        usl: Upper specification limit
        
    Returns:
        Percentage of out-of-specification (0-100)
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

def calculate_empirical_oos(data: pd.Series, lsl: Optional[float] = None, usl: Optional[float] = None) -> float:
    """
    Calculate empirical out-of-specification percentage.
    
    Args:
        data: Input data series
        lsl: Lower specification limit
        usl: Upper specification limit
        
    Returns:
        Percentage of out-of-specification (0-100)
    """
    clean_data = data.dropna()
    n = len(clean_data)
    
    if n == 0:
        return 0.0
    
    oos_count = 0
    
    if lsl is not None:
        oos_count += np.sum(clean_data < lsl)
    
    if usl is not None:
        oos_count += np.sum(clean_data > usl)
    
    return (oos_count / n) * 100

def calculate_bootstrap_ci(
    data: pd.Series,
    distribution_name: str,
    distribution_params: Dict[str, float],
    lsl: Optional[float] = None,
    usl: Optional[float] = None,
    confidence: float = 0.95,
    n_bootstrap: int = 1000
) -> Dict[str, Any]:
    """
    Calculate bootstrap confidence intervals for capability indices.
    
    Args:
        data: Input data series
        distribution_name: Name of fitted distribution  
        distribution_params: Parameters of fitted distribution
        lsl: Lower specification limit
        usl: Upper specification limit
        confidence: Confidence level
        n_bootstrap: Number of bootstrap samples
        
    Returns:
        Dictionary with confidence intervals
    """
    np.random.seed(42)  # For reproducibility
    clean_data = data.dropna().values
    n = len(clean_data)
    
    bootstrap_results = {"n_bootstrap": n_bootstrap}
    
    # Storage for bootstrap statistics
    pp_values, ppk_values = [], []
    
    try:
        for i in range(n_bootstrap):
            # Bootstrap resample
            bootstrap_sample = np.random.choice(clean_data, size=n, replace=True)
            
            # Calculate capability indices for bootstrap sample
            mean_boot = np.mean(bootstrap_sample)
            std_boot = np.std(bootstrap_sample, ddof=1)
            
            if std_boot > 0:
                if lsl is not None and usl is not None:
                    # Two-sided specifications
                    spec_width = usl - lsl
                    pp_boot = spec_width / (6 * std_boot)
                    ppk_boot = min((usl - mean_boot) / (3 * std_boot), (mean_boot - lsl) / (3 * std_boot))
                    
                    pp_values.append(pp_boot)
                    ppk_values.append(ppk_boot)
                    
                elif lsl is not None:
                    # Lower specification only
                    ppk_boot = (mean_boot - lsl) / (3 * std_boot)
                    ppk_values.append(ppk_boot)
                    
                elif usl is not None:
                    # Upper specification only
                    ppk_boot = (usl - mean_boot) / (3 * std_boot)
                    ppk_values.append(ppk_boot)
        
        # Calculate confidence intervals
        alpha = 1 - confidence
        lower_percentile = (alpha / 2) * 100
        upper_percentile = (1 - alpha / 2) * 100
        
        if pp_values:
            bootstrap_results["Pp_ci_lower"] = np.percentile(pp_values, lower_percentile)
            bootstrap_results["Pp_ci_upper"] = np.percentile(pp_values, upper_percentile)
            
        if ppk_values:
            bootstrap_results["Ppk_ci_lower"] = np.percentile(ppk_values, lower_percentile)
            bootstrap_results["Ppk_ci_upper"] = np.percentile(ppk_values, upper_percentile)
            
    except Exception as e:
        bootstrap_results["bootstrap_error"] = str(e)
    
    return bootstrap_results

def calculate_gof_adjusted_ci(
    capability_results: Dict[str, Any],
    data: pd.Series,
    distribution_name: str,
    distribution_params: Dict[str, float],
    lsl: Optional[float] = None,
    usl: Optional[float] = None,
    confidence: float = 0.95
) -> Dict[str, Any]:
    """
    Calculate goodness-of-fit adjusted confidence intervals.
    Adjusts CI width based on how well the distribution fits the data.
    
    Args:
        capability_results: Current capability results
        data: Input data series
        distribution_name: Name of fitted distribution
        distribution_params: Parameters of fitted distribution
        lsl: Lower specification limit
        usl: Upper specification limit
        confidence: Confidence level
        
    Returns:
        Dictionary with adjusted confidence intervals
    """
    from services.distributions import fit_distribution
    
    clean_data = data.dropna()
    n = len(clean_data)
    
    # Get goodness of fit statistics
    fit_result = fit_distribution(clean_data, distribution_name)
    
    if not fit_result["success"]:
        return {}
    
    ks_pvalue = fit_result.get("ks_pvalue", 0.5)
    
    # Adjustment factor based on goodness of fit
    # Better fit (higher p-value) → narrower CI (smaller adjustment)
    # Poor fit (lower p-value) → wider CI (larger adjustment)
    if ks_pvalue >= 0.1:
        adjustment_factor = 1.0  # Good fit, no adjustment
        quality_msg = "Good fit"
    elif ks_pvalue >= 0.05:
        adjustment_factor = 1.2  # Fair fit, slight widening
        quality_msg = "Fair fit"
    elif ks_pvalue >= 0.01:
        adjustment_factor = 1.5  # Poor fit, moderate widening
        quality_msg = "Poor fit"
    else:
        adjustment_factor = 2.0   # Very poor fit, significant widening
        quality_msg = "Very poor fit"
    
    gof_results = {
        "gof_adjustment_factor": adjustment_factor,
        "gof_quality": quality_msg,
        "ks_pvalue": ks_pvalue
    }
    
    # Start with normal analytic CI if available, otherwise use bootstrap
    base_ci = {}
    
    if distribution_name == "normal":
        base_ci = calculate_analytic_ci_normal(capability_results, clean_data, lsl, usl, confidence)
    else:
        # Use simplified normal approximation for non-normal distributions
        base_ci = calculate_bootstrap_ci(clean_data, distribution_name, distribution_params, lsl, usl, confidence, 500)
    
    # Apply adjustment factor to widen intervals
    for key, value in base_ci.items():
        if key.endswith('_ci_lower') or key.endswith('_ci_upper'):
            metric_name = key.replace('_ci_lower', '').replace('_ci_upper', '')
            if metric_name in capability_results:
                point_estimate = capability_results[metric_name]
                
                # Adjust interval width
                if key.endswith('_ci_lower'):
                    adjusted_width = (point_estimate - value) * adjustment_factor
                    gof_results[key] = max(0, point_estimate - adjusted_width)
                else:  # _ci_upper
                    adjusted_width = (value - point_estimate) * adjustment_factor
                    gof_results[key] = point_estimate + adjusted_width
    
    return gof_results

def calculate_normal_ppk(
    data: pd.Series,
    lsl: Optional[float] = None,
    usl: Optional[float] = None,
    confidence: float = 0.95
) -> Dict[str, Any]:
    """
    Calculate Ppk using normal distribution assumption.
    
    Args:
        data: Input data series
        lsl: Lower specification limit
        usl: Upper specification limit
        confidence: Confidence level
        
    Returns:
        Dictionary with normal Ppk results
    """
    clean_data = data.dropna()
    n = len(clean_data)
    
    if n < 2:
        return {"success": False, "message": "Insufficient data", "method": "normal"}
    
    mean = np.mean(clean_data)
    std_dev = np.std(clean_data, ddof=1)
    
    result = {
        "method": "normal",
        "n": n,
        "mean": mean,
        "std_dev": std_dev,
        "success": False
    }
    
    try:
        if lsl is not None and usl is not None:
            # Two-sided
            ppu = (usl - mean) / (3 * std_dev)
            ppl = (mean - lsl) / (3 * std_dev)
            ppk = min(ppu, ppl)
            
            result.update({
                "Pp": (usl - lsl) / (6 * std_dev),
                "Ppk": ppk,
                "Ppu": ppu,
                "Ppl": ppl
            })
            
        elif usl is not None:
            # Upper one-sided
            ppu = (usl - mean) / (3 * std_dev)
            result.update({
                "Ppk": ppu,
                "Ppu": ppu
            })
            
        elif lsl is not None:
            # Lower one-sided
            ppl = (mean - lsl) / (3 * std_dev)
            result.update({
                "Ppk": ppl,
                "Ppl": ppl
            })
        
        else:
            result["message"] = "No specification limits provided"
            return result
        
        # Calculate confidence intervals using analytic method
        if "Ppk" in result:
            ci_results = calculate_analytic_ci_normal(result, clean_data, lsl, usl, confidence)
            result.update(ci_results)
        
        result["success"] = True
        result["message"] = "Normal distribution Ppk calculated"
        
    except Exception as e:
        result["message"] = f"Error: {str(e)}"
    
    return result

def calculate_fitted_distribution_ppk(
    data: pd.Series,
    distribution_name: str,
    distribution_params: Dict[str, float],
    lsl: Optional[float] = None,
    usl: Optional[float] = None,
    confidence: float = 0.95
) -> Dict[str, Any]:
    """
    Calculate Ppk using fitted distribution percentiles.
    
    Args:
        data: Input data series
        distribution_name: Name of fitted distribution
        distribution_params: Parameters of fitted distribution
        lsl: Lower specification limit
        usl: Upper specification limit
        confidence: Confidence level
        
    Returns:
        Dictionary with fitted distribution Ppk results
    """
    from services.distributions import calculate_quantile
    
    clean_data = data.dropna()
    n = len(clean_data)
    
    result = {
        "method": "fitted_distribution",
        "distribution": distribution_name,
        "n": n,
        "success": False
    }
    
    try:
        # Calculate key percentiles from fitted distribution
        # 0.135% and 99.865% correspond to μ±3σ in normal distribution
        p0_135 = calculate_quantile(distribution_name, distribution_params, 0.00135)
        p50 = calculate_quantile(distribution_name, distribution_params, 0.5)
        p99_865 = calculate_quantile(distribution_name, distribution_params, 0.99865)
        
        # Calculate equivalent "3-sigma" spread
        # For fitted distribution, we use the 99.865% - 0.135% range as 6-sigma equivalent
        sigma_equivalent = (p99_865 - p0_135) / 6
        
        result.update({
            "p0_135": p0_135,
            "p50": p50,
            "p99_865": p99_865,
            "sigma_equivalent": sigma_equivalent
        })
        
        if sigma_equivalent <= 0:
            result["message"] = "Invalid distribution parameters - zero spread"
            return result
        
        if lsl is not None and usl is not None:
            # Two-sided
            ppu = (usl - p50) / (3 * sigma_equivalent)
            ppl = (p50 - lsl) / (3 * sigma_equivalent)
            ppk = min(ppu, ppl)
            
            result.update({
                "Pp": (usl - lsl) / (6 * sigma_equivalent),
                "Ppk": ppk,
                "Ppu": ppu,
                "Ppl": ppl
            })
            
        elif usl is not None:
            # Upper one-sided
            ppu = (usl - p50) / (3 * sigma_equivalent)
            result.update({
                "Ppk": ppu,
                "Ppu": ppu
            })
            
        elif lsl is not None:
            # Lower one-sided
            ppl = (p50 - lsl) / (3 * sigma_equivalent)
            result.update({
                "Ppk": ppl,
                "Ppl": ppl
            })
        
        else:
            result["message"] = "No specification limits provided"
            return result
        
        # Calculate bootstrap confidence intervals
        bootstrap_ci = calculate_bootstrap_ci(
            data, distribution_name, distribution_params, lsl, usl, confidence, 500
        )
        result.update(bootstrap_ci)
        
        result["success"] = True
        result["message"] = f"Fitted distribution ({distribution_name}) Ppk calculated"
        
    except Exception as e:
        result["message"] = f"Error: {str(e)}"
    
    return result

def calculate_empirical_ppk(
    data: pd.Series,
    lsl: Optional[float] = None,
    usl: Optional[float] = None,
    confidence: float = 0.95
) -> Dict[str, Any]:
    """
    Calculate Ppk using empirical (sample) percentiles.
    
    Args:
        data: Input data series
        lsl: Lower specification limit
        usl: Upper specification limit
        confidence: Confidence level
        
    Returns:
        Dictionary with empirical Ppk results
    """
    clean_data = data.dropna()
    n = len(clean_data)
    
    result = {
        "method": "empirical",
        "n": n,
        "success": False
    }
    
    if n < 10:
        result["message"] = "Insufficient data for reliable empirical percentiles (n < 10)"
        return result
    
    try:
        # Calculate empirical percentiles
        p0_135 = np.percentile(clean_data, 0.135)
        p50 = np.percentile(clean_data, 50)
        p99_865 = np.percentile(clean_data, 99.865)
        
        # Alternative: use IQR-based sigma estimation
        q25 = np.percentile(clean_data, 25)
        q75 = np.percentile(clean_data, 75)
        iqr_sigma = (q75 - q25) / 1.35  # For normal distribution, IQR ≈ 1.35σ
        
        # For empirical method, use direct percentile approach
        method_used = "percentile-based"
        
        result.update({
            "p0_135": p0_135,
            "p50": p50,
            "p99_865": p99_865,
            "q25": q25,
            "q75": q75,
            "iqr_sigma": iqr_sigma,
            "method_used": method_used
        })
        
        # Check for valid spread
        if (p99_865 - p0_135) <= 0:
            result["message"] = "Invalid empirical spread - zero variation"
            return result
        
        if lsl is not None and usl is not None:
            # Two-sided - use direct percentile approach
            Pp = (usl - lsl) / (p99_865 - p0_135) if (p99_865 - p0_135) > 0 else float('inf')
            Ppu = (usl - p50) / (p99_865 - p50) if (p99_865 - p50) > 0 else float('inf')
            Ppl = (p50 - lsl) / (p50 - p0_135) if (p50 - p0_135) > 0 else float('inf')
            Ppk = min(Ppu, Ppl)
            
            result.update({
                "Pp": Pp,
                "Ppk": Ppk,
                "Ppu": Ppu,
                "Ppl": Ppl
            })
            
        elif usl is not None:
            # Upper one-sided
            Ppk = (usl - p50) / (p99_865 - p50) if (p99_865 - p50) > 0 else float('inf')
            result.update({
                "Ppk": Ppk,
                "Ppu": Ppk
            })
            
        elif lsl is not None:
            # Lower one-sided
            Ppk = (p50 - lsl) / (p50 - p0_135) if (p50 - p0_135) > 0 else float('inf')
            result.update({
                "Ppk": Ppk,
                "Ppl": Ppk
            })
        
        else:
            result["message"] = "No specification limits provided"
            return result
        
        # Calculate bootstrap confidence intervals for empirical method
        empirical_bootstrap = _calculate_empirical_bootstrap_ci(clean_data, lsl, usl, confidence)
        result.update(empirical_bootstrap)
        
        result["success"] = True
        result["message"] = f"Empirical Ppk calculated using {method_used} method"
        
    except Exception as e:
        result["message"] = f"Error: {str(e)}"
    
    return result

def _calculate_empirical_bootstrap_ci(
    data: np.ndarray,
    lsl: Optional[float] = None,
    usl: Optional[float] = None,
    confidence: float = 0.95,
    n_bootstrap: int = 1000
) -> Dict[str, Any]:
    """
    Calculate bootstrap confidence intervals for empirical Ppk.
    """
    np.random.seed(42)
    ppk_values = []
    
    try:
        for _ in range(n_bootstrap):
            bootstrap_sample = np.random.choice(data, size=len(data), replace=True)
            
            # Calculate empirical percentiles for bootstrap sample
            p0_135 = np.percentile(bootstrap_sample, 0.135)
            p50 = np.percentile(bootstrap_sample, 50)
            p99_865 = np.percentile(bootstrap_sample, 99.865)
            
            sigma_equiv = (p99_865 - p0_135) / 6
            
            if sigma_equiv > 0:
                if lsl is not None and usl is not None:
                    ppu = (usl - p50) / (3 * sigma_equiv)
                    ppl = (p50 - lsl) / (3 * sigma_equiv)
                    ppk = min(ppu, ppl)
                elif usl is not None:
                    ppk = (usl - p50) / (3 * sigma_equiv)
                elif lsl is not None:
                    ppk = (p50 - lsl) / (3 * sigma_equiv)
                else:
                    continue
                
                ppk_values.append(ppk)
        
        if ppk_values:
            alpha = 1 - confidence
            lower_percentile = (alpha / 2) * 100
            upper_percentile = (1 - alpha / 2) * 100
            
            return {
                "Ppk_ci_lower": np.percentile(ppk_values, lower_percentile),
                "Ppk_ci_upper": np.percentile(ppk_values, upper_percentile),
                "n_bootstrap_empirical": len(ppk_values)
            }
        
    except Exception as e:
        pass
    
    return {}