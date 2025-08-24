import pandas as pd
import numpy as np
from scipy import stats
from typing import Dict, Any

def calculate_basic_statistics(data: pd.Series) -> Dict[str, Any]:
    """
    Calculate basic descriptive statistics for a data series.
    
    Args:
        data: Input data series
        
    Returns:
        Dictionary with statistical measures
    """
    # Remove missing values
    clean_data = data.dropna()
    n = len(clean_data)
    
    if n == 0:
        return {
            "n": 0,
            "missing_count": len(data) - n,
            "message": "No valid data points"
        }
    
    # Basic statistics
    mean = np.mean(clean_data)
    median = np.median(clean_data)
    std_dev = np.std(clean_data, ddof=1)  # Sample standard deviation
    variance = np.var(clean_data, ddof=1)
    
    # Robust statistics
    mad = stats.median_abs_deviation(clean_data, scale='normal')
    q1 = np.percentile(clean_data, 25)
    q3 = np.percentile(clean_data, 75)
    iqr = q3 - q1
    
    # Shape statistics
    skewness = stats.skew(clean_data)
    kurtosis = stats.kurtosis(clean_data)
    
    # Range statistics
    min_val = np.min(clean_data)
    max_val = np.max(clean_data)
    range_val = max_val - min_val
    
    # Percentiles
    percentiles = {
        "p1": np.percentile(clean_data, 1),
        "p5": np.percentile(clean_data, 5),
        "p10": np.percentile(clean_data, 10),
        "p25": q1,
        "p50": median,
        "p75": q3,
        "p90": np.percentile(clean_data, 90),
        "p95": np.percentile(clean_data, 95),
        "p99": np.percentile(clean_data, 99)
    }
    
    return {
        "n": n,
        "missing_count": len(data) - n,
        "missing_percentage": (len(data) - n) / len(data) * 100 if len(data) > 0 else 0,
        "mean": mean,
        "median": median,
        "std_dev": std_dev,
        "variance": variance,
        "mad": mad,
        "min": min_val,
        "max": max_val,
        "range": range_val,
        "q1": q1,
        "q3": q3,
        "iqr": iqr,
        "skewness": skewness,
        "kurtosis": kurtosis,
        "percentiles": percentiles,
        "cv": (std_dev / mean) * 100 if mean != 0 else float('inf')  # Coefficient of variation
    }

def calculate_process_capability_basic(data: pd.Series, lsl: float = None, usl: float = None) -> Dict[str, Any]:
    """
    Calculate basic process capability indices (assuming normal distribution).
    
    Args:
        data: Input data series
        lsl: Lower specification limit
        usl: Upper specification limit
        
    Returns:
        Dictionary with capability indices
    """
    stats = calculate_basic_statistics(data)
    n = stats["n"]
    mean = stats["mean"]
    std_dev = stats["std_dev"]
    
    result = {
        "n": n,
        "mean": mean,
        "std_dev": std_dev,
        "specs": {"LSL": lsl, "USL": usl}
    }
    
    if n < 2:
        result["message"] = "Insufficient data for capability analysis"
        return result
    
    # Calculate capability indices
    if lsl is not None and usl is not None:
        # Two-sided specifications
        cp = (usl - lsl) / (6 * std_dev) if std_dev > 0 else float('inf')
        cpu = (usl - mean) / (3 * std_dev) if std_dev > 0 else float('inf')
        cpl = (mean - lsl) / (3 * std_dev) if std_dev > 0 else float('inf')
        cpk = min(cpu, cpl)
        
        result.update({
            "Cp": cp,
            "Cpu": cpu,
            "Cpl": cpl,
            "Cpk": cpk,
            "Pp": cp,  # For basic calculation, Pp = Cp
            "Ppk": cpk  # For basic calculation, Ppk = Cpk
        })
        
    elif usl is not None:
        # Upper specification only
        cpu = (usl - mean) / (3 * std_dev) if std_dev > 0 else float('inf')
        result.update({
            "Cpu": cpu,
            "Cpk": cpu,
            "Ppu": cpu,
            "Ppk": cpu
        })
        
    elif lsl is not None:
        # Lower specification only
        cpl = (mean - lsl) / (3 * std_dev) if std_dev > 0 else float('inf')
        result.update({
            "Cpl": cpl,
            "Cpk": cpl,
            "Ppl": cpl,
            "Ppk": cpl
        })
    
    else:
        result["message"] = "No specification limits provided"
    
    return result

def detect_outliers(data: pd.Series, method: str = "iqr", threshold: float = 1.5) -> Dict[str, Any]:
    """
    Detect outliers in data using specified method.
    
    Args:
        data: Input data series
        method: Outlier detection method ('iqr' or 'zscore')
        threshold: Threshold for outlier detection
        
    Returns:
        Dictionary with outlier information
    """
    clean_data = data.dropna()
    
    if method == "iqr":
        q1 = np.percentile(clean_data, 25)
        q3 = np.percentile(clean_data, 75)
        iqr = q3 - q1
        lower_bound = q1 - threshold * iqr
        upper_bound = q3 + threshold * iqr
        
        outliers = clean_data[(clean_data < lower_bound) | (clean_data > upper_bound)]
        
    elif method == "zscore":
        z_scores = np.abs(stats.zscore(clean_data))
        outliers = clean_data[z_scores > threshold]
    
    else:
        raise ValueError("Method must be 'iqr' or 'zscore'")
    
    return {
        "method": method,
        "threshold": threshold,
        "outlier_count": len(outliers),
        "outlier_percentage": len(outliers) / len(clean_data) * 100 if len(clean_data) > 0 else 0,
        "outlier_values": outliers.tolist(),
        "outlier_indices": outliers.index.tolist()
    }

def calculate_confidence_interval(data: pd.Series, confidence: float = 0.95) -> Dict[str, Any]:
    """
    Calculate confidence interval for the mean.
    
    Args:
        data: Input data series
        confidence: Confidence level (0.95 for 95%)
        
    Returns:
        Dictionary with confidence interval
    """
    clean_data = data.dropna()
    n = len(clean_data)
    
    if n < 2:
        return {
            "n": n,
            "confidence": confidence,
            "message": "Insufficient data for confidence interval"
        }
    
    mean = np.mean(clean_data)
    std_err = stats.sem(clean_data)
    
    if std_err > 0:
        ci = stats.t.interval(confidence, n-1, loc=mean, scale=std_err)
        margin_of_error = (ci[1] - ci[0]) / 2
    else:
        ci = (mean, mean)
        margin_of_error = 0
    
    return {
        "n": n,
        "confidence": confidence,
        "mean": mean,
        "std_error": std_err,
        "ci_lower": ci[0],
        "ci_upper": ci[1],
        "margin_of_error": margin_of_error
    }