import pandas as pd
import numpy as np
from typing import Dict, Any

def validate_data(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Validate uploaded data for process capability analysis.
    
    Args:
        df: Input DataFrame
        
    Returns:
        Dictionary with validation results
    """
    result = {
        "is_valid": True,
        "message": "",
        "warnings": [],
        "issues": []
    }
    
    # Check if DataFrame is empty
    if df.empty:
        result["is_valid"] = False
        result["message"] = "DataFrame is empty"
        return result
    
    # Check for numeric columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) == 0:
        result["is_valid"] = False
        result["message"] = "No numeric columns found"
        return result
    
    # Check for missing values
    missing_counts = df.isnull().sum()
    if missing_counts.sum() > 0:
        missing_info = {}
        for col, count in missing_counts.items():
            if count > 0:
                missing_info[col] = count
        result["warnings"].append(f"Missing values found: {missing_info}")
    
    # Check sample size
    if len(df) < 10:
        result["warnings"].append("Sample size is small (n < 10), results may be unstable")
    elif len(df) < 30:
        result["warnings"].append("Sample size is moderate (n < 30), consider collecting more data")
    
    # Check for constant columns
    constant_cols = []
    for col in numeric_cols:
        if df[col].nunique() <= 1:
            constant_cols.append(col)
    
    if constant_cols:
        result["warnings"].append(f"Constant columns found: {constant_cols}")
    
    # Check for extreme values
    for col in numeric_cols:
        col_data = df[col].dropna()
        if len(col_data) > 0:
            q1 = col_data.quantile(0.25)
            q3 = col_data.quantile(0.75)
            iqr = q3 - q1
            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr
            
            outliers = col_data[(col_data < lower_bound) | (col_data > upper_bound)]
            if len(outliers) > 0:
                result["warnings"].append(
                    f"Potential outliers found in {col}: {len(outliers)} values outside IQR range"
                )
    
    if result["warnings"]:
        result["message"] = "Data validation completed with warnings"
    else:
        result["message"] = "Data validation passed"
    
    return result

def validate_specs(specs: Dict[str, Any]) -> Dict[str, Any]:
    """
    Validate specification limits.
    
    Args:
        specs: Dictionary containing LSL, USL, target
        
    Returns:
        Dictionary with validation results
    """
    result = {
        "is_valid": True,
        "message": "",
        "warnings": [],
        "spec_type": None  # 'one_sided_lower', 'one_sided_upper', 'two_sided'
    }
    
    lsl = specs.get("LSL")
    usl = specs.get("USL")
    target = specs.get("target")
    
    # Check if at least one specification is provided
    if lsl is None and usl is None:
        result["is_valid"] = False
        result["message"] = "At least one specification limit (LSL or USL) must be provided"
        return result
    
    # Determine specification type
    if lsl is not None and usl is not None:
        result["spec_type"] = "two_sided"
        if lsl >= usl:
            result["is_valid"] = False
            result["message"] = "LSL must be less than USL"
            return result
    elif lsl is not None:
        result["spec_type"] = "one_sided_lower"
    elif usl is not None:
        result["spec_type"] = "one_sided_upper"
    
    # Check target validity
    if target is not None:
        if result["spec_type"] == "two_sided":
            if not (lsl <= target <= usl):
                result["warnings"].append("Target value is outside specification limits")
        elif result["spec_type"] == "one_sided_lower":
            if target < lsl:
                result["warnings"].append("Target value is below lower specification limit")
        elif result["spec_type"] == "one_sided_upper":
            if target > usl:
                result["warnings"].append("Target value is above upper specification limit")
    
    if result["warnings"]:
        result["message"] = "Specification validation completed with warnings"
    else:
        result["message"] = "Specification validation passed"
    
    return result

def check_normality(data: pd.Series, alpha: float = 0.05) -> Dict[str, Any]:
    """
    Perform normality test on data.
    
    Args:
        data: Input data series
        alpha: Significance level
        
    Returns:
        Dictionary with normality test results
    """
    from scipy import stats
    
    result = {
        "is_normal": False,
        "p_value": None,
        "test_statistic": None,
        "critical_values": None,
        "significance_level": alpha
    }
    
    # Remove missing values
    clean_data = data.dropna()
    
    if len(clean_data) < 3:
        result["message"] = "Insufficient data for normality test (n < 3)"
        return result
    
    # Anderson-Darling test
    ad_result = stats.anderson(clean_data, dist='norm')
    
    result["test_statistic"] = ad_result.statistic
    result["critical_values"] = ad_result.critical_values
    result["significance_levels"] = ad_result.significance_level
    
    # Check if test statistic exceeds critical value at specified alpha
    critical_index = np.where(ad_result.significance_level == alpha * 100)[0]
    if len(critical_index) > 0:
        critical_value = ad_result.critical_values[critical_index[0]]
        result["is_normal"] = ad_result.statistic < critical_value
        result["p_value"] = None  # AD test doesn't provide exact p-values
    
    result["message"] = "Data appears normal" if result["is_normal"] else "Data does not appear normal"
    
    return result