import numpy as np
import pandas as pd
from typing import Dict, Any

def check_process_stability(data: pd.Series, window_size: int = None) -> Dict[str, Any]:
    """
    Perform a simple process stability check using statistical tests.
    
    Args:
        data: Input data series
        window_size: Window size for moving range calculation
        
    Returns:
        Dictionary with stability assessment
    """
    clean_data = data.dropna()
    n = len(clean_data)
    
    result = {
        "is_stable": True,
        "message": "Process appears stable",
        "tests_performed": [],
        "warnings": []
    }
    
    if n < 10:
        result["is_stable"] = False
        result["message"] = "Insufficient data for stability assessment (n < 10)"
        return result
    
    # Test 1: Check for excessive variation using I-MR logic
    try:
        individual_values = clean_data.values
        moving_ranges = np.abs(np.diff(individual_values))
        
        # Calculate control limits using moving range
        mr_bar = np.mean(moving_ranges)
        ucl_individual = np.mean(individual_values) + 2.66 * mr_bar
        lcl_individual = np.mean(individual_values) - 2.66 * mr_bar
        ucl_mr = 3.27 * mr_bar
        
        # Check for out-of-control points
        out_of_control_individual = np.sum((individual_values > ucl_individual) | 
                                         (individual_values < lcl_individual))
        out_of_control_mr = np.sum(moving_ranges > ucl_mr)
        
        if out_of_control_individual > 0 or out_of_control_mr > 0:
            result["is_stable"] = False
            result["warnings"].append(f"Found {out_of_control_individual} individual values and {out_of_control_mr} moving ranges outside control limits")
            
        result["tests_performed"].append("I-MR Control Limits")
        
    except Exception as e:
        result["warnings"].append(f"I-MR test failed: {str(e)}")
    
    # Test 2: Check for trends using run test
    try:
        median_val = np.median(individual_values)
        above_median = individual_values > median_val
        
        # Count runs (consecutive points on same side of median)
        runs = 1
        for i in range(1, len(above_median)):
            if above_median[i] != above_median[i-1]:
                runs += 1
        
        # Expected runs and standard deviation
        n_above = np.sum(above_median)
        n_below = n - n_above
        
        if n_above > 0 and n_below > 0:
            expected_runs = (2 * n_above * n_below) / n + 1
            runs_std = np.sqrt((2 * n_above * n_below * (2 * n_above * n_below - n)) / 
                              (n * n * (n - 1)))
            
            # Check if too few runs (indicates trend)
            if runs_std > 0:
                z_score = (runs - expected_runs) / runs_std
                if z_score < -2:  # Too few runs, possible trend
                    result["is_stable"] = False
                    result["warnings"].append(f"Possible trend detected (runs test z-score: {z_score:.2f})")
        
        result["tests_performed"].append("Runs Test")
        
    except Exception as e:
        result["warnings"].append(f"Runs test failed: {str(e)}")
    
    # Test 3: Check for shifts using split-half test
    try:
        mid_point = n // 2
        first_half_mean = np.mean(individual_values[:mid_point])
        second_half_mean = np.mean(individual_values[mid_point:])
        
        # Use pooled standard deviation for shift detection
        pooled_std = np.std(individual_values, ddof=1)
        
        if pooled_std > 0:
            shift_magnitude = abs(second_half_mean - first_half_mean) / pooled_std
            
            if shift_magnitude > 2:  # Significant shift detected
                result["is_stable"] = False
                result["warnings"].append(f"Possible process shift detected (magnitude: {shift_magnitude:.2f} sigma)")
        
        result["tests_performed"].append("Split-Half Shift Test")
        
    except Exception as e:
        result["warnings"].append(f"Shift test failed: {str(e)}")
    
    # Compile final message
    if not result["is_stable"]:
        if len(result["warnings"]) > 1:
            result["message"] = "Multiple stability issues detected"
        elif len(result["warnings"]) == 1:
            result["message"] = result["warnings"][0]
        else:
            result["message"] = "Process stability questionable"
    
    return result