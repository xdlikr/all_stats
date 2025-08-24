import numpy as np
import pandas as pd
from scipy import stats
from typing import Dict, Any, Optional, Tuple
import warnings

def calculate_tolerance_intervals_montes(
    data: pd.DataFrame,
    value_col: str,
    lot_col: str,
    conf_level: float = 0.95,
    prop_cover: float = 0.95,
    lsl: Optional[float] = None,
    usl: Optional[float] = None
) -> Dict[str, Any]:
    """
    Calculate tolerance intervals using the NoSlope method from TI_MONTES.r
    
    This function implements the random effects model approach for tolerance intervals
    as described in Richard K. Burdick et al.'s paper, applicable to data without
    time trends (slope = 0).
    
    Args:
        data: DataFrame containing the data
        value_col: Column name containing the measurement values (Y)
        lot_col: Column name containing the lot identifiers (Lot)
        conf_level: Confidence level (e.g., 0.95)
        prop_cover: Proportion of population to cover (e.g., 0.99)
        
    Returns:
        Dictionary with tolerance interval results
    """
    
    result = {
        "success": False,
        "method": "NoSlope (TI_MONTES)",
        "conf_level": conf_level,
        "prop_cover": prop_cover,
        "message": ""
    }
    
    try:
        # Data preparation
        df = data[[value_col, lot_col]].dropna()
        if len(df) < 2:
            result["message"] = "Insufficient data for tolerance interval calculation"
            return result
            
        Y = df[value_col].values
        Lot = df[lot_col].astype('category')
        
        # Calculate normal distribution Z-scores
        Z_beta = stats.norm.ppf(prop_cover)
        Z_gamma = stats.norm.ppf(conf_level) 
        Z_beta2 = stats.norm.ppf((1 + prop_cover) / 2)  # For two-sided interval
        
        # Calculate intermediate statistics (Recreate Table I)
        I = len(Lot.cat.categories)  # Number of lots
        s = I - 1  # Lot degrees of freedom
        
        if s <= 0:
            result["message"] = "Need at least 2 lots for calculation"
            return result
        
        # Count observations per lot
        J_i = df.groupby(lot_col)[value_col].count().values
        r = np.sum(J_i) - I  # Error degrees of freedom
        
        if r <= 0:
            result["message"] = "Insufficient within-lot replicates"
            return result
        
        # Harmonic mean of replicate counts
        J_H = I / np.sum(1.0 / J_i)
        
        # Lot means and overall mean
        Ybar_i = df.groupby(lot_col)[value_col].mean().values
        Ybar_star = np.mean(Ybar_i)  # Unweighted mean of lot means
        
        # Lot-to-lot variance (sample variance of lot means)
        S_sq_L = np.var(Ybar_i, ddof=1)
        
        # Within-lot variances and pooled error variance
        lot_vars = df.groupby(lot_col)[value_col].var(ddof=1).fillna(0).values
        S_sq_E = np.sum((J_i - 1) * lot_vars) / r
        
        # Variance estimates
        Var_hat_Ybar_star = S_sq_L / I
        Var_hat_Y = S_sq_L + (1 - 1/J_H) * S_sq_E
        n_E = Var_hat_Y / Var_hat_Ybar_star  # Effective sample size
        
        # MLS method calculation of upper limit for Y variance
        # (Modified Large-Sample method)
        alpha = 1 - conf_level
        H1 = s / stats.chi2.ppf(alpha, s) - 1
        H2 = r / stats.chi2.ppf(alpha, r) - 1
        
        # Upper limit for Y variance
        U = Var_hat_Y + np.sqrt((H1 * S_sq_L)**2 + (H2 * (1 - 1/J_H) * S_sq_E)**2)
        
        # Calculate various tolerance intervals
        
        # 1. One-sided tolerance interval (based on Hoffman 2010)
        L1 = Ybar_star - Z_beta * np.sqrt(U) - Z_gamma * np.sqrt(Var_hat_Ybar_star)
        U1 = Ybar_star + Z_beta * np.sqrt(U) + Z_gamma * np.sqrt(Var_hat_Ybar_star)
        
        # 2. Two-sided tolerance interval (based on Hoffman & Kringle 2005)
        L2 = Ybar_star - Z_beta2 * np.sqrt(1 + 1/n_E) * np.sqrt(U)
        U2 = Ybar_star + Z_beta2 * np.sqrt(1 + 1/n_E) * np.sqrt(U)
        
        # 3. One-sided tolerance interval (using HK1 adjustment)
        L1_HK1 = Ybar_star - Z_beta * np.sqrt(1 + 1/n_E) * np.sqrt(U)
        U1_HK1 = Ybar_star + Z_beta * np.sqrt(1 + 1/n_E) * np.sqrt(U)
        
        # Store results
        result.update({
            "success": True,
            "message": f"Tolerance intervals calculated using NoSlope method for {I} lots",
            
            # Data characteristics
            "n_lots": I,
            "n_total": len(Y),
            "lot_df": s,
            "error_df": r,
            "harmonic_mean_reps": J_H,
            "effective_sample_size": n_E,
            
            # Variance components
            "ybar_star": Ybar_star,
            "lot_variance": S_sq_L,
            "error_variance": S_sq_E,
            "total_variance": Var_hat_Y,
            "variance_upper_limit": U,
            
            # Z-scores used
            "z_beta": Z_beta,
            "z_gamma": Z_gamma,
            "z_beta2": Z_beta2,
            
            # Tolerance interval results (6 methods as in R)
            "one_sided_lower": L1,
            "one_sided_upper": U1,
            "two_sided_lower": L2, 
            "two_sided_upper": U2,
            "one_sided_hk1_lower": L1_HK1,
            "one_sided_hk1_upper": U1_HK1,
            
            # Determine which interval to use as default based on specifications
            "interval": None,
            "type": None,
            "factor": None,
        })
        
        # Choose appropriate default interval based on specifications
        if lsl is not None and usl is not None:
            # Two-sided specifications - use two-sided TI
            result["interval"] = (L2, U2)
            result["type"] = "two-sided"
            result["factor"] = Z_beta2 * np.sqrt(1 + 1/n_E)
        elif lsl is not None or usl is not None:
            # One-sided specifications - use one-sided TI
            result["interval"] = (L1, U1) 
            result["type"] = "one-sided"
            result["factor"] = Z_beta  # For one-sided, factor is just Z_beta part
        else:
            # No specifications provided - default to two-sided
            result["interval"] = (L2, U2)
            result["type"] = "two-sided"
            result["factor"] = Z_beta2 * np.sqrt(1 + 1/n_E)
        
    except Exception as e:
        result["message"] = f"Error calculating tolerance intervals: {str(e)}"
        
    return result

def coverage_scan_montes(
    data: pd.DataFrame,
    value_col: str,
    lot_col: str,
    lsl: Optional[float] = None,
    usl: Optional[float] = None,
    p_min: float = 0.80,
    p_max: float = 0.995,
    p_step: float = 0.005,
    conf_level: float = 0.95
) -> Dict[str, Any]:
    """
    Perform coverage scan using the NoSlope method to find maximum coverage
    where tolerance interval is within specifications.
    """
    
    result = {
        "success": False,
        "message": "Coverage scan failed",
        "n_lots": 0
    }
    
    if lsl is None and usl is None:
        result["message"] = "No specification limits provided"
        return result
    
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
            ti_result = calculate_tolerance_intervals_montes(
                data, value_col, lot_col, conf_level, p_val
            )
            
            if ti_result["success"]:
                if interval_type == "two-sided":
                    lower_bound = ti_result["two_sided_lower"]
                    upper_bound = ti_result["two_sided_upper"]
                    
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
                    # One-sided case
                    if usl is not None:
                        # Upper one-sided
                        upper_bound = ti_result["one_sided_upper"]
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
                        lower_bound = ti_result["one_sided_lower"]
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
    
    if results:
        result.update({
            "success": True,
            "message": "Coverage scan completed using NoSlope method",
            "n_lots": results[0].get("n_lots", 0) if results else 0,
            "interval_type": interval_type,
            "p_star": p_star,
            "oos_ti": oos_ti,
            "results": results,
            "scan_range": (p_min, p_max, p_step)
        })
    
    return result