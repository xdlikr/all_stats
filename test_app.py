#!/usr/bin/env python3
"""
Test script for the Process Capability Analysis application.
This script tests the core functionality without running the Streamlit app.
"""

import numpy as np
import pandas as pd
from services.statistics import calculate_basic_statistics
from services.distributions import compare_distributions
from services.capability import calculate_capability_metrics
from services.tolerance import calculate_tolerance_intervals
from services.oos import compare_oos_methods

def test_basic_functionality():
    """Test the basic functionality with sample data."""
    print("🧪 Testing Process Capability Analysis Application...")
    
    # Create sample normal data
    np.random.seed(42)
    sample_data = np.random.normal(loc=10.0, scale=1.0, size=100)
    data_series = pd.Series(sample_data, name="value")
    
    print(f"📊 Sample data: n={len(sample_data)}, mean={np.mean(sample_data):.3f}, std={np.std(sample_data):.3f}")
    
    # Test 1: Basic statistics
    print("\n1. Testing basic statistics...")
    stats = calculate_basic_statistics(data_series)
    print(f"   Sample size: {stats['n']}")
    print(f"   Mean: {stats['mean']:.3f}")
    print(f"   Std Dev: {stats['std_dev']:.3f}")
    print(f"   Skewness: {stats['skewness']:.3f}")
    
    # Test 2: Distribution fitting
    print("\n2. Testing distribution fitting...")
    dist_comparison = compare_distributions(data_series)
    print(f"   Fitted {dist_comparison['n_distributions']} distributions")
    best_dist = dist_comparison['best_distribution']
    print(f"   Best distribution: {best_dist}")
    
    if best_dist:
        best_result = dist_comparison['results'][best_dist]
        print(f"   AIC: {best_result['aic']:.3f}")
        print(f"   Parameters: {best_result['params']}")
    
    # Test 3: Capability metrics
    print("\n3. Testing capability metrics...")
    if best_dist:
        capability = calculate_capability_metrics(
            data_series, best_dist, best_result['params'],
            lsl=7.0, usl=13.0, target=10.0
        )
        
        if capability['success']:
            cp = capability.get('Cp', 'N/A')
            cpk = capability.get('Cpk', 'N/A')
            pp = capability.get('Pp', 'N/A')
            ppk = capability.get('Ppk', 'N/A')
            print(f"   Cp: {cp:.3f}" if cp != 'N/A' else "   Cp: N/A")
            print(f"   Cpk: {cpk:.3f}" if cpk != 'N/A' else "   Cpk: N/A")
            print(f"   Pp: {pp:.3f}" if pp != 'N/A' else "   Pp: N/A")
            print(f"   Ppk: {ppk:.3f}" if ppk != 'N/A' else "   Ppk: N/A")
        else:
            print(f"   Capability calculation failed: {capability['message']}")
    
    # Test 4: Tolerance intervals
    print("\n4. Testing tolerance intervals...")
    ti_result = calculate_tolerance_intervals(sample_data, p=0.95, conf=0.95)
    
    if ti_result['success']:
        lower, upper = ti_result['interval']
        print(f"   {ti_result['type']} tolerance interval: [{lower:.3f}, {upper:.3f}]")
        print(f"   Tolerance factor: {ti_result['factor']:.3f}")
    else:
        print(f"   Tolerance interval failed: {ti_result['message']}")
    
    # Test 5: %OOS methods
    print("\n5. Testing %OOS methods...")
    if best_dist:
        oos_comparison = compare_oos_methods(
            data_series, best_dist, best_result['params'],
            lsl=7.0, usl=13.0
        )
        
        print(f"   Compared {oos_comparison['n_methods']} methods:")
        for result in oos_comparison['comparison_table']:
            if result['Success']:
                print(f"     {result['Method']}: {result['%OOS']:.3f}%")
            else:
                print(f"     {result['Method']}: Failed - {result['Message']}")
    
    print("\n✅ All tests completed successfully!")
    print("\nTo run the full application:")
    print("   streamlit run app_new.py")

if __name__ == "__main__":
    test_basic_functionality()