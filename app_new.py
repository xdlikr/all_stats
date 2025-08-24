import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime

# Configure Streamlit page
st.set_page_config(
    page_title="Process Capability Analysis Platform",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Apple-style Custom CSS
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=SF+Pro+Display:wght@300;400;500;600;700&display=swap');

/* Clean spacing reduction - no negative margins */

/* Global App Styling */
.stApp {
    padding-top: 0px !important;
}

/* Main containers */
.main .block-container {
    padding-top: 1rem !important;
    padding-left: 5rem;
    padding-right: 5rem;
    max-width: none;
}

/* Hide unnecessary Streamlit elements */
div[data-testid="stToolbar"],
#MainMenu {
    display: none !important;
}

/* Sidebar optimizations */
.stSidebar .block-container {
    padding-top: 0.25rem !important;
}

.stSidebar h3 {
    margin-top: 0rem !important;
    padding-top: 0rem !important;
}

.stFileUploader {
    margin-top: 0rem !important;
}

.stSidebar .element-container:first-child {
    margin-top: 0rem !important;
}

/* Move sidebar content up further */
.stSidebar > div > div {
    padding-top: 0rem !important;
}

/* Target specific sidebar containers */
section[data-testid="stSidebar"] > div:first-child {
    padding-top: 0.25rem !important;
}

/* Remove default spacing from sidebar elements */
.stSidebar .stMarkdown {
    margin-top: 0rem !important;
}

.stSidebar .stMarkdown h3:first-child {
    margin-top: 0rem !important;
}

/* Simple, clean Header */
.apple-header {
    background: #ffffff;
    border: 1px solid #e5e5e7;
    border-radius: 8px;
    padding: 1.5rem 2rem;
    margin: 1rem 0;
    box-shadow: 0 2px 8px rgba(0, 0, 0, 0.08);
}

.apple-header h1 {
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
    font-weight: 600;
    font-size: 2rem;
    margin: 0;
    color: #1d1d1f;
    line-height: 1.2;
}

.apple-header p {
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
    font-weight: 400;
    font-size: 1rem;
    margin: 0.5rem 0 0 0;
    color: #86868b;
    line-height: 1.4;
}

.header-meta {
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
    font-weight: 500;
    font-size: 0.875rem;
    margin: 0.75rem 0 0 0;
    color: #424245;
    opacity: 0.8;
}

/* Apple-style Tabs */
.stTabs [data-baseweb="tab-list"] {
    gap: 8px;
    background: rgba(244, 244, 246, 0.6);
    border-radius: 12px;
    padding: 4px;
    border: none;
    box-shadow: inset 0 1px 3px rgba(0, 0, 0, 0.1);
}

.stTabs [data-baseweb="tab"] {
    background: transparent;
    border-radius: 8px;
    color: #6b7280;
    font-weight: 500;
    border: none;
    padding: 0.75rem 1.5rem;
    font-family: 'SF Pro Display', -apple-system, BlinkMacSystemFont, sans-serif;
    transition: all 0.2s ease;
}

.stTabs [aria-selected="true"] {
    background: white;
    color: #1d1d1f;
    box-shadow: 0 1px 3px rgba(0, 0, 0, 0.1), 
                0 1px 2px rgba(0, 0, 0, 0.06);
}

/* Subheaders */
.stApp h3 {
    font-family: 'SF Pro Display', -apple-system, BlinkMacSystemFont, sans-serif;
    font-weight: 600;
    color: #1d1d1f;
    font-size: 1.5rem;
    margin-bottom: 1rem;
}

/* Card-style containers */
.apple-card {
    background: white;
    border-radius: 12px;
    padding: 0.75rem 1.5rem 0.25rem 1.5rem;
    box-shadow: 0 1px 3px rgba(0, 0, 0, 0.1), 
                0 1px 2px rgba(0, 0, 0, 0.06);
    border: 1px solid rgba(0, 0, 0, 0.05);
    margin-bottom: 0.75rem;
}

.apple-card h3 {
    margin-bottom: 0.25rem !important;
    margin-top: 0 !important;
    padding-top: 0 !important;
    line-height: 1.2;
}
</style>
""", unsafe_allow_html=True)

def initialize_session_state():
    """Initialize session state variables."""
    if 'data' not in st.session_state:
        st.session_state.data = None
    if 'specs' not in st.session_state:
        st.session_state.specs = {"LSL": None, "USL": None}
    if 'selected_distribution' not in st.session_state:
        st.session_state.selected_distribution = None
    if 'distribution_params' not in st.session_state:
        st.session_state.distribution_params = None

def render_sidebar():
    """Render the sidebar with all user controls."""
    with st.sidebar:
        # Add CSS for better file uploader styling
        st.markdown("""
        <style>
        .stFileUploader > div > div > div > div {
            background-color: #f8f9fa;
            border: 2px dashed #dee2e6;
            border-radius: 8px;
            padding: 1rem;
        }
        </style>
        """, unsafe_allow_html=True)
        
        st.markdown("### 📂 Data Import")
        
        uploaded_file = st.file_uploader(
            "Choose CSV or Excel file", 
            type=["csv", "xlsx", "xls"],
            help="Upload your data file to begin analysis"
        )

        if uploaded_file:
            if st.session_state.get('uploaded_filename') != uploaded_file.name:
                with st.spinner("Loading data..."):
                    try:
                        if uploaded_file.name.endswith('.csv'):
                            df = pd.read_csv(uploaded_file)
                        else:
                            df = pd.read_excel(uploaded_file)

                        st.session_state.data = df
                        st.session_state.specs = {"LSL": None, "USL": None}
                        st.session_state.selected_distribution = None
                        st.session_state.uploaded_filename = uploaded_file.name
                        # Clear cached results when new data is loaded
                        if 'ti_results' in st.session_state:
                            del st.session_state.ti_results
                    except Exception as e:
                        st.error(f"❌ Error loading file: {str(e)}")

        st.markdown("---")

        if st.session_state.data is not None:
            # Add CSS for better sidebar styling
            st.markdown("""
            <style>
            .stNumberInput > div > div > input {
                text-align: center;
            }
            .stButton > button {
                height: 2.5rem;
                margin-top: 1.8rem;
            }
            .small-metric [data-testid="metric-container"] > div:first-child {
                font-size: 0.75rem !important;
            }
            .small-metric [data-testid="metric-container"] > div:nth-child(2) {
                font-size: 1.1rem !important;
            }
            </style>
            """, unsafe_allow_html=True)
            
            st.markdown("### 📊 Column Selection")
            
            # Consolidate all column selections in one section
            numeric_columns = st.session_state.data.select_dtypes(include=[np.number]).columns.tolist()
            categorical_cols = st.session_state.data.select_dtypes(include=['object', 'category']).columns.tolist()
            
            current_col = st.session_state.specs.get("value_col")
            current_index = numeric_columns.index(current_col) if current_col in numeric_columns else 0

            value_col = st.selectbox(
                "Analysis Column (Numeric)", 
                numeric_columns,
                index=current_index,
                help="Select the numeric column for statistical analysis"
            )
            
            lot_col = st.selectbox(
                "Lot/Batch Column (Categorical)",
                [None] + categorical_cols,
                help="Required for tolerance intervals - groups data by lot/batch"
            )
            
            st.markdown("")  # Add some spacing
            
            # Add LSL/USL to column selection section
            st.markdown("**Specification Limits**")
            spec_col1, spec_col2, spec_col3 = st.columns([3, 3, 1.2])
            with spec_col1:
                lsl = st.number_input(
                    "LSL", 
                    value=st.session_state.specs.get("LSL"), 
                    placeholder="Lower", 
                    key="sidebar_lsl",
                    format="%.3f",
                    help="Lower Specification Limit"
                )
            with spec_col2:
                usl = st.number_input(
                    "USL", 
                    value=st.session_state.specs.get("USL"), 
                    placeholder="Upper", 
                    key="sidebar_usl",
                    format="%.3f",
                    help="Upper Specification Limit"
                )
            with spec_col3:
                if st.button("🗑️", help="Clear specs", key="clear_all_specs", use_container_width=True):
                    st.session_state.specs.update({"LSL": None, "USL": None})
                    st.rerun()
            
            if (st.session_state.specs.get("value_col") != value_col or 
                st.session_state.specs.get("lot_col") != lot_col):
                st.session_state.specs["value_col"] = value_col
                st.session_state.specs["lot_col"] = lot_col
                # Clear cached results when columns change
                if 'ti_results' in st.session_state:
                    del st.session_state.ti_results
            
            st.markdown("---")

            st.markdown("### 📈 Quick Stats")
            data_series = st.session_state.data[value_col].dropna()
            
            
            # More compact stats with smaller font
            stat_col1, stat_col2 = st.columns(2)
            with stat_col1:
                st.markdown('<div class="small-metric">', unsafe_allow_html=True)
                st.metric("n", len(data_series), help="Sample size")
                st.metric("σ", f"{data_series.std():.3f}", help="Standard deviation")
                st.markdown('</div>', unsafe_allow_html=True)
            with stat_col2:
                st.markdown('<div class="small-metric">', unsafe_allow_html=True)
                st.metric("x̄", f"{data_series.mean():.3f}", help="Sample mean")  
                st.metric("Range", f"{data_series.max() - data_series.min():.3f}", help="Max - Min")
                st.markdown('</div>', unsafe_allow_html=True)

            st.markdown("---")

            # Handle spec updates and validation
            current_specs = st.session_state.specs
            if (current_specs.get("LSL") != lsl or current_specs.get("USL") != usl):
                st.session_state.specs.update({"LSL": lsl, "USL": usl})
                # Clear cached results when specs change
                if 'ti_results' in st.session_state:
                    del st.session_state.ti_results
                st.rerun()

            # Validation and status
            if lsl is not None and usl is not None and lsl >= usl:
                st.error("❌ LSL must be less than USL")
            elif lsl is not None or usl is not None:
                specs_status = []
                if lsl is not None: specs_status.append(f"LSL: {lsl:.3f}")
                if usl is not None: specs_status.append(f"USL: {usl:.3f}")
                st.success("✅ " + " | ".join(specs_status))

def render_overview_tab():
    """Render the main data visualization tab with distribution fits."""
    
    if 'value_col' not in st.session_state.specs or st.session_state.specs['value_col'] is None:
        st.info("⬅️ Please upload data and select a column in the sidebar to begin the analysis.")
        return
        
    data_series = st.session_state.data[st.session_state.specs['value_col']].dropna()
    
    from services.distributions import compare_distributions, get_distribution_fit_quality
    from services.visualizations import create_merged_distribution_plot

    with st.spinner("Fitting distributions for overview..."):
        comparison = compare_distributions(data_series)

    if comparison["n_distributions"] == 0:
        st.error("❌ No distributions could be fitted to the data.")
        return

    st.success(f"✅ Fitted {comparison['n_distributions']} distributions")

    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("#### Histogram with Distribution Fits")
        merged_fig = create_merged_distribution_plot(data_series, comparison)
        st.plotly_chart(merged_fig, use_container_width=True)

    with col2:
        st.markdown("#### Distribution Comparison")
        comp_df = pd.DataFrame(comparison["comparison_table"])
        comp_df["Quality"] = comp_df.apply(
            lambda row: get_distribution_fit_quality(row["AIC"], row["KS p-value"]),
            axis=1
        )
        
        display_cols = ["Distribution", "AIC", "BIC", "KS p-value", "Quality"]
        display_df = comp_df[display_cols].copy()
        
        for col in ["AIC", "BIC", "KS p-value"]:
            if col in display_df.columns:
                display_df[col] = display_df[col].apply(lambda x: f"{x:.3f}" if pd.notnull(x) else "N/A")
        
        st.dataframe(display_df, use_container_width=True)

        st.markdown("#### Select Distribution for Analysis")
        best_dist = comparison["best_distribution"]
        dist_options = comp_df["Distribution"].tolist()
        
        current_selection = st.session_state.get('selected_distribution', best_dist)
        current_index = dist_options.index(current_selection) if current_selection in dist_options else 0

        selected_dist = st.selectbox(
            "Choose the distribution to use in other tabs:",
            dist_options,
            index=current_index
        )
        
        if st.session_state.get('selected_distribution') != selected_dist:
            st.session_state.selected_distribution = selected_dist
            st.session_state.distribution_params = comp_df[comp_df["Distribution"] == selected_dist]["Parameters"].iloc[0]
            # Clear cached results when distribution changes
            if 'ti_results' in st.session_state:
                del st.session_state.ti_results
            st.rerun()

def render_capability_tab():
    """Render integrated capability analysis with tolerance intervals"""
    if not all([st.session_state.data is not None, 
                st.session_state.specs.get("LSL") is not None or st.session_state.specs.get("USL") is not None]):
        st.warning("⬅️ Please set specification limits in the sidebar.")
        return
    
    
    value_col = st.session_state.specs.get("value_col")
    data_series = st.session_state.data[value_col].dropna()
    lsl = st.session_state.specs.get("LSL")
    usl = st.session_state.specs.get("USL")
    
    # Import required functions
    from services.capability import (
        calculate_normal_ppk, 
        calculate_fitted_distribution_ppk, 
        calculate_empirical_ppk
    )
    from services.tolerance_montes import calculate_tolerance_intervals_montes, coverage_scan_montes
    from utils.validators import check_normality
    
    # Left-Right Layout: Results | Methods Description
    left_col, right_col = st.columns([3, 2])
    
    # === LEFT SIDE: COMPACT RESULTS ===
    with left_col:
        # Calculate all methods upfront
        normal_result = calculate_normal_ppk(data_series, lsl, usl)
        empirical_result = calculate_empirical_ppk(data_series, lsl, usl)
        
        fitted_result = None
        distribution_name = st.session_state.selected_distribution
        distribution_params = st.session_state.distribution_params
        if distribution_name and distribution_params:
            fitted_result = calculate_fitted_distribution_ppk(
                data_series, distribution_name, distribution_params, lsl, usl
            )
        
        # Compact header with all key info
        normality = check_normality(data_series)
        normal_status = "⚠️ Non-normal" if not normality["is_normal"] else "✓ Normal"
        st.markdown(f"### 📊 Process Analysis Results  `n={len(data_series)} | {normal_status}`")
        
        # --- Process Capability with OOS (ultra-compact) ---
        from services.oos import calculate_oos
        
        cap_col1, cap_col2, cap_col3 = st.columns(3)
        
        # Calculate OOS for each method
        normal_oos = calculate_oos(data_series, method="normal", lsl=lsl, usl=usl)
        empirical_oos = calculate_oos(data_series, method="empirical", lsl=lsl, usl=usl)
        fitted_oos = None
        if distribution_name and distribution_params:
            fitted_oos = calculate_oos(data_series, method="fitted", distribution_name=distribution_name, distribution_params=distribution_params, lsl=lsl, usl=usl)
        
        for i, (result, name, col, oos_result) in enumerate([
            (normal_result, "Normal", cap_col1, normal_oos),
            (empirical_result, "Empirical", cap_col2, empirical_oos), 
            (fitted_result, distribution_name.capitalize() if distribution_name else "Fitted", cap_col3, fitted_oos)
        ]):
            with col:
                if result and result.get("success") and result.get('Ppk') is not None:
                    # Display Ppk with OOS in parentheses
                    oos_text = ""
                    if oos_result and oos_result.get("success"):
                        oos_text = f" ({oos_result['oos_percentage']:.3f}%)"
                    st.metric(f"{name} Ppk", f"{result['Ppk']:.3f}{oos_text}")
                    
                    # Ultra-compact: combine all info with method explanation
                    info_parts = []
                    if result.get('Pp') is not None:
                        info_parts.append(f"Pp: {result['Pp']:.3f}")
                    if result.get('Ppk_ci_lower') is not None:
                        # Add method explanation for CI
                        if name == "Normal":
                            info_parts.append(f"CI: [{result['Ppk_ci_lower']:.3f}, {result['Ppk_ci_upper']:.3f}] (Bissell)")
                        elif name == "Empirical":
                            info_parts.append(f"CI: [{result['Ppk_ci_lower']:.3f}, {result['Ppk_ci_upper']:.3f}] (Bootstrap)")
                        else:
                            info_parts.append(f"CI: [{result['Ppk_ci_lower']:.3f}, {result['Ppk_ci_upper']:.3f}] (Bootstrap)")
                    if info_parts:
                        st.caption(" | ".join(info_parts))
                else:
                    st.metric(f"{name} Ppk", "N/A")
                    if name == "Fitted" and not distribution_name:
                        st.caption("Need distribution")
        
        # Add OOS explanation
        st.caption("💡 **%OOS values** in parentheses show the percentage of production expected **Out-Of-Specification**")
        
        # --- Tolerance Intervals Section ---
        st.markdown("---")
        st.markdown("### 🎯 Tolerance Intervals Analysis")
        st.info("**TI complements Ppk**: While Ppk measures current capability, TI predicts where future production will fall using multi-lot statistical bounds.")
        
        ti_col1, ti_col2, ti_col3 = st.columns([2, 2, 3])
        with ti_col1:
            p_coverage = st.selectbox("Coverage %", [90.0, 95.0, 99.0, 99.9], index=1, key="ti_coverage") / 100
        with ti_col2:
            conf_level = st.selectbox("Conf %", [90, 95, 99], index=1, key="ti_confidence") / 100
        with ti_col3:
            calc_ti = st.button("Calculate TI (NoSlope Method)", use_container_width=True)
        
        # Check if lot column is selected
        lot_col = st.session_state.specs.get("lot_col")
        if lot_col is None:
            st.warning("⚠️ Please select a lot/batch column in the sidebar for TI calculations")
            return
        
        # Check if we need to recalculate based on parameter changes
        need_recalc = (calc_ti or 
                      'ti_results' not in st.session_state or 
                      st.session_state.ti_results.get('p_coverage') != p_coverage or
                      st.session_state.ti_results.get('conf_level') != conf_level or
                      st.session_state.ti_results.get('lsl') != lsl or
                      st.session_state.ti_results.get('usl') != usl or
                      st.session_state.ti_results.get('lot_col') != lot_col)
        
        if need_recalc:
            with st.spinner("Calculating..."):
                ti_result = calculate_tolerance_intervals_montes(
                    st.session_state.data, value_col, lot_col, conf_level, p_coverage, lsl, usl
                )
                scan_result = coverage_scan_montes(
                    st.session_state.data, value_col, lot_col, lsl, usl, conf_level=conf_level
                )
                st.session_state.ti_results = {
                    'ti_result': ti_result,
                    'scan_result': scan_result,
                    'p_coverage': p_coverage,
                    'conf_level': conf_level,
                    'lsl': lsl,  # Store current specs for comparison
                    'usl': usl,
                    'lot_col': lot_col
                }
        
        # Display tolerance results (more compact)
        if hasattr(st.session_state, 'ti_results'):
            ti_result = st.session_state.ti_results['ti_result']
            scan_result = st.session_state.ti_results['scan_result']
            
            if ti_result["success"]:
                lower, upper = ti_result["interval"]
                factor_name = 'k' if ti_result['type'] == 'two-sided' else 'g'
                
                tol_col1, tol_col2, tol_col3, tol_col4 = st.columns(4)
                
                with tol_col1:
                    st.metric("TI Lower", f"{lower:.3f}")
                with tol_col2:
                    st.metric("TI Upper", f"{upper:.3f}")
                with tol_col3:
                    st.metric(f"Factor ({factor_name})", f"{ti_result['factor']:.3f}")
                    st.caption(f"Lots: {ti_result.get('n_lots', 'N/A')}")
                with tol_col4:
                    if scan_result and scan_result["success"]:
                        st.metric("Max Coverage", f"{scan_result['p_star']*100:.1f}%")
                        st.caption(f"OOS: {scan_result['oos_ti']*100:.3f}%")
                    else:
                        st.metric("Max Coverage", "Failed")
                
                # Coverage scan visualization (compact)
                if scan_result and scan_result["success"]:
                    from services.visualizations import create_coverage_scan_plot
                    fig = create_coverage_scan_plot(scan_result, st.session_state.specs)
                    st.plotly_chart(fig, use_container_width=True, height=250)
                    
                    # Mathematical formula annotation below the chart
                    with st.expander("📊 Calculated Parameters & Formula", expanded=False):
                        # Get values for formatting
                        ybar_star = ti_result.get('ybar_star', 0)
                        n_e = ti_result.get('effective_sample_size', 0)
                        n_lots = ti_result.get('n_lots', 0)
                        z_beta = ti_result.get('z_beta', 0)
                        z_beta2 = ti_result.get('z_beta2', 0)
                        z_gamma = ti_result.get('z_gamma', 0)
                        lot_var = ti_result.get('lot_variance', 0)
                        error_var = ti_result.get('error_variance', 0)
                        var_upper = ti_result.get('variance_upper_limit', 0)
                        var_ybar_star = lot_var / n_lots if n_lots > 0 else 0
                        interval_type = ti_result.get('type', 'unknown')
                        lower, upper = ti_result["interval"]
                        
                        # Display appropriate formula and calculation
                        if interval_type == "two-sided":
                            factor = z_beta2 * (1 + 1/n_e)**0.5 if n_e > 0 else 0
                            st.markdown(f"""
                            **Two-Sided Formula Applied:**
                            $$TI = {ybar_star:.3f} \\pm {z_beta2:.3f} \\sqrt{{1 + 1/{n_e:.3f}}} \\sqrt{{{var_upper:.3f}}}$$
                            $$TI = {ybar_star:.3f} \\pm {factor:.3f} \\times {var_upper**0.5:.3f}$$
                            $$TI = [{lower:.3f}, {upper:.3f}]$$
                            """)
                        else:
                            st.markdown(f"""
                            **One-Sided Formula Applied:**
                            $$TI = {ybar_star:.3f} \\pm {z_beta:.3f} \\sqrt{{{var_upper:.3f}}} \\pm {z_gamma:.3f} \\sqrt{{{var_ybar_star:.3f}}}$$
                            $$TI = {ybar_star:.3f} \\pm {z_beta * var_upper**0.5:.3f} \\pm {z_gamma * var_ybar_star**0.5:.3f}$$
                            $$TI = [{lower:.3f}, {upper:.3f}]$$
                            """)
                        
                        st.markdown(f"""
                        **Calculated Values:**
                        | Parameter | Value | Description |
                        |-----------|-------|-------------|
                        | $I$ | {n_lots} | Number of lots |
                        | $\\bar{{Y}}^{{**}}$ | {ybar_star:.3f} | Mean of lot means |
                        | $n_E$ | {n_e:.3f} | Effective sample size |
                        | $S^2_L$ | {lot_var:.3f} | Lot-to-lot variance |
                        | $S^2_E$ | {error_var:.3f} | Within-lot variance |
                        | $U$ | {var_upper:.3f} | Variance upper limit |
                        | Type | {interval_type.replace('-', ' ').title()} | Auto-selected based on specs |
                        
                        **Settings:** Coverage = {p_coverage * 100:.1f}%, Confidence = {conf_level * 100:.0f}%
                        """)
            else:
                st.error(f"❌ {ti_result['message']}")
        
    
    # === RIGHT SIDE: METHODS DESCRIPTION ===
    with right_col:
        st.markdown("### 📋 Methods & Formulas")
        
        # Process Capability Methods
        with st.expander("📊 Process Capability & %OOS Methods", expanded=False):
            st.markdown("""
            Three complementary methods for calculating process capability indices and out-of-specification percentages.
            """)
            
            # Create a clean comparison table
            st.markdown("""
            | Method | Ppk Calculation | %OOS Calculation | Confidence Interval | Assumptions |
            |--------|-----------------|------------------|-------------------|-------------|
            | **Normal** | Sample μ, σ | Normal CDF | Bissell's approximation | Normal distribution |
            | **Empirical** | Sample percentiles | Direct count | Bootstrap (1000x) | None (non-parametric) |
            | **Fitted** | Distribution quantiles | Distribution CDF | Bootstrap (500x) | Best-fit distribution |
            """)
            
            st.markdown("---")
            
            # Detailed formulas
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.markdown("""
                **🔵 Normal Method**
                
                **Formula:**
                ```
                Ppk = min((USL-μ)/3σ, (μ-LSL)/3σ)
                %OOS = Φ((LSL-μ)/σ) + [1-Φ((USL-μ)/σ)]
                ```
                
                **Features:**
                - Classical industrial approach
                - Fast analytical calculation
                - Bissell's CI formula
                - Requires normality assumption
                """)
            
            with col2:
                st.markdown("""
                **🟢 Empirical Method**
                
                **Formula:**
                ```
                P₀.₁₃₅, P₅₀, P₉₉.₈₆₅ from data
                Ppk = min((USL-P₅₀)/(P₉₉.₈₆₅-P₅₀), 
                         (P₅₀-LSL)/(P₅₀-P₀.₁₃₅))
                %OOS = count(X ∉ [LSL,USL]) / n
                ```
                
                **Features:**
                - Distribution-free approach
                - Direct percentile calculation
                - Bootstrap confidence intervals
                - Robust to outliers
                """)
            
            with col3:
                st.markdown("""
                **🟡 Fitted Distribution**
                
                **Formula:**
                ```
                σₑq = (P₉₉.₈₆₅-P₀.₁₃₅)/6 from fitted dist
                Ppk = min((USL-P₅₀)/3σₑq, 
                         (P₅₀-LSL)/3σₑq)
                %OOS = F(LSL) + [1-F(USL)]
                ```
                
                **Features:**
                - Accounts for actual distribution shape
                - Uses Weibull, Gamma, etc.
                - Theoretical %OOS from CDF
                - Most accurate if fit is good
                """)
            
            st.markdown("---")
            st.info("💡 **%OOS Values**: Displayed in parentheses after each Ppk value for easy comparison")
        
        # Enhanced Tolerance Intervals Method (combined explanation)
        with st.expander("📐 Tolerance Intervals (NoSlope Method)", expanded=False):
            st.markdown(r"""
            **Random Effects Model (NoSlope Method)**
            
            Based on Burdick et al. method for multi-lot data without time trends.
            Implementation automatically selects interval type based on specifications.
            
            **Statistical Model**
            $$Y_{ij} = \mu + \alpha_i + \varepsilon_{ij}$$
            where $\alpha_i \sim N(0, \sigma^2_L)$ and $\varepsilon_{ij} \sim N(0, \sigma^2_E)$
            
            **Key Parameters**
            - $I$ = number of lots, $s = I-1$ = lot degrees of freedom
            - $J_i$ = replicates per lot $i$, $r = \sum J_i - I$ = error degrees of freedom
            - $J_H = I / \sum(1/J_i)$ = harmonic mean of replicates
            - $\bar{Y}^{**} = \frac{1}{I}\sum\bar{Y}_i$ = unweighted mean of lot means
            - $n_E = \hat{\text{Var}}(Y) / \hat{\text{Var}}(\bar{Y}^{**})$ = effective sample size
            
            **Variance Components**
            - $S^2_L = \text{Var}(\bar{Y}_i)$ = lot-to-lot variance (sample variance of lot means)
            - $S^2_E = \frac{\sum(J_i-1)s^2_i}{r}$ = pooled within-lot variance
            - $\hat{\text{Var}}(Y) = S^2_L + (1-1/J_H) \cdot S^2_E$ = total process variance
            - $\hat{\text{Var}}(\bar{Y}^{**}) = S^2_L / I$ = variance of grand mean
            
            **MLS Upper Limit for Variance**
            $$U = \hat{\text{Var}}(Y) + \sqrt{(H_1 S^2_L)^2 + (H_2(1-1/J_H)S^2_E)^2}$$
            where $H_1 = s/\chi^2_{\alpha,s} - 1$, $H_2 = r/\chi^2_{\alpha,r} - 1$
            
            **Tolerance Intervals (Automatic Selection)**
            
            **Two-Sided** (when both LSL & USL provided - Hoffman & Kringle 2005)
            $$TI = \bar{Y}^{**} \pm z_{(1+p)/2} \sqrt{1 + 1/n_E} \sqrt{U}$$
            
            **One-Sided** (when only LSL or USL provided - Hoffman 2010)
            $$TI = \bar{Y}^{**} \pm z_p \sqrt{U} \pm z_\gamma \sqrt{\hat{\text{Var}}(\bar{Y}^{**})}$$
            
            **Alternative One-Sided HK1** (also computed)
            $$TI = \bar{Y}^{**} \pm z_p \sqrt{1 + 1/n_E} \sqrt{U}$$
            
            **Coverage Scan**: Find maximum $p^*$ where $TI \subseteq [LSL, USL]$
            
            **Z-Score Definitions**
            - $z_p$ = normal quantile for coverage proportion $p$
            - $z_\gamma$ = normal quantile for confidence level $\gamma$  
            - $z_{(1+p)/2}$ = normal quantile for two-sided coverage
            """)
        
        # Statistical Notes
        with st.expander("⚠️ Statistical Considerations", expanded=False):
            st.markdown("""
            **Normality Assumption**
            - Tolerance intervals assume normal distribution
            - Capability indices robust to mild non-normality
            - Consider empirical methods for highly non-normal data
            
            **Sample Size Effects**  
            - Larger n → narrower confidence intervals
            - Minimum n=30 recommended for stable estimates
            - Bootstrap methods require n≥20
            
            **Interpretation Guidelines**
            - Ppk ≥ 1.33: Process capable
            - Ppk < 1.0: Process incapable  
            - Wide CI → need more data
            - Compare methods for robustness check
            """)

def render_export_tab():
    """Render the export functionality tab."""
    from services.export import ExportManager, PDF_AVAILABLE
    
    if 'value_col' not in st.session_state.specs or st.session_state.specs['value_col'] is None:
        st.warning("⬅️ Please upload data and complete analysis in other tabs first.")
        return
    
    st.markdown("### 💾 Export Analysis Results")
    st.caption("Download your analysis results in various formats")
    
    # Get current analysis data
    value_col = st.session_state.specs.get("value_col")
    data_series = st.session_state.data[value_col].dropna()
    lsl = st.session_state.specs.get("LSL")
    usl = st.session_state.specs.get("USL")
    
    # Check if we have capability analysis results
    has_capability_data = (lsl is not None or usl is not None)
    has_tolerance_data = hasattr(st.session_state, 'ti_results') and st.session_state.ti_results
    
    export_manager = ExportManager()
    
    # Create columns for different export options
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### 📄 Data Export")
        
        # Raw data export
        if st.button("📊 Export Raw Data (CSV)", use_container_width=True):
            csv_data = export_manager.export_raw_data(st.session_state.data)
            filename = export_manager.create_filename("raw_data", "csv")
            st.download_button(
                label="⬇️ Download CSV",
                data=csv_data,
                file_name=filename,
                mime="text/csv"
            )
        
        # Excel export with multiple sheets
        if st.button("📈 Export Analysis Summary (Excel)", use_container_width=True):
            # Prepare capability results if available
            capability_results = {}
            if has_capability_data:
                from services.capability import (
                    calculate_normal_ppk, 
                    calculate_fitted_distribution_ppk, 
                    calculate_empirical_ppk
                )
                
                capability_results["Normal"] = calculate_normal_ppk(data_series, lsl, usl)
                capability_results["Empirical"] = calculate_empirical_ppk(data_series, lsl, usl)
                
                distribution_name = st.session_state.selected_distribution
                distribution_params = st.session_state.distribution_params
                if distribution_name and distribution_params:
                    capability_results["Fitted"] = calculate_fitted_distribution_ppk(
                        data_series, distribution_name, distribution_params, lsl, usl
                    )
            
            # Create summary table
            summary_df = export_manager.create_analysis_summary(
                data_series, 
                capability_results,
                st.session_state.ti_results if has_tolerance_data else None,
                {"LSL": lsl, "USL": usl}
            )
            
            # Prepare data for Excel
            excel_data = {
                "Raw_Data": st.session_state.data,
                "Analysis_Summary": summary_df
            }
            
            excel_bytes = export_manager.export_data_to_excel(excel_data)
            filename = export_manager.create_filename("analysis_summary", "xlsx")
            st.download_button(
                label="⬇️ Download Excel Report",
                data=excel_bytes,
                file_name=filename,
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )
    
    with col2:
        st.markdown("#### 📋 Reports & Charts")
        
        # PDF report (if available)
        if PDF_AVAILABLE and has_capability_data:
            if st.button("📄 Generate PDF Report", use_container_width=True):
                try:
                    # Prepare capability results
                    capability_results = {}
                    from services.capability import (
                        calculate_normal_ppk, 
                        calculate_fitted_distribution_ppk, 
                        calculate_empirical_ppk
                    )
                    
                    capability_results["Normal"] = calculate_normal_ppk(data_series, lsl, usl)
                    capability_results["Empirical"] = calculate_empirical_ppk(data_series, lsl, usl)
                    
                    distribution_name = st.session_state.selected_distribution
                    distribution_params = st.session_state.distribution_params
                    if distribution_name and distribution_params:
                        capability_results["Fitted"] = calculate_fitted_distribution_ppk(
                            data_series, distribution_name, distribution_params, lsl, usl
                        )
                    
                    pdf_bytes = export_manager.generate_pdf_report(
                        data_series,
                        capability_results,
                        st.session_state.ti_results if has_tolerance_data else None,
                        {"LSL": lsl, "USL": usl}
                    )
                    
                    filename = export_manager.create_filename("capability_report", "pdf")
                    st.download_button(
                        label="⬇️ Download PDF Report",
                        data=pdf_bytes,
                        file_name=filename,
                        mime="application/pdf"
                    )
                except Exception as e:
                    st.error(f"PDF generation failed: {str(e)}")
        
        elif has_capability_data:
            st.info("📄 PDF reports require `reportlab` package. Install with: `pip install reportlab`")
        
        else:
            st.info("📊 Complete capability analysis first to generate reports")
        
        # Chart export
        st.markdown("**Export Charts**")
        chart_format = st.selectbox("Chart Format", ["PNG", "SVG", "HTML"], key="chart_format")
        
        if st.button(f"📸 Export Charts ({chart_format})", use_container_width=True):
            st.info("🚧 Chart export functionality - coming soon!")
    
    # Show export status/info
    st.markdown("---")
    
    with st.expander("ℹ️ Export Information", expanded=False):
        st.markdown("""
        **Available Export Options:**
        
        📊 **Raw Data (CSV)**: Your original uploaded data  
        📈 **Analysis Summary (Excel)**: Multi-sheet workbook with raw data and statistical summary  
        📄 **PDF Report**: Comprehensive analysis report (requires reportlab package)  
        📸 **Charts**: Export visualizations in various formats
        
        **File Naming Convention:**  
        All exported files are automatically named with timestamps to avoid conflicts.
        
        **Privacy Note:**  
        All exports are generated locally in your browser. No data is sent to external servers.
        """)

def render_control_charts_tab():
    """Render the new categorical control chart tab."""
    
    if 'value_col' not in st.session_state.specs or st.session_state.specs['value_col'] is None:
        st.warning("⬅️ Please select a column in the sidebar first.")
        return

    from services.control_charts import create_categorical_control_chart

    df = st.session_state.data
    value_col = st.session_state.specs['value_col']
    lsl = st.session_state.specs.get("LSL")
    usl = st.session_state.specs.get("USL")

    st.markdown("#### Chart Settings")
    
    categorical_cols = [None] + df.select_dtypes(include=['object', 'category']).columns.tolist()
    # Get all columns for sorting (including numeric and datetime)
    all_sortable_cols = [None] + [col for col in df.columns if col != st.session_state.specs['value_col']]

    col1, col2, col3 = st.columns(3)
    with col1:
        primary_cat = st.selectbox("Primary Category (X-Axis)", categorical_cols, help="The main category to plot on the X-axis.")
    with col2:
        grouping_cat = st.selectbox("Group By (Optional)", categorical_cols, help="An optional higher-level category.")
    with col3:
        sort_col = st.selectbox("Sort X-Axis By", all_sortable_cols, help="Sort categories by this column (ascending order).")

    if primary_cat is None and grouping_cat is not None:
        category_col = grouping_cat
        grouping_col = None
    else:
        category_col = primary_cat
        grouping_col = grouping_cat

    if category_col is None:
        st.info("Please select at least one category for the X-axis.")
        return

    with st.spinner("Creating control chart..."):
        fig = create_categorical_control_chart(
            data=df,
            value_col=value_col,
            category_col=category_col,
            grouping_col=grouping_col,
            sort_col=sort_col,
            lsl=lsl,
            usl=usl
        )
        st.plotly_chart(fig, use_container_width=True)

def main():
    initialize_session_state()
    render_sidebar()
    
    st.markdown('''
    <div class="apple-header">
        <h1>Process Capability Analysis Platform</h1>
        <p>Advanced statistical process control with tolerance intervals and capability assessment</p>
        <div class="header-meta">CMC Quantitative Sciences • Aug 2025</div>
    </div>
    ''', unsafe_allow_html=True)
    
    tabs = st.tabs([
        "📊 Overview", "⚡ Capability & Tolerance", 
        "📈 Control Charts", "💾 Export"
    ])
    
    with tabs[0]:
        render_overview_tab()
    
    with tabs[1]:
        render_capability_tab()
        
    with tabs[2]:
        render_control_charts_tab()
        
    with tabs[3]:
        render_export_tab()

if __name__ == "__main__":
    main()