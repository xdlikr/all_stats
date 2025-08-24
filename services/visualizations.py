import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from typing import Dict, Any, Optional, List, Tuple
from scipy import stats
from services.distributions import calculate_quantile, calculate_cdf

def create_histogram_with_fit(
    data: pd.Series,
    distribution_name: str,
    distribution_params: Dict[str, float],
    bins: int = 30,
    title: str = "Histogram with Distribution Fit"
) -> go.Figure:
    """
    Create histogram with overlaid distribution fit.
    
    Args:
        data: Input data series
        distribution_name: Name of fitted distribution
        distribution_params: Parameters of fitted distribution
        bins: Number of histogram bins
        title: Plot title
        
    Returns:
        Plotly figure
    """
    clean_data = data.dropna()
    
    # Create histogram
    fig = go.Figure()
    
    # Add histogram
    fig.add_trace(go.Histogram(
        x=clean_data,
        nbinsx=bins,
        opacity=0.7,
        name='Data',
        histnorm='probability density',
        marker_color='lightblue',
        marker_line_color='darkblue',
        marker_line_width=1
    ))
    
    # Generate fitted curve
    x_min, x_max = clean_data.min(), clean_data.max()
    x_range = x_max - x_min
    x_fit = np.linspace(x_min - 0.1*x_range, x_max + 0.1*x_range, 200)
    
    # Calculate PDF values for fitted distribution
    if distribution_name == "normal":
        y_fit = stats.norm.pdf(x_fit, loc=distribution_params["loc"], scale=distribution_params["scale"])
    elif distribution_name == "lognormal":
        y_fit = stats.lognorm.pdf(x_fit, s=distribution_params["sigma"], scale=np.exp(distribution_params["mu"]))
    elif distribution_name == "weibull":
        y_fit = stats.weibull_min.pdf(x_fit, c=distribution_params["c"], scale=distribution_params["scale"])
    elif distribution_name == "gamma":
        y_fit = stats.gamma.pdf(x_fit, a=distribution_params["a"], scale=distribution_params["scale"])
    elif distribution_name == "logistic":
        y_fit = stats.logistic.pdf(x_fit, loc=distribution_params["loc"], scale=distribution_params["scale"])
    elif distribution_name == "exponential":
        y_fit = stats.expon.pdf(x_fit, scale=distribution_params["scale"])
    else:
        y_fit = np.zeros_like(x_fit)  # Fallback
    
    # Add fitted curve
    fig.add_trace(go.Scatter(
        x=x_fit,
        y=y_fit,
        mode='lines',
        name=f'{distribution_name.capitalize()} Fit',
        line=dict(color='red', width=2)
    ))
    
    fig.update_layout(
        title=dict(text=title, font=dict(size=14)),
        xaxis_title='Value',
        yaxis_title='Probability Density',
        template='plotly_white',
        legend=dict(x=0.7, y=0.9),
        font=dict(size=10),
        margin=dict(l=50, r=50, t=50, b=50)
    )
    
    return fig

def create_qq_plot(
    data: pd.Series,
    distribution_name: str,
    distribution_params: Dict[str, float],
    title: str = "Q-Q Plot"
) -> go.Figure:
    """
    Create Q-Q plot for distribution fit assessment.
    
    Args:
        data: Input data series
        distribution_name: Name of fitted distribution
        distribution_params: Parameters of fitted distribution
        title: Plot title
        
    Returns:
        Plotly figure
    """
    clean_data = data.dropna().sort_values()
    n = len(clean_data)
    
    # Calculate theoretical quantiles
    p_values = (np.arange(1, n + 1) - 0.5) / n
    
    theoretical_quantiles = []
    for p in p_values:
        try:
            q = calculate_quantile(distribution_name, distribution_params, p)
            theoretical_quantiles.append(q)
        except:
            theoretical_quantiles.append(np.nan)
    
    theoretical_quantiles = np.array(theoretical_quantiles)
    
    # Create Q-Q plot
    fig = go.Figure()
    
    # Add scatter points
    fig.add_trace(go.Scatter(
        x=theoretical_quantiles,
        y=clean_data.values,
        mode='markers',
        name='Data Points',
        marker=dict(color='blue', size=6, opacity=0.7)
    ))
    
    # Add reference line (perfect fit)
    min_val = min(np.nanmin(theoretical_quantiles), clean_data.min())
    max_val = max(np.nanmax(theoretical_quantiles), clean_data.max())
    
    fig.add_trace(go.Scatter(
        x=[min_val, max_val],
        y=[min_val, max_val],
        mode='lines',
        name='Perfect Fit',
        line=dict(color='red', width=2, dash='dash')
    ))
    
    # Calculate correlation coefficient for goodness of fit
    valid_mask = ~np.isnan(theoretical_quantiles)
    if np.sum(valid_mask) > 1:
        correlation = np.corrcoef(theoretical_quantiles[valid_mask], clean_data.values[valid_mask])[0, 1]
        title += f" (R² = {correlation**2:.4f})"
    
    fig.update_layout(
        title=dict(text=title, font=dict(size=14)),
        xaxis_title=f'Theoretical Quantiles ({distribution_name.capitalize()})',
        yaxis_title='Sample Quantiles',
        template='plotly_white',
        showlegend=True,
        font=dict(size=10),
        margin=dict(l=50, r=50, t=50, b=50)
    )
    
    return fig

def create_distribution_comparison_grid(
    data: pd.Series,
    comparison_results: Dict[str, Any],
    max_distributions: int = 6
) -> go.Figure:
    """
    Create a grid of small distribution comparison plots.
    
    Args:
        data: Input data series
        comparison_results: Results from compare_distributions
        max_distributions: Maximum number of distributions to show
        
    Returns:
        Plotly figure with subplots
    """
    comparison_table = comparison_results["comparison_table"]
    n_dists = min(len(comparison_table), max_distributions)
    
    if n_dists == 0:
        return go.Figure().add_annotation(text="No distributions fitted successfully")
    
    # Create subplots (1 row for each distribution: histogram+fit only)
    cols = min(n_dists, 4)  # Max 4 columns for more compact layout
    rows = (n_dists + cols - 1) // cols  # Ceiling division for number of rows needed
    
    subplot_titles = []
    for i in range(n_dists):
        dist_info = comparison_table[i]
        dist_name = dist_info["Distribution"]
        aic = dist_info["AIC"]
        aic_str = f"AIC: {aic:.2f}" if aic is not None else "AIC: N/A"
        subplot_titles.append(f"{dist_name.capitalize()} ({aic_str})")
    
    fig = make_subplots(
        rows=rows,
        cols=cols,
        subplot_titles=subplot_titles,
        vertical_spacing=0.1,  # Adjusted spacing for single row layout
        horizontal_spacing=0.05  # Adjusted spacing
    )
    
    clean_data = data.dropna()
    
    for i, dist_info in enumerate(comparison_table[:n_dists]):
        dist_name = dist_info["Distribution"]
        dist_params = dist_info["Parameters"]
        
        row = (i // cols) + 1
        col = (i % cols) + 1
        
        # Add histogram
        fig.add_trace(
            go.Histogram(
                x=clean_data,
                nbinsx=15,  # Fewer bins for cleaner look
                opacity=0.7,
                histnorm='probability density',
                marker_color='lightblue',
                marker_line_color='darkblue',
                marker_line_width=1,
                showlegend=False,
                name=f'{dist_name} hist'
            ),
            row=row, col=col
        )
        
        # Add fitted curve
        x_min, x_max = clean_data.min(), clean_data.max()
        x_range = x_max - x_min
        x_fit = np.linspace(x_min - 0.1*x_range, x_max + 0.1*x_range, 100)
        
        try:
            if dist_name == "normal":
                y_fit = stats.norm.pdf(x_fit, loc=dist_params["loc"], scale=dist_params["scale"])
            elif dist_name == "lognormal":
                y_fit = stats.lognorm.pdf(x_fit, s=dist_params["sigma"], scale=np.exp(dist_params["mu"]))
            elif dist_name == "weibull":
                y_fit = stats.weibull_min.pdf(x_fit, c=dist_params["c"], scale=dist_params["scale"])
            elif dist_name == "gamma":
                y_fit = stats.gamma.pdf(x_fit, a=dist_params["a"], scale=dist_params["scale"])
            elif dist_name == "logistic":
                y_fit = stats.logistic.pdf(x_fit, loc=dist_params["loc"], scale=dist_params["scale"])
            elif dist_name == "exponential":
                y_fit = stats.expon.pdf(x_fit, scale=dist_params["scale"])
            else:
                y_fit = np.zeros_like(x_fit)
            
            fig.add_trace(
                go.Scatter(
                    x=x_fit,
                    y=y_fit,
                    mode='lines',
                    line=dict(color='red', width=2),
                    showlegend=False,
                    name=f'{dist_name} fit'
                ),
                row=row, col=col
            )
        except:
            pass  # Skip if curve generation fails
    
    fig.update_layout(
        height=300 * rows,  # Single row of histograms
        title=dict(text="Distribution Comparison Grid", font=dict(size=14)),
        template='plotly_white',
        showlegend=False,
        font=dict(size=10),  # Smaller font for compact view
        margin=dict(l=40, r=40, t=60, b=40)
    )
    
    return fig

def create_basic_statistics_plot(data: pd.Series, title: str = "Data Distribution") -> go.Figure:
    """
    Create basic statistics visualization (histogram + box plot).
    
    Args:
        data: Input data series
        title: Plot title
        
    Returns:
        Plotly figure
    """
    clean_data = data.dropna()
    
    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.05,
        row_heights=[0.8, 0.2]
    )
    
    # Histogram
    fig.add_trace(
        go.Histogram(
            x=clean_data,
            nbinsx=30,
            opacity=0.7,
            marker_color='lightblue',
            marker_line_color='darkblue',
            marker_line_width=1,
            name='Distribution'
        ),
        row=1, col=1
    )
    
    # Box plot
    fig.add_trace(
        go.Box(
            x=clean_data,
            name='Data',
            marker_color='lightgreen',
            boxmean='sd'  # Show mean and standard deviation
        ),
        row=2, col=1
    )
    
    fig.update_layout(
        title=dict(text=title, font=dict(size=14)),
        template='plotly_white',
        showlegend=False,
        font=dict(size=10),
        margin=dict(l=50, r=50, t=50, b=50)
    )
    
    fig.update_yaxes(title_text="Frequency", row=1, col=1)
    
    return fig

def create_capability_summary_plot(
    capability_results: Dict[str, Any],
    specs: Dict[str, Any],
    title: str = "Process Capability Summary"
) -> go.Figure:
    """
    Create capability indices summary visualization.
    
    Args:
        capability_results: Results from capability calculation
        specs: Specification limits
        title: Plot title
        
    Returns:
        Plotly figure
    """
    # Extract indices
    indices = {}
    ci_lower = {}
    ci_upper = {}
    
    for metric in ['Pp', 'Ppk']:
        if metric in capability_results and capability_results[metric] != float('inf'):
            indices[metric] = capability_results[metric]
            # Check for confidence intervals
            ci_lower[metric] = capability_results.get(f"{metric}_ci_lower", None)
            ci_upper[metric] = capability_results.get(f"{metric}_ci_upper", None)
    
    if not indices:
        return go.Figure().add_annotation(text="No capability indices available")
    
    # Create bar chart
    fig = go.Figure()
    
    metrics = list(indices.keys())
    values = list(indices.values())
    
    # Add bars
    fig.add_trace(go.Bar(
        x=metrics,
        y=values,
        marker_color=['lightblue', 'darkblue', 'lightgreen', 'darkgreen'],
        name='Capability Index'
    ))
    
    # Add error bars if confidence intervals are available
    error_y_values = []
    for metric in metrics:
        if ci_lower.get(metric) is not None and ci_upper.get(metric) is not None:
            error_y_values.append(ci_upper[metric] - indices[metric])
        else:
            error_y_values.append(0)
    
    if any(e > 0 for e in error_y_values):
        fig.data[0].error_y = dict(
            type='data',
            array=error_y_values,
            visible=True
        )
    
    # Add capability threshold lines
    fig.add_hline(y=1.0, line_dash="dash", line_color="orange", 
                  annotation_text="Minimum Acceptable (1.0)")
    fig.add_hline(y=1.33, line_dash="dash", line_color="green", 
                  annotation_text="Good Capability (1.33)")
    
    fig.update_layout(
        title=dict(text=title, font=dict(size=14)),
        xaxis_title="Capability Index",
        yaxis_title="Value",
        template='plotly_white',
        yaxis=dict(range=[0, max(2.0, max(values) * 1.1)]),
        font=dict(size=10),
        margin=dict(l=50, r=50, t=50, b=50)
    )
    
    return fig

def create_oos_comparison_plot(
    oos_results: List[Dict[str, Any]],
    title: str = "%OOS Method Comparison"
) -> go.Figure:
    """
    Create %OOS method comparison visualization.
    
    Args:
        oos_results: List of OOS calculation results
        title: Plot title
        
    Returns:
        Plotly figure
    """
    methods = []
    oos_values = []
    colors = []
    
    method_names = {
        "fitted": "Fitted Distribution",
        "empirical": "Empirical Data", 
        "ti_method3": "Tolerance Interval",
        "alt_fitted": "Alternative Distribution"
    }
    
    method_colors = {
        "fitted": "blue",
        "empirical": "green",
        "ti_method3": "orange",
        "alt_fitted": "purple"
    }
    
    for result in oos_results:
        if result.get("Success", False):
            method = result["Method"]
            oos_val = result["%OOS"]
            
            methods.append(method_names.get(method, method))
            oos_values.append(oos_val)
            colors.append(method_colors.get(method, "gray"))
    
    if not methods:
        return go.Figure().add_annotation(text="No %OOS results available")
    
    fig = go.Figure(data=[
        go.Bar(
            x=methods,
            y=oos_values,
            marker_color=colors,
            text=[f"{val:.4f}%" for val in oos_values],
            textposition='auto'
        )
    ])
    
    fig.update_layout(
        title=title,
        xaxis_title="Calculation Method",
        yaxis_title="%OOS",
        template='plotly_white'
    )
    
    return fig

def create_coverage_scan_plot(
    scan_results: Dict[str, Any],
    specs: Dict[str, Any],
    title: str = "Tolerance Interval Coverage Scan"
) -> go.Figure:
    """
    Create coverage scan visualization.
    
    Args:
        scan_results: Results from coverage_scan
        specs: Specification limits
        title: Plot title
        
    Returns:
        Plotly figure
    """
    if not scan_results.get("success") or not scan_results.get("results"):
        return go.Figure().add_annotation(text="No coverage scan data available")
    
    df = pd.DataFrame(scan_results["results"])
    lsl = specs.get("LSL")
    usl = specs.get("USL")
    
    fig = go.Figure()
    
    # Plot tolerance interval bounds
    if "lower_bound" in df.columns and "upper_bound" in df.columns:
        # Two-sided
        fig.add_trace(go.Scatter(x=df["coverage_%"], y=df["lower_bound"], mode='lines', name='Lower Tolerance Bound', line=dict(color='blue')))
        fig.add_trace(go.Scatter(x=df["coverage_%"], y=df["upper_bound"], mode='lines', name='Upper Tolerance Bound', line=dict(color='red')))
        if lsl is not None:
            fig.add_hline(y=lsl, line_dash="dash", line_color="green", annotation_text="LSL")
        if usl is not None:
            fig.add_hline(y=usl, line_dash="dash", line_color="orange", annotation_text="USL")
        fig.update_layout(yaxis_title="Tolerance Interval Bounds")

    elif "upper_bound" in df.columns:
        # Upper one-sided
        fig.add_trace(go.Scatter(x=df["coverage_%"], y=df["upper_bound"], mode='lines', name='Upper Tolerance Bound', line=dict(color='red')))
        if usl is not None:
            fig.add_hline(y=usl, line_dash="dash", line_color="orange", annotation_text="USL")
        fig.update_layout(yaxis_title="Upper Tolerance Bound")

    elif "lower_bound" in df.columns:
        # Lower one-sided
        fig.add_trace(go.Scatter(x=df["coverage_%"], y=df["lower_bound"], mode='lines', name='Lower Tolerance Bound', line=dict(color='blue')))
        if lsl is not None:
            fig.add_hline(y=lsl, line_dash="dash", line_color="green", annotation_text="LSL")
        fig.update_layout(yaxis_title="Lower Tolerance Bound")

    # Highlight optimal coverage point
    p_star = scan_results.get("p_star", 0)
    if p_star > 0:
        p_star_percent = p_star * 100
        fig.add_vline(x=p_star_percent, line_dash="dot", line_color="purple",
                      annotation_text=f"Optimal Coverage: {p_star_percent:.2f}%")
    
    fig.update_layout(
        title=title,
        xaxis_title="Coverage (%)",
        template='plotly_white'
    )
    
    return fig

def create_merged_distribution_plot(
    data: pd.Series,
    comparison_results: Dict[str, Any],
    title: str = "Distribution Fits Comparison"
) -> go.Figure:
    """
    Create a single plot with a histogram and multiple overlaid distribution fits.
    
    Args:
        data: Input data series
        comparison_results: Results from compare_distributions
        title: Plot title
        
    Returns:
        Plotly figure
    """
    clean_data = data.dropna()
    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.05,
        row_heights=[0.8, 0.2]
    )

    # 1. Add histogram of the actual data to the top subplot
    fig.add_trace(go.Histogram(
        x=clean_data,
        nbinsx=30,
        opacity=0.7,
        name='Data',
        histnorm='probability density',
        marker_color='lightblue'
    ), row=1, col=1)

    # 2. Iterate through fitted distributions and add their curves to the top subplot
    comparison_table = comparison_results.get("comparison_table", [])
    colors = px.colors.qualitative.Plotly

    x_min, x_max = clean_data.min(), clean_data.max()
    x_range = x_max - x_min
    x_fit = np.linspace(x_min - 0.1 * x_range, x_max + 0.1 * x_range, 200)

    for i, dist_info in enumerate(comparison_table):
        dist_name = dist_info["Distribution"]
        dist_params = dist_info["Parameters"]
        color = colors[i % len(colors)]

        y_fit = np.zeros_like(x_fit)
        try:
            if dist_name == "normal":
                y_fit = stats.norm.pdf(x_fit, **dist_params)
            elif dist_name == "lognormal":
                y_fit = stats.lognorm.pdf(x_fit, s=dist_params["sigma"], scale=np.exp(dist_params["mu"]))
            elif dist_name == "weibull":
                y_fit = stats.weibull_min.pdf(x_fit, c=dist_params["c"], scale=dist_params["scale"])
            elif dist_name == "gamma":
                y_fit = stats.gamma.pdf(x_fit, a=dist_params["a"], scale=dist_params["scale"])
            elif dist_name == "logistic":
                y_fit = stats.logistic.pdf(x_fit, **dist_params)
            
            if np.any(y_fit):
                fig.add_trace(go.Scatter(
                    x=x_fit,
                    y=y_fit,
                    mode='lines',
                    name=f'{dist_name.capitalize()} Fit',
                    line=dict(color=color, width=2)
                ), row=1, col=1)
        except Exception:
            continue

    # 3. Add the box plot to the bottom subplot
    fig.add_trace(go.Box(
        x=clean_data,
        name='Box Plot',
        marker_color='lightgreen',
        boxmean='sd'
    ), row=2, col=1)

    fig.update_layout(
        title=dict(text=title, font=dict(size=16)),
        yaxis_title='Probability Density',
        template='plotly_white',
        legend_title_text='Distribution',
        font=dict(size=12),
        margin=dict(l=50, r=50, t=50, b=50),
        height=500
    )
    
    return fig
