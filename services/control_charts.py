import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from typing import Dict, Any, Optional, List, Tuple
from scipy import stats

def create_categorical_control_chart(
    data: pd.DataFrame,
    value_col: str,
    category_col: str,
    grouping_col: Optional[str] = None,
    sort_col: Optional[str] = None,
    lsl: Optional[float] = None,
    usl: Optional[float] = None
) -> go.Figure:
    """
    Create a categorical control chart that plots the average of each category.

    Args:
        data: Input DataFrame.
        value_col: The column with the measurement values.
        category_col: The primary categorical column for the X-axis.
        grouping_col: Optional secondary column for grouping.
        sort_col: Optional column to sort categories by (ascending).
        lsl: Lower Specification Limit.
        usl: Upper Specification Limit.

    Returns:
        A Plotly figure object.
    """
    if value_col not in data.columns or category_col not in data.columns:
        return go.Figure().add_annotation(text="Value or category column not found in data.")

    df = data.copy()

    # --- Data Aggregation ---
    group_by_cols = [category_col]
    if grouping_col and grouping_col in df.columns:
        group_by_cols.insert(0, grouping_col)

    # Include sort column in aggregation if specified
    aggregations = {value_col: 'mean'}
    if sort_col and sort_col in df.columns:
        # Use 'first' to get the first value for sorting (could also use min/max)
        aggregations[sort_col] = 'first'

    agg_df = df.groupby(group_by_cols, as_index=False).agg(aggregations)
    agg_df = agg_df.rename(columns={value_col: 'y_mean'})
    
    # Sort by the specified column if provided
    if sort_col and sort_col in agg_df.columns:
        agg_df = agg_df.sort_values(by=sort_col).reset_index(drop=True)
        print(f"DEBUG: Sorted by {sort_col}")
    
    # --- Control Limit Calculation ---
    original_values = df[value_col].dropna()
    mean = original_values.mean()
    std_dev = original_values.std()

    ucl_3s = mean + 3 * std_dev
    lcl_3s = mean - 3 * std_dev

    fig = go.Figure()

    # JMP-style colors
    jmp_colors = {
        'data': '#4472C4',       # Professional blue for data points
        'mean': '#27AE60',       # Green for mean line
        'control': '#E74C3C',    # Red for control limits
        'spec': '#8E44AD',       # Purple for specification limits
        'outlier': '#E74C3C'     # Red for outliers
    }

    # --- Plotting ---
    # Create x-axis labels
    if grouping_col and grouping_col in agg_df.columns:
        # Create combined labels like "Group1 | Category1"
        x_axis = agg_df[grouping_col].astype(str) + " | " + agg_df[category_col].astype(str)
    else:
        x_axis = agg_df[category_col].astype(str)
    
    # Store the order for explicit ordering
    x_axis_order = x_axis.tolist()

    # Main data trace with JMP styling
    fig.add_trace(go.Scatter(
        x=x_axis,
        y=agg_df['y_mean'],
        mode='lines+markers',
        name='Batch Average',
        line=dict(color=jmp_colors['data'], width=2),
        marker=dict(size=8, color=jmp_colors['data'], 
                   line=dict(width=1, color='white'))
    ))

    # Add Control Limit lines with JMP styling
    fig.add_hline(y=mean, line_dash="solid", line_color=jmp_colors['mean'], 
                  line_width=2, annotation_text=f"Mean ({mean:.3f})")
    fig.add_hline(y=ucl_3s, line_dash="dash", line_color=jmp_colors['control'], 
                  line_width=2, annotation_text=f"+3σ ({ucl_3s:.3f})")
    fig.add_hline(y=lcl_3s, line_dash="dash", line_color=jmp_colors['control'], 
                  line_width=2, annotation_text=f"-3σ ({lcl_3s:.3f})")

    # Add Specification Limit lines with JMP styling
    if usl is not None:
        fig.add_hline(y=usl, line_dash="solid", line_color=jmp_colors['spec'], 
                      line_width=2, annotation_text=f"USL ({usl:.3f})")
    if lsl is not None:
        fig.add_hline(y=lsl, line_dash="solid", line_color=jmp_colors['spec'], 
                      line_width=2, annotation_text=f"LSL ({lsl:.3f})")

    # Highlight outliers with JMP styling
    outliers = agg_df[(agg_df['y_mean'] > ucl_3s) | (agg_df['y_mean'] < lcl_3s)]
    if not outliers.empty:
        if grouping_col and grouping_col in outliers.columns:
            outlier_x = outliers[grouping_col].astype(str) + " | " + outliers[category_col].astype(str)
        else:
            outlier_x = outliers[category_col].astype(str)
        
        fig.add_trace(go.Scatter(
            x=outlier_x,
            y=outliers['y_mean'],
            mode='markers',
            name='Outside 3σ',
            marker=dict(color=jmp_colors['outlier'], size=12, symbol='x', 
                       line=dict(width=2, color='white'))
        ))

    # Apply JMP styling
    fig.update_layout(
        xaxis_title="Categories",
        yaxis_title="Average Value",
        legend=dict(
            orientation="h", 
            yanchor="bottom", 
            y=1.02, 
            xanchor="right", 
            x=1,
            bgcolor='rgba(255,255,255,0.8)',
            bordercolor='#CCCCCC',
            borderwidth=1
        )
    )
    
    # Force x-axis to respect the sorted order
    fig.update_xaxes(
        categoryorder='array',
        categoryarray=x_axis_order
    )
    
    # Import and apply JMP style function
    from services.visualizations import apply_jmp_style
    fig = apply_jmp_style(fig, "Categorical Control Chart", height=450)

    return fig

def calculate_imr_limits(data: pd.Series, subgroup_size: int = 1) -> Dict[str, Any]:
    """
    Calculate I-MR (Individual and Moving Range) control limits.
    """
    clean_data = data.dropna()
    n = len(clean_data)
    
    if n < 2:
        return {"success": False, "message": "Insufficient data for I-MR chart (n < 2)", "n": n}
    
    x_values = clean_data.values
    x_bar = np.mean(x_values)
    mr_values = np.abs(np.diff(x_values))
    mr_bar = np.mean(mr_values)
    
    d2 = 1.128
    D4 = 3.267
    
    ucl_x = x_bar + 3 * (mr_bar / d2)
    lcl_x = x_bar - 3 * (mr_bar / d2)
    
    return {
        "success": True,
        "individual_chart": {"ucl": ucl_x, "cl": x_bar, "lcl": lcl_x},
        "mr_chart": {"ucl": D4 * mr_bar, "cl": mr_bar, "lcl": 0}
    }

def create_imr_chart(
    data: pd.Series,
    title: str = "I-MR Control Chart",
    show_violations: bool = True,
    levey_jennings: bool = False
) -> Dict[str, Any]:
    """
    Create I-MR control chart using Plotly.
    """
    # This function is now legacy but kept for potential future use.
    # The main app uses create_categorical_control_chart.
    return {"success": False, "message": "This chart type is deprecated."}