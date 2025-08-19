import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from functools import lru_cache

@st.cache_data(show_spinner=False)
def load_csv(file):
    """
    Load and validate a time series CSV file.
    
    Args:
        file: Uploaded CSV file with 'Date' column and numeric columns.
    
    Returns:
        pandas.DataFrame with 'Date' as index, or None if invalid.
    """
    try:
        df = pd.read_csv(file, index_col=0, parse_dates=True)
        if df.empty:
            raise ValueError("CSV file is empty")
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if not numeric_cols.size:
            raise ValueError("No numeric columns found to plot")
        # Handle missing values by interpolating
        df[numeric_cols] = df[numeric_cols].interpolate(method='time', limit_direction='both').bfill().ffill()
        return df
    except Exception as e:
        st.error(f"Error loading CSV: {str(e)}")
        return None

@lru_cache(maxsize=32)
def preprocess_series(df_values: tuple, rolling_window: int, normalize: bool, start_zero: bool):
    """
    Preprocess a time series with rolling average, normalization, or zero-start.
    
    Args:
        df_values: Tuple of series values.
        rolling_window: Number of days for rolling average (0 for none).
        normalize: If True, scale to [0,1].
        start_zero: If True, subtract first value to start at 0.
    
    Returns:
        numpy.ndarray of processed values.
    """
    arr = pd.Series(df_values).copy()
    if rolling_window > 0:
        arr = arr.rolling(window=rolling_window, min_periods=1).mean()
    if normalize:
        arr_min, arr_max = arr.min(), arr.max()
        if arr_max != arr_min:  # Avoid division by zero
            arr = (arr - arr_min) / (arr_max - arr_min)
        else:
            arr = arr * 0  # If all values are same, scale to 0
    if start_zero:
        arr = arr - arr.iloc[0] if not pd.isna(arr.iloc[0]) else arr
    return arr.values

def make_interactive_plot(
    df: pd.DataFrame,
    series_meta: dict,  # {col: {"color":..., "style":...}, ...}
    rolling_window: int,
    normalize: bool,
    start_zero: bool,
    recessions: list,  # [{"name":..., "start":..., "end":...}, ...]
    show_labels: bool,
    title: str,
    xlabel: str,
    ylabel: str,
    theme: str,
    width: int = 1200,  # Default high-resolution width
    height: int = 600,  # Default high-resolution height
    xaxis_range: list = None,  # [start_date, end_date]
    yaxis_range: list = None   # [min_value, max_value]
) -> go.Figure:
    """
    Create an interactive Plotly figure with time series and recession bands.
    
    Args:
        df: DataFrame with time series data.
        series_meta: Dict mapping column names to color and style.
        rolling_window: Days for rolling average.
        normalize: Scale series to [0,1].
        start_zero: Subtract first value to start at 0.
        recessions: List of recession periods.
        show_labels: Show recession names above bands.
        title: Plot title.
        xlabel: X-axis label.
        ylabel: Y-axis label.
        theme: 'Light' or 'Dark'.
        width: Figure width in pixels.
        height: Figure height in pixels.
        xaxis_range: List of [start_date, end_date] for x-axis.
        yaxis_range: List of [min_value, max_value] for y-axis.
    
    Returns:
        plotly.graph_objects.Figure
    """
    # Initialize figure
    fig = go.Figure()
    global_min, global_max = np.inf, -np.inf
    
    # Use scattergl for large datasets
    trace_type = go.Scattergl if len(df) > 10000 else go.Scatter
    
    # Add traces for each series
    for col, meta in series_meta.items():
        y = preprocess_series(
            tuple(df[col].values),
            rolling_window,
            normalize,
            start_zero
        )
        x = df.index
        if np.all(np.isnan(y)):
            st.warning(f"Series '{col}' contains only NaN values after preprocessing.")
            continue
        global_min = min(global_min, np.nanmin(y))
        global_max = max(global_max, np.nanmax(y))
        fig.add_trace(trace_type(
            x=x,
            y=y,
            name=col,
            line=dict(color=meta["color"], dash=meta["style"]),
            mode='lines'
        ))
    
    # Add recession bands
    for rec in recessions:
        start = pd.to_datetime(rec["start"], format='%Y/%m/%d')
        end = pd.to_datetime(rec["end"], format='%Y/%m/%d')
        if start < df.index.max() and end > df.index.min():
            fig.add_shape(
                type="rect",
                xref="x",
                yref="y",
                x0=start,
                x1=end,
                y0=yaxis_range[0] if yaxis_range else global_min,
                y1=yaxis_range[1] if yaxis_range else global_max,
                fillcolor="gray",
                opacity=0.2,
                layer="below",
                line_width=0
            )
            if show_labels:
                fig.add_annotation(
                    x=start + (end - start) / 2,
                    y=global_max,
                    text=rec["name"],
                    showarrow=False,
                    yanchor="bottom",
                    font=dict(family="Times New Roman", size=10),
                    xref="x",
                    yref="y"
                )
    
    # Update layout with high-resolution settings and bottom legend
    fig.update_layout(
        title=title,
        xaxis_title=xlabel,
        yaxis_title=ylabel,
        template="simple_white" if theme == "Light" else "plotly_dark",
        margin=dict(t=150 if show_labels else 100, b=100),  # Increased bottom margin for legend
        showlegend=True,
        hovermode="x unified",
        width=width,
        height=height,
        font=dict(
            family="Times New Roman",
            size=12
        ),
        legend=dict(
            orientation="h",  # Horizontal legend
            yanchor="top",
            y=-0.2,  # Place below the plot
            xanchor="center",
            x=0.5,   # Center horizontally
            font=dict(family="Times New Roman", size=12),
            bgcolor="rgba(255,255,255,0.5)" if theme == "Light" else "rgba(0,0,0,0.5)"
        ),
        xaxis=dict(
            showline=True,
            linecolor="black" if theme == "Light" else "white",
            linewidth=1,
            mirror=False,                       # Bottom only; True: Bottom and Top
            ticks="outside",                    # ticks outside
            gridcolor="rgba(128,128,128,0.5)",  # Semi-transparent gray for visibility
            griddash='dash',                   # Dashed grid lines
            showgrid=True,
            range=xaxis_range if xaxis_range else None  # Apply user-specified date range
        ),
        yaxis=dict(
            showline=True,
            linecolor="black" if theme == "Light" else "white",
            linewidth=1,
            mirror=False,                       # Bottom only; True: Bottom and Top
            ticks="outside",                    # ticks outside
            gridcolor="rgba(128,128,128,0.5)",  # Semi-transparent gray for visibility
            griddash='dash',                   # Dashed grid lines
            showgrid=True,
            range=yaxis_range if yaxis_range else None  # Apply user-specified y-axis range
        )
    )
    
    # Ensure y-axis includes zero if start_zero is True
    if start_zero:
        fig.update_yaxes(zeroline=True, zerolinecolor='gray', zerolinewidth=1)
    
    return fig
