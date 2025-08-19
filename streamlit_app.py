import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from recession_data import load_recession_data
from utils import load_csv, preprocess_series, make_interactive_plot
import re

# Initialize session state for defaults and persistence
if "series_meta" not in st.session_state:
    st.session_state.series_meta = {}
if "selected_columns" not in st.session_state:
    st.session_state.selected_columns = []
if "reset_trigger" not in st.session_state:
    st.session_state.reset_trigger = 0

# App title and description
st.title("Time Series Explorer with Recession Context")
st.markdown(
    "Upload a CSV with time series data (Date in yyyy/mm/dd, numeric columns) to visualize rolling R² values with recession bands. "
    "Customize colors, line styles, and more, then download your plot as PNG or PDF."
)

# Sidebar for configuration
st.sidebar.header("Configuration")

# CSV file upload for time series data
data_file = st.sidebar.file_uploader("Upload your CSV", type="csv", help="CSV must have a 'Date' column in yyyy/mm/dd format and numeric columns.")

# Optional recession CSV upload
recession_file = st.sidebar.file_uploader(
    "Upload custom recession CSV (optional)",
    type="csv",
    help="CSV with columns: name, start, end (dates in yyyy/mm/dd). Leave blank to use default US recessions (1949–2025)."
)

# Load recession data
recessions = load_recession_data(recession_file)

# Initialize DataFrame
df = None
if data_file:
    df = load_csv(data_file)
    if df is None:
        st.error("Failed to load CSV. Please check the format and try again.")
        st.stop()

# Main UI for customization
if df is not None:
    # Column selection with default to first 3 numeric columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    default_cols = numeric_cols[:3] if len(numeric_cols) >= 3 else numeric_cols
    selected_columns = st.multiselect(
        "Choose series to plot",
        numeric_cols,
        default=default_cols if st.session_state.reset_trigger == 0 else st.session_state.selected_columns,
        help="Select one or more series to visualize."
    )
    st.session_state.selected_columns = selected_columns

    # Initialize series metadata
    default_colors = ['#005F73', '#0A9396', '#1f77b4', '#AFA194', '#492F20', '#746D5A', '#1A1A1A', '#94D2BD']  # Plotly default palette
    default_styles = ['solid', 'dash', 'dot', 'longdash']
    
    # Clear series_meta for unselected columns
    st.session_state.series_meta = {
        col: st.session_state.series_meta[col]
        for col in selected_columns if col in st.session_state.series_meta
    }
    
    # Assign colors and styles based on selection order
    for i, col in enumerate(selected_columns):
        if col not in st.session_state.series_meta:
            st.session_state.series_meta[col] = {
                "color": default_colors[i % len(default_colors)],  # Assign color based on selection index
                "style": default_styles[i % len(default_styles)]  # Assign style based on selection index
            }

    # Series customization
    st.sidebar.subheader("Series Customization")
    for col in selected_columns:
        st.sidebar.markdown(f"**{col}**")
        st.session_state.series_meta[col]["color"] = st.sidebar.color_picker(
            f"Color for {col}",
            value=st.session_state.series_meta[col]["color"],
            key=f"color_{col}",
            help="Choose a color for this series."
        )
        st.session_state.series_meta[col]["style"] = st.sidebar.selectbox(
            f"Line style for {col}",
            ["solid", "dash", "dot", "longdash"],
            index=default_styles.index(st.session_state.series_meta[col]["style"]) if st.session_state.series_meta[col]["style"] in default_styles else 0,
            key=f"style_{col}",
            help="Choose a line style for this series."
        )

    # Plot settings
    st.sidebar.subheader("Plot Settings")
    title = st.sidebar.text_input("Plot title", "Rolling R² for Forecasting Models", help="Enter the title for your plot.")
    xlabel = st.sidebar.text_input("X-axis label", "Date", help="Enter the label for the x-axis.")
    ylabel = st.sidebar.text_input("Y-axis label", "R²", help="Enter the label for the y-axis.")
    
    rolling_window = st.sidebar.selectbox(
        "Rolling average window",
        ["None", "30 days", "90 days", "180 days"],
        index=0,
        help="Apply a rolling average to smooth the series (e.g., 30 days)."
    )
    rolling_window_days = 0 if rolling_window == "None" else int(rolling_window.split()[0])
    
    normalize = st.sidebar.checkbox("Normalize to [0,1]", help="Scale each series to the range 0 to 1.")
    start_zero = st.sidebar.checkbox("Start each series at zero", help="Subtract the first value so each series starts at 0.")
    downsample = st.sidebar.checkbox("Downsample to weekly for performance", help="Reduce data to weekly averages for faster plotting.")
    theme = st.sidebar.selectbox("Theme", ["Light", "Dark"], help="Choose light or dark plot theme.")
    show_labels = st.sidebar.checkbox("Label recession periods", value=True, help="Show recession names above shaded bands.")

    # Axis range settings
    st.sidebar.subheader("Axis Range Settings")
    # X-axis (date range)
    min_date = df.index.min().date()
    max_date = df.index.max().date()
    xaxis_start = st.sidebar.date_input(
        "X-axis start date",
        value=min_date,
        min_value=min_date,
        max_value=max_date,
        help="Select the start date for the x-axis."
    )
    xaxis_end = st.sidebar.date_input(
        "X-axis end date",
        value=max_date,
        min_value=min_date,
        max_value=max_date,
        help="Select the end date for the x-axis."
    )
    # Validate date range
    if xaxis_start >= xaxis_end:
        st.sidebar.error("Start date must be before end date.")
        st.stop()

    # Y-axis (numeric range)
    y_min = float(df[numeric_cols].min().min())
    y_max = float(df[numeric_cols].max().max())
    yaxis_min = st.sidebar.number_input(
        "Y-axis minimum",
        value=y_min,
        step=0.1,
        help="Enter the minimum value for the y-axis."
    )
    yaxis_max = st.sidebar.number_input(
        "Y-axis maximum",
        value=y_max,
        step=0.1,
        help="Enter the maximum value for the y-axis."
    )
    # Validate y-axis range
    if yaxis_min >= yaxis_max:
        st.sidebar.error("Y-axis minimum must be less than maximum.")
        st.stop()

    # Download settings
    st.sidebar.subheader("Download Settings")
    filename = st.sidebar.text_input("Filename", "plot", help="Enter the filename for the downloaded plot (without extension).")
    format_choice = st.sidebar.selectbox("Format", ["PNG", "PDF"], help="Choose the format for the downloaded plot.")

    # Validate filename
    if not re.match(r'^[a-zA-Z0-9_-]+$', filename):
        st.error("Filename must contain only letters, numbers, underscores, or hyphens.")
        st.stop()

    # Reset button
    if st.sidebar.button("Reset to Defaults"):
        st.session_state.series_meta = {}
        st.session_state.selected_columns = []
        st.session_state.reset_trigger += 1
        st.experimental_rerun()

    # Generate plot button
    if st.button("Generate Plot") and selected_columns:
        # Downsample if selected
        plot_df = df[selected_columns].copy()
        if downsample:
            plot_df = plot_df.resample("W").mean()

        # Create series metadata dictionary
        series_meta = {col: st.session_state.series_meta[col] for col in selected_columns}

        # Generate plot
        try:
            fig = make_interactive_plot(
                df=plot_df,
                series_meta=series_meta,
                rolling_window=rolling_window_days,
                normalize=normalize,
                start_zero=start_zero,
                recessions=recessions,
                show_labels=show_labels,
                title=title,
                xlabel=xlabel,
                ylabel=ylabel,
                theme=theme,
                width=1200,  # High-resolution width
                height=600,  # High-resolution height
                xaxis_range=[xaxis_start, xaxis_end],  # User-specified date range
                yaxis_range=[yaxis_min, yaxis_max]     # User-specified y-axis range
            )
            # Display plot with high-quality rendering
            st.plotly_chart(
                fig,
                use_container_width=True,
                config={
                    'displayModeBar': True,
                    'toImageButtonOptions': {
                        'format': 'png',
                        'scale': 2,  # Increase resolution for browser download
                        'width': 1200,
                        'height': 600
                    }
                }
            )

            # Download button with high DPI
            img = fig.to_image(format=format_choice.lower(), scale=3, width=1200, height=600)
            mime = "image/png" if format_choice == "PNG" else "application/pdf"
            st.download_button(
                "Download Figure",
                data=img,
                file_name=f"{filename}.{format_choice.lower()}",
                mime=mime,
                help="Download the plot as PNG or PDF in high resolution."
            )
        except Exception as e:
            st.error(f"Error generating plot: {str(e)}")
    elif not selected_columns:
        st.warning("Please select at least one series to plot.")
else:
    st.info("Please upload a CSV file to start.")
