import streamlit as st
import pandas as pd

# Default US recession periods from 1949 to 2025
default_recessions = [
    {"name": "COVID-19", "start": "2020/02/01", "end": "2020/04/01"},
    {"name": "Great Recession", "start": "2007/12/01", "end": "2009/06/01"},
    {"name": "Dot-com Bust", "start": "2001/03/01", "end": "2001/11/01"},
    {"name": "Gulf War", "start": "1990/07/01", "end": "1991/03/01"},
    {"name": "Volcker Shock II", "start": "1981/07/01", "end": "1982/11/01"},
    {"name": "Volcker Shock I", "start": "1980/01/01", "end": "1980/07/01"},
    {"name": "First Oil Crisis", "start": "1973/11/01", "end": "1975/03/01"},
    {"name": "Vietnam War", "start": "1969/12/01", "end": "1970/11/01"},
    {"name": "Pre-Kennedy", "start": "1960/04/01", "end": "1961/02/01"},
    {"name": "Industrial slowdown", "start": "1957/08/01", "end": "1958/04/01"},
    {"name": "Post–Korean War", "start": "1953/07/01", "end": "1954/05/01"},
    {"name": "Postwar Inflation Correction", "start": "1948/11/01", "end": "1949/10/01"}
]

@st.cache_data(show_spinner=False)
def load_recession_data(file=None):
    """
    Load recession data from default list or user-uploaded CSV.
    
    Args:
        file: Uploaded CSV file with columns 'name', 'start', 'end' (optional).
    
    Returns:
        List of dictionaries with keys 'name', 'start', 'end'.
    
    Raises:
        ValueError: If CSV lacks required columns or has invalid dates.
    """
    if file is None:
        return default_recessions
    
    try:
        df = pd.read_csv(file)
        required_cols = ['name', 'start', 'end']
        if not all(col in df.columns for col in required_cols):
            raise ValueError("Recession CSV must have 'name', 'start', 'end' columns")
        
        # Validate date formats
        for col in ['start', 'end']:
            try:
                df[col] = pd.to_datetime(df[col], format='%Y/%m/%d', errors='raise')
            except ValueError:
                raise ValueError(f"Invalid date format in '{col}' column. Use yyyy/mm/dd.")
        
        return [
            {'name': str(row['name']), 'start': row['start'].strftime('%Y/%m/%d'), 'end': row['end'].strftime('%Y/%m/%d')}
            for _, row in df.iterrows()
        ]
    except Exception as e:
        st.error(f"Error loading recession CSV: {str(e)}")
        return default_recessions  # Fallback to default on error