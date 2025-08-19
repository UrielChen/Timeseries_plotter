# Time-Series Plotter (Streamlit)

Upload a CSV and instantly get clean, publication-ready time-series charts.  
Supports rolling average, normalization, align-to-zero, US recession shading, axis ranges, and one-click PNG/PDF export.

**Live app:** https://timeseries-plotter.streamlit.app

---

## Features
- 📤 **CSV upload**: auto parse date index, numeric columns only
- 🧹 **Preprocess**: rolling mean, clip/winsorize (optional), normalize (min-max / z-score), align to zero
- 🗓 **Recession bands**: built-in US recessions with labels (toggle on/off)
- 📐 **Axes**: set x/y ranges; clean white theme with axis lines & grids
- 🎨 **Styling**: legend control, dark/light theme (optional)
- 📦 **Export**: high-res **PNG** / **PDF** (Plotly + Kaleido)

---

## Quick Start (Local)
```bash
# Python 3.9+ (recommended 3.11)
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
streamlit run streamlit_app.py
```
