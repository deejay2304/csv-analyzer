# app.py
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from io import StringIO

# Page setup
st.set_page_config(page_title="CSV Analyzer", layout="centered")
st.title("📊 CSV Analyzer — Upload CSV & See Metrics")

uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])

@st.cache_data
def compute_metrics(df: pd.DataFrame):
    overview = {
        "rows": len(df),
        "columns": len(df.columns),
        "memory (KB)": round(df.memory_usage(deep=True).sum() / 1024, 2)
    }

    col_stats = {}
    for col in df.columns:
        series = df[col]
        info = {
            "dtype": str(series.dtype),
            "missing": int(series.isna().sum())
        }

        if pd.api.types.is_numeric_dtype(series):
            info["mean"] = float(series.mean())
            info["sum"] = float(series.sum())
            info["min"] = float(series.min())
            info["max"] = float(series.max())
        else:
            top_vals = series.fillna("(missing)").value_counts().head(5)
            info["top_values"] = top_vals.to_dict()

        col_stats[col] = info

    return overview, col_stats

def show_overview(overview):
    st.subheader("Dataset Overview")
    c1, c2, c3 = st.columns(3)
    c1.metric("Rows", overview["rows"])
    c2.metric("Columns", overview["columns"])
    c3.metric("Memory (KB)", overview["memory (KB)"])

def show_column_details(df, stats):
    st.subheader("Inspect Columns")
    col = st.selectbox("Select a column", df.columns)
    col_data = stats[col]

    st.write(f"**Type:** {col_data['dtype']} | **Missing:** {col_data['missing']}")
    if "mean" in col_data:
        st.write(f"**Mean:** {col_data['mean']:.2f} | **Sum:** {col_data['sum']:.2f}")
        st.write(f"**Min:** {col_data['min']:.2f} | **Max:** {col_data['max']:.2f}")
        fig = px.histogram(df, x=col, title=f"Distribution of {col}")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.write("Top values:")
        top_vals = pd.Series(col_data["top_values"])
        st.table(top_vals)
        fig = px.bar(top_vals, x=top_vals.index, y=top_vals.values, title=f"Top values in {col}")
        st.plotly_chart(fig, use_container_width=True)

def show_preview(df):
    st.subheader("Preview Data")
    st.dataframe(df.head(20))

# Main logic
if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file)
        st.success("✅ File uploaded successfully!")
        overview, stats = compute_metrics(df)
        show_overview(overview)
        show_preview(df)
        show_column_details(df, stats)
    except Exception as e:
        st.error(f"❌ Error reading file: {e}")
else:
    st.info("👆 Upload a CSV file to start.")
