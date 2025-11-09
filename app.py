# app.py
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from datetime import datetime

st.set_page_config(layout="wide", page_title="Fashion E-commerce Dashboard")

# ---------- Small helper to read CSV when available ----------
@st.cache_data
def read_csv_from_path(path: str):
    return pd.read_csv(path, parse_dates=['order_date'])

def clean_and_enrich(df: pd.DataFrame) -> pd.DataFrame:
    # basic cleaning / derived columns
    df = df.copy()
    df['qty'] = pd.to_numeric(df.get('qty', 0), errors='coerce').fillna(0).astype(int)
    df['price'] = pd.to_numeric(df.get('price', 0.0), errors='coerce').fillna(0.0)
    df['cogs'] = pd.to_numeric(df.get('cogs', 0.0), errors='coerce').fillna(0.0)
    # settlement amount: prefer 'stlmnt' or fallback to price*qty
    if 'stlmnt' in df.columns:
        df['settlement_amount'] = pd.to_numeric(df.get('stlmnt'), errors='coerce').fillna(df['price'] * df['qty'])
    else:
        df['settlement_amount'] = df['price'] * df['qty']
    df['profit'] = df['settlement_amount'] - (df['cogs'] * df['qty'])
    # ensure order_date parsed
    if 'order_date' in df.columns:
        df['order_date'] = pd.to_datetime(df['order_date'], errors='coerce')
    else:
        st.warning("CSV has no 'order_date' column — date-based visuals will be disabled.")
    return df

# ---------- Try to load data.csv from repo first ----------
df = None
CSV_PATH = "data.csv"

try:
    df = read_csv_from_path(CSV_PATH)
except FileNotFoundError:
    st.info(f"Could not find '{CSV_PATH}' in the app folder.")
except Exception as e:
    # Show the error so it's easier to debug during development
    st.error(f"Error reading '{CSV_PATH}': {e}")

# ---------- If not found, ask user to upload a CSV via browser ----------
if df is None:
    st.sidebar.markdown("## Upload CSV (fallback)")
    uploaded_file = st.sidebar.file_uploader("Upload your data.csv (or any CSV with required columns)", type=["csv"])
    if uploaded_file is None:
        st.sidebar.markdown(
            """
            **How to fix:**  
            1. Put `data.csv` in the root of your app repository and redeploy (recommended for Streamlit Cloud).  
            2. Or upload the CSV here from your local machine to run the app now.
            """
        )
        st.stop()
    else:
        try:
            df = pd.read_csv(uploaded_file, parse_dates=['order_date'])
        except Exception as e:
            st.error(f"Uploaded file could not be read as CSV: {e}")
            st.stop()

# ---------- Clean/enrich dataframe ----------
df = clean_and_enrich(df)

# ---------- Sidebar filters ----------
st.sidebar.header("Filters")
min_date = df['order_date'].min().date() if 'order_date' in df.columns and pd.notnull(df['order_date'].min()) else None
max_date = df['order_date'].max().date() if 'order_date' in df.columns and pd.notnull(df['order_date'].max()) else None

if min_date and max_date:
    date_range = st.sidebar.date_input("Order date range", [min_date, max_date])
else:
    date_range = None

channels = sorted(df['channel'].dropna().unique()) if 'channel' in df.columns else []
channels_selected = st.sidebar.multiselect("Channel(s)", options=channels, default=channels)

styles = sorted(df['style_id'].dropna().unique()) if 'style_id' in df.columns else []
styles_selected = st.sidebar.multiselect("Style ID(s)", options=styles, default=None)

sizes = sorted(df['size'].dropna().unique()) if 'size' in df.columns else []
sizes_selected = st.sidebar.multiselect("Size(s)", options=sizes, default=None)

# apply filters
mask = pd.Series(True, index=df.index)
if date_range and len(date_range) == 2 and 'order_date' in df.columns:
    start_dt = pd.to_datetime(date_range[0])
    end_dt = pd.to_datetime(date_range[1])
    mask &= (df['order_date'] >= start_dt) & (df['order_date'] <= end_dt)
if channels_selected:
    mask &= df['channel'].isin(channels_selected)
if styles_selected:
    mask &= df['style_id'].isin(styles_selected)
if sizes_selected:
    mask &= df['size'].isin(sizes_selected)

dff = df[mask].copy()

# ---------- Top row: KPIs ----------
st.title("Fashion E-commerce Dashboard")
kpi1, kpi2, kpi3, kpi4 = st.columns(4)

total_sales = dff['settlement_amount'].sum() if not dff.empty else 0
total_orders = dff['order_id'].nunique() if 'order_id' in dff.columns else len(dff)
total_qty = dff['qty'].sum() if 'qty' in dff.columns else 0
avg_order_value = total_sales / total_orders if total_orders else 0
total_profit = dff['profit'].sum() if 'profit' in dff.columns else 0

kpi1.metric("Total Sales", f"₹{total_sales:,.0f}")
kpi2.metric("Total Orders", f"{total_orders:,}")
kpi3.metric("Quantity Sold", f"{total_qty:,}")
kpi4.metric("Total Profit", f"₹{total_profit:,.0f}")

st.markdown("---")

# ---------- Charts ----------
st.subheader("Sales Trend")
if not dff.empty and 'order_date' in dff.columns:
    sales_by_date = dff.groupby(pd.Grouper(key='order_date', freq='D')).agg({'settlement_amount':'sum'}).reset_index()
    fig_trend = px.line(sales_by_date, x='order_date', y='settlement_amount', title="Daily Sales", labels={'order_date':'Date','settlement_amount':'Sales (₹)'})
    st.plotly_chart(fig_trend, use_container_width=True)
else:
    st.info("No date-based sales data available for the selected filters.")

st.subheader("Sales by Channel")
if 'channel' in dff.columns and not dff.empty:
    channel_agg = dff.groupby('channel').agg({'settlement_amount':'sum','order_id':'nunique'}).rename(columns={'order_id':'orders'}).reset_index()
    fig_channel = px.bar(channel_agg.sort_values('settlement_amount', ascending=False),
                         x='channel', y='settlement_amount', text='orders',
                         title="Sales by Channel")
    fig_channel.update_layout(xaxis={'categoryorder':'total descending'})
    st.plotly_chart(fig_channel, use_container_width=True)
else:
    st.write("No channel data.")

st.subheader("Profit & Performance by Style")
if 'style_id' in dff.columns and not dff.empty:
    top_styles = (dff.groupby('style_id')
                    .agg(total_sales=('settlement_amount','sum'),
                         total_qty=('qty','sum'),
                         avg_profit=('profit','mean'))
                    .reset_index()
                    .sort_values('total_sales', ascending=False).head(10))
    st.markdown("**Top styles by sales (top 10)**")
    st.dataframe(top_styles.style.format({'total_sales':'₹{:,}','avg_profit':'₹{:.2f}','total_qty':'{:,}'}))
    fig_style_profit = px.box(dff, x='style_id', y='profit', title='Profit distribution by style (all styles)', points='outliers')
    st.plotly_chart(fig_style_profit, use_container_width=True)
else:
    st.write("No style_id data.")

left, right = st.columns(2)
with left:
    st.subheader("Quantity by Size")
    if 'size' in dff.columns and not dff.empty:
        size_agg = dff.groupby('size').agg({'qty':'sum'}).reset_index().sort_values('qty', ascending=False)
        fig_size = px.bar(size_agg, x='size', y='qty', title='Quantity sold by size', labels={'qty':'Quantity'})
        st.plotly_chart(fig_size, use_container_width=True)
    else:
        st.write("No size data.")

with right:
    st.subheader("Top Colors")
    if 'color' in dff.columns and not dff.empty:
        color_agg = dff.groupby('color').agg({'qty':'sum'}).reset_index().sort_values('qty', ascending=False).head(10)
        fig_color = px.pie(color_agg, values='qty', names='color', title='Top 10 Colors by Quantity')
        st.plotly_chart(fig_color, use_container_width=True)
    else:
        st.write("No color data.")

st.markdown("---")
st.subheader("Orders (sample)")
if not dff.empty:
    st.dataframe(dff.sort_values('order_date', ascending=False).head(500))
    csv = dff.to_csv(index=False).encode('utf-8')
    st.download_button("Download filtered data as CSV", data=csv, file_name='filtered_orders.csv', mime='text/csv')
else:
    st.info("No orders to show for selected filters.")

st.markdown("---")
st.caption("Columns used: order_date, order_id, sku_id, channel, qty, price, cogs, order_status, size, color, stlmnt, style_id, style_color_id")
