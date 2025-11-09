# app.py
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from datetime import datetime

st.set_page_config(layout="wide", page_title="Fashion E-commerce Dashboard")

# ---------- Helpers ----------
@st.cache_data
def load_data(path="data.csv"):
    df = pd.read_csv(path, parse_dates=['order_date'])
    # basic cleaning / derived columns
    df['qty'] = pd.to_numeric(df['qty'], errors='coerce').fillna(0).astype(int)
    df['price'] = pd.to_numeric(df['price'], errors='coerce').fillna(0.0)
    df['cogs'] = pd.to_numeric(df['cogs'], errors='coerce').fillna(0.0)
    # settlement amount assumed same as price*qty unless separate column exists
    if 'stlmnt' in df.columns:
        df['settlement_amount'] = pd.to_numeric(df['stlmnt'], errors='coerce').fillna(df['price'] * df['qty'])
    else:
        df['settlement_amount'] = df['price'] * df['qty']
    df['profit'] = df['settlement_amount'] - (df['cogs'] * df['qty'])
    df['order_date'] = pd.to_datetime(df['order_date'])
    return df

# ---------- Load ----------
df = load_data("data.csv")

# ---------- Sidebar filters ----------
st.sidebar.header("Filters")
min_date = df['order_date'].min().date()
max_date = df['order_date'].max().date()
date_range = st.sidebar.date_input("Order date range", [min_date, max_date])

channels = st.sidebar.multiselect("Channel(s)", options=sorted(df['channel'].dropna().unique()), default=sorted(df['channel'].dropna().unique()))
styles = st.sidebar.multiselect("Style ID(s)", options=sorted(df['style_id'].dropna().unique()), default=None)
sizes = st.sidebar.multiselect("Size(s)", options=sorted(df['size'].dropna().unique()), default=None)

# apply filters
start_dt = pd.to_datetime(date_range[0])
end_dt = pd.to_datetime(date_range[1])
mask = (df['order_date'] >= start_dt) & (df['order_date'] <= end_dt)
if channels:
    mask &= df['channel'].isin(channels)
if styles:
    mask &= df['style_id'].isin(styles)
if sizes:
    mask &= df['size'].isin(sizes)

dff = df[mask].copy()

# ---------- Top row: KPIs ----------
st.title("Fashion E-commerce Dashboard")
kpi1, kpi2, kpi3, kpi4 = st.columns(4)

total_sales = dff['settlement_amount'].sum()
total_orders = dff['order_id'].nunique()
total_qty = dff['qty'].sum()
avg_order_value = total_sales / total_orders if total_orders else 0
total_profit = dff['profit'].sum()

kpi1.metric("Total Sales", f"₹{total_sales:,.0f}")
kpi2.metric("Total Orders", f"{total_orders:,}")
kpi3.metric("Quantity Sold", f"{total_qty:,}")
kpi4.metric("Total Profit", f"₹{total_profit:,.0f}")

st.markdown("---")

# ---------- Charts ----------
# 1) Sales trend
st.subheader("Sales Trend")
if not dff.empty:
    sales_by_date = dff.groupby(pd.Grouper(key='order_date', freq='D')).agg({'settlement_amount':'sum'}).reset_index()
    fig_trend = px.line(sales_by_date, x='order_date', y='settlement_amount', title="Daily Sales", labels={'order_date':'Date','settlement_amount':'Sales (₹)'})
    st.plotly_chart(fig_trend, use_container_width=True)
else:
    st.info("No data for the selected filters.")

# 2) Sales by channel
st.subheader("Sales by Channel")
channel_agg = dff.groupby('channel').agg({'settlement_amount':'sum','order_id':'nunique'}).rename(columns={'order_id':'orders'}).reset_index()
if not channel_agg.empty:
    fig_channel = px.bar(channel_agg.sort_values('settlement_amount', ascending=False),
                         x='channel', y='settlement_amount', text='orders',
                         title="Sales by Channel (click bars to inspect)")
    fig_channel.update_layout(xaxis={'categoryorder':'total descending'})
    st.plotly_chart(fig_channel, use_container_width=True)
else:
    st.write("No channel data.")

# 3) Profit by style (box + top styles)
st.subheader("Profit & Performance by Style")
if 'style_id' in dff.columns:
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
    st.write("No style_id column.")

# 4) Size & color mix
left, right = st.columns(2)
with left:
    st.subheader("Quantity by Size")
    size_agg = dff.groupby('size').agg({'qty':'sum'}).reset_index().sort_values('qty', ascending=False)
    if not size_agg.empty:
        fig_size = px.bar(size_agg, x='size', y='qty', title='Quantity sold by size', labels={'qty':'Quantity'})
        st.plotly_chart(fig_size, use_container_width=True)
    else:
        st.write("No size data.")

with right:
    st.subheader("Top Colors")
    color_agg = dff.groupby('color').agg({'qty':'sum'}).reset_index().sort_values('qty', ascending=False).head(10)
    if not color_agg.empty:
        fig_color = px.pie(color_agg, values='qty', names='color', title='Top 10 Colors by Quantity')
        st.plotly_chart(fig_color, use_container_width=True)
    else:
        st.write("No color data.")

st.markdown("---")

# ---------- Orders table with download ----------
st.subheader("Orders (sample)")
if not dff.empty:
    st.dataframe(dff.sort_values('order_date', ascending=False).head(500))
    csv = dff.to_csv(index=False).encode('utf-8')
    st.download_button("Download filtered data as CSV", data=csv, file_name='filtered_orders.csv', mime='text/csv')
else:
    st.info("No orders to show for selected filters.")

# ---------- Footer / notes ----------
st.markdown("---")
st.caption("Columns used: order_date, order_id, sku_id, channel, qty, price, cogs, order_status, size, color, stlmnt, style_id, style_color_id")

