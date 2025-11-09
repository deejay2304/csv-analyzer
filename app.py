# app.py — Corrected dashboard for your data.csv
import streamlit as st
import pandas as pd
import plotly.express as px

st.set_page_config(page_title="E-commerce Dashboard", layout="wide")

# ---------------------- Load Data ----------------------
@st.cache_data
def load_data(path):
    df = pd.read_csv(path, parse_dates=['order_date'])
    df['sales'] = df['price'] * df['qty']
    df['profit'] = (df['price'] - df['cogs']) * df['qty']
    return df

# Upload or read CSV
uploaded = st.file_uploader("Upload your data.csv", type=["csv"])
if uploaded:
    df = load_data(uploaded)
else:
    st.stop()

# ---------------------- Filter Data ----------------------
st.sidebar.header("Filters")

# Date Range
min_date, max_date = df['order_date'].min().date(), df['order_date'].max().date()
date_range = st.sidebar.date_input("Order Date Range", [min_date, max_date])

# Channel Filter
channels = sorted(df['channel'].dropna().unique())
selected_channels = st.sidebar.multiselect("Select Channel(s)", options=channels, default=channels)

# Style Filter
styles = sorted(df['style_id'].dropna().unique())
selected_styles = st.sidebar.multiselect("Select Style(s)", options=styles, default=[])

# Order Status Filter
statuses = sorted(df['order_status'].dropna().unique())
selected_statuses = st.sidebar.multiselect("Select Order Status", options=statuses, default=["Delivered"])

# Filter logic
mask = (
    (df['order_date'].dt.date >= date_range[0])
    & (df['order_date'].dt.date <= date_range[1])
    & (df['channel'].isin(selected_channels))
    & (df['order_status'].isin(selected_statuses))
)
if selected_styles:
    mask &= df['style_id'].isin(selected_styles)

filtered = df[mask].copy()

# ---------------------- KPIs ----------------------
st.title("📊 Fashion E-commerce Dashboard")
st.markdown("Showing data for selected filters")

k1, k2, k3, k4 = st.columns(4)
total_sales = filtered['sales'].sum()
total_profit = filtered['profit'].sum()
total_orders = filtered['order_id'].nunique()
total_qty = filtered['qty'].sum()

k1.metric("Total Sales", f"₹{total_sales:,.0f}")
k2.metric("Total Profit", f"₹{total_profit:,.0f}")
k3.metric("Total Orders", f"{total_orders:,}")
k4.metric("Total Quantity", f"{total_qty:,}")

st.markdown("---")

# ---------------------- Sales Trend ----------------------
st.subheader("📆 Sales Trend Over Time")
sales_trend = (
    filtered.groupby(filtered['order_date'].dt.date)['sales'].sum().reset_index()
)
fig_trend = px.line(sales_trend, x='order_date', y='sales',
                    title="Sales Over Time", labels={'order_date': 'Date', 'sales': 'Sales (₹)'})
st.plotly_chart(fig_trend, use_container_width=True)

# ---------------------- Channel-wise Sales ----------------------
st.subheader("🛍 Channel-wise Sales")
channel_sales = filtered.groupby('channel')['sales'].sum().reset_index().sort_values('sales', ascending=False)
fig_channel = px.bar(channel_sales, x='channel', y='sales', text_auto=True,
                     title="Sales by Channel", labels={'sales': 'Sales (₹)'})
st.plotly_chart(fig_channel, use_container_width=True)

# ---------------------- Style Performance ----------------------
st.subheader("👗 Top Performing Styles")
style_perf = (
    filtered.groupby('style_id')[['sales', 'profit', 'qty']]
    .sum().reset_index().sort_values('sales', ascending=False)
)
fig_style = px.bar(style_perf.head(15), x='style_id', y='sales', text_auto=True,
                   title="Top 15 Styles by Sales", labels={'sales': 'Sales (₹)'})
st.plotly_chart(fig_style, use_container_width=True)

# ---------------------- Size & Color Mix ----------------------
c1, c2 = st.columns(2)

with c1:
    st.subheader("📏 Size Mix")
    size_mix = filtered.groupby('size')['qty'].sum().reset_index().sort_values('qty', ascending=False)
    fig_size = px.bar(size_mix, x='size', y='qty', text_auto=True,
                      title="Quantity Sold by Size", labels={'qty': 'Quantity'})
    st.plotly_chart(fig_size, use_container_width=True)

with c2:
    st.subheader("🎨 Color Mix")
    color_mix = filtered.groupby('color')['qty'].sum().reset_index().sort_values('qty', ascending=False).head(10)
    fig_color = px.pie(color_mix, values='qty', names='color', title="Top 10 Colors by Quantity Sold")
    st.plotly_chart(fig_color, use_container_width=True)

# ---------------------- Cancellation Analysis ----------------------
st.subheader("❌ Cancellation Rate Over Time")
cancel_df = df.groupby([df['order_date'].dt.date, 'order_status'])['order_id'].nunique().reset_index()
cancel_pivot = cancel_df.pivot(index='order_date', columns='order_status', values='order_id').fillna(0)
cancel_pivot['Cancel Rate (%)'] = (cancel_pivot.get('Cancelled', 0) / cancel_pivot.sum(axis=1)) * 100
fig_cancel = px.line(cancel_pivot.reset_index(), x='order_date', y='Cancel Rate (%)',
                     title="Cancellation Rate Over Time (%)")
st.plotly_chart(fig_cancel, use_container_width=True)

# ---------------------- Data Table ----------------------
st.markdown("---")
st.subheader("📋 Filtered Orders Preview")
st.dataframe(filtered.head(500))
csv = filtered.to_csv(index=False).encode('utf-8')
st.download_button("Download filtered data as CSV", csv, "filtered_orders.csv", "text/csv")

st.caption("Note: Sales = price × qty, Profit = (price − cogs) × qty, Only Delivered orders are used by default.")
