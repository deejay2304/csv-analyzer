# app.py
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from io import StringIO
from datetime import timedelta

st.set_page_config(page_title="OMS Orders Analyzer", layout="wide")
st.title("OMS Orders Analyzer — Streamlit App (converted + SKU scatter & Cohorts)")

##########################
# Helper utilities
##########################

COMMON_ALIASES = {
    "order date": "order_date",
    "order_date": "order_date",
    "orderdate": "order_date",
    "ordered_at": "order_date",
    "placed_at": "order_date",

    "delivery date": "delivery_date",
    "delivery_date": "delivery_date",
    "delivered_at": "delivery_date",

    "return date": "return_date",
    "ret_del_date": "return_date",
    "return_date": "return_date",

    "channel name": "channel",
    "channel": "channel",
    "sales channel": "channel",

    "product sku code": "sku",
    "sku": "sku",
    "sku_code": "sku",
    "product_sku": "sku",

    "qty": "qty",
    "quantity": "qty",
    "ordered_qty": "qty",

    "total": "total",
    "order_value": "total",
    "amount": "total",

    "net profit": "net_profit",
    "net_profit": "net_profit",
    "profit": "net_profit",

    "expected profit": "expected_profit",
    "expected_profit": "expected_profit",

    "cost of goods": "cogs",
    "cogs": "cogs",
    "cost_price": "cogs",

    "payment received": "payment_received",
    "payment_received": "payment_received",

    "payment delta": "payment_delta",
    "payment_delta": "payment_delta",
    "payment_delay_days": "payment_delta",

    "order status": "order_status",
    "status": "order_status",

    "stlmnt": "settlement_status",
    "settlement": "settlement_status",
    "settlement_status": "settlement_status",

    "order id": "order_id",
    "order_id": "order_id",

    "buyer id": "buyer_id",
    "user_id": "buyer_id",
    "customer_id": "buyer_id",
    "buyerid": "buyer_id"
}

def canonicalize_columns(df: pd.DataFrame):
    new_cols = {}
    for c in df.columns:
        key = c.strip().lower()
        if key in COMMON_ALIASES:
            new_cols[c] = COMMON_ALIASES[key]
        else:
            k2 = key.replace(" ", "_").replace("-", "_")
            new_cols[c] = k2
    df = df.rename(columns=new_cols)
    return df

def to_datetime_safe(df, col_candidates):
    for c in col_candidates:
        if c in df.columns:
            try:
                df[c] = pd.to_datetime(df[c], errors="coerce")
            except Exception:
                df[c] = pd.to_datetime(df[c].astype(str), errors="coerce")
    return df

def numeric_safe(df, cols):
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df

@st.cache_data
def prepare_df_from_uploaded(uploaded_file):
    try:
        raw = uploaded_file.getvalue().decode("utf-8", errors="replace")
        df = pd.read_csv(StringIO(raw))
    except Exception:
        try:
            raw = uploaded_file.getvalue().decode("latin-1", errors="replace")
            df = pd.read_csv(StringIO(raw))
        except Exception as e:
            raise e

    df = canonicalize_columns(df)

    dt_candidates = ["order_date", "delivery_date", "return_date"]
    df = to_datetime_safe(df, dt_candidates)

    num_candidates = ["qty", "total", "net_profit", "expected_profit", "cogs", "payment_received", "payment_delta"]
    df = numeric_safe(df, num_candidates)

    if "order_date" in df.columns:
        df["order_date_only"] = df["order_date"].dt.date
        df["order_month"] = df["order_date"].dt.to_period("M").astype(str)
        df["order_week"] = df["order_date"].dt.to_period("W").astype(str)
        df["order_dayofweek"] = df["order_date"].dt.day_name()
    else:
        df["order_date_only"] = pd.NaT
        df["order_month"] = None
        df["order_week"] = None
        df["order_dayofweek"] = None

    if "order_status" in df.columns:
        df["order_status"] = df["order_status"].astype(str)
    if "channel" in df.columns:
        df["channel"] = df["channel"].astype(str)

    return df

##########################
# UI: Sidebar - filters
##########################
st.sidebar.header("Filters")
uploaded_file = st.sidebar.file_uploader("Upload orders CSV", type=["csv"])
sample_button = st.sidebar.button("Load sample (small) dataset")

if sample_button and uploaded_file is None:
    n = 300
    rng = np.random.default_rng(123)
    dates = pd.date_range(end=pd.Timestamp.today(), periods=90).to_pydatetime().tolist()
    df = pd.DataFrame({
        "order_id": np.arange(n),
        "order_date": rng.choice(dates, size=n),
        "channel": rng.choice(["Meesho", "Myntra", "Flipkart", "Website"], size=n),
        "order_status": rng.choice(["Delivered", "Returned", "Cancelled", "Pending"], size=n, p=[0.7,0.15,0.1,0.05]),
        "sku": rng.choice([f"SKU{i}" for i in range(1,31)], size=n),
        "qty": rng.integers(1,5,size=n),
        "total": rng.integers(100,3000,size=n),
    })
    df["net_profit"] = (df["total"] * (rng.uniform(0.05, 0.35, size=n))).round(2)
    df["payment_delta"] = rng.integers(0,30,size=n)
    df = df.assign(order_date=pd.to_datetime(df["order_date"]))
    df = canonicalize_columns(df)
else:
    df = None

if uploaded_file is not None:
    try:
        df = prepare_df_from_uploaded(uploaded_file)
        st.sidebar.success("File loaded")
    except Exception as e:
        st.sidebar.error(f"Could not read file: {e}")
        st.stop()

if df is None:
    st.info("Upload a CSV from the sidebar to analyze your real data, or click 'Load sample' to try a demo dataset.")
    st.stop()

min_date = df["order_date"].min() if "order_date" in df.columns else None
max_date = df["order_date"].max() if "order_date" in df.columns else None

if min_date is not None and not pd.isna(min_date):
    date_range = st.sidebar.date_input("Order Date range", value=(min_date.date(), max_date.date()))
    start, end = pd.to_datetime(date_range[0]), pd.to_datetime(date_range[1]) + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)
    mask_date = (df["order_date"] >= start) & (df["order_date"] <= end)
else:
    mask_date = pd.Series([True]*len(df), index=df.index)

channels = sorted(df["channel"].dropna().unique().tolist()) if "channel" in df.columns else []
sel_channels = st.sidebar.multiselect("Channels", options=channels, default=channels if channels else None)

statuses = sorted(df["order_status"].dropna().unique().tolist()) if "order_status" in df.columns else []
sel_statuses = st.sidebar.multiselect("Order statuses", options=statuses, default=statuses if statuses else None)

sku_search = st.sidebar.text_input("SKU contains (filter)", value="")

min_qty = int(st.sidebar.number_input("Min qty (filter)", min_value=0, value=0, step=1))

mask = mask_date.copy()
if sel_channels:
    mask = mask & df["channel"].isin(sel_channels)
if sel_statuses:
    mask = mask & df["order_status"].isin(sel_statuses)
if sku_search:
    mask = mask & df["sku"].astype(str).str.contains(sku_search, case=False, na=False)
if "qty" in df.columns:
    mask = mask & (df["qty"].fillna(0) >= min_qty)

df_f = df[mask].copy()

st.markdown("## Key metrics")
kpi1, kpi2, kpi3, kpi4 = st.columns(4)
total_orders = int(df_f.shape[0])
total_revenue = float(df_f["total"].sum()) if "total" in df_f.columns else None
total_profit = float(df_f["net_profit"].sum()) if "net_profit" in df_f.columns else None
return_rate = None
if "order_status" in df_f.columns:
    total_returns = int((df_f["order_status"].str.lower() == "returned").sum())
    return_rate = round(100 * total_returns / total_orders, 2) if total_orders else 0.0

kpi1.metric("Orders", f"{total_orders}")
kpi2.metric("Revenue", f"{total_revenue:,.2f}" if total_revenue is not None else "n/a")
kpi3.metric("Net Profit", f"{total_profit:,.2f}" if total_profit is not None else "n/a")
kpi4.metric("Return Rate", f"{return_rate} %" if return_rate is not None else "n/a")

st.markdown("---")

##########################
# 1) Time-series: Revenue & Net Profit with rolling avg
##########################
st.subheader("Revenue & Net Profit over time")
if "order_date" in df_f.columns and ("total" in df_f.columns or "net_profit" in df_f.columns):
    tmp = df_f.dropna(subset=["order_date"]).copy()
    tmp = tmp.set_index("order_date").sort_index()
    daily = tmp.resample("D").agg({
        "total": "sum" if "total" in tmp.columns else "sum",
        "net_profit": "sum" if "net_profit" in tmp.columns else "sum"
    }).fillna(0)
    daily["rev_7d"] = daily.get("total", 0).rolling(7, min_periods=1).mean()
    daily["profit_7d"] = daily.get("net_profit", 0).rolling(7, min_periods=1).mean()

    fig = px.line(daily.reset_index(), x="order_date", y=["total", "net_profit"], labels={"value":"Amount", "order_date":"Date"}, title="Daily totals")
    fig.update_traces(mode="lines+markers")
    if "rev_7d" in daily.columns:
        fig.add_scatter(x=daily.reset_index()["order_date"], y=daily["rev_7d"], mode="lines", name="Revenue (7d MA)", line=dict(dash="dash"))
    if "profit_7d" in daily.columns:
        fig.add_scatter(x=daily.reset_index()["order_date"], y=daily["profit_7d"], mode="lines", name="Profit (7d MA)", line=dict(dash="dash"))
    st.plotly_chart(fig, use_container_width=True)
else:
    st.info("Time-series requires 'order_date' + 'total' or 'net_profit' columns.")

##########################
# 2) Channel x Status stacked bar
##########################
st.subheader("Orders by Channel and Status")
if "channel" in df_f.columns and "order_status" in df_f.columns:
    pivot = df_f.pivot_table(index="channel", columns="order_status", values="order_id" if "order_id" in df_f.columns else "total", aggfunc="count", fill_value=0)
    pivot = pivot.reset_index()
    melted = pivot.melt(id_vars="channel", var_name="order_status", value_name="count")
    fig2 = px.bar(melted, x="channel", y="count", color="order_status", title="Orders by Channel and Status", barmode="stack")
    st.plotly_chart(fig2, use_container_width=True)
else:
    st.info("Channel x Status requires 'channel' and 'order_status' columns.")

##########################
# 3) Top SKUs Pareto
##########################
st.subheader("Top SKUs (Pareto by Revenue or Profit)")
metric_choice = st.selectbox("Metric for ranking SKUs", options=["total", "net_profit", "qty"] , index=0 if "total" in df_f.columns else 1)
if metric_choice not in df_f.columns:
    st.info(f"Metric '{metric_choice}' not found in data. Choose a different metric or upload data with that column.")
else:
    sku_grp = df_f.groupby("sku").agg({metric_choice: "sum", "order_id": "count" if "order_id" in df_f.columns else ("qty" if "qty" in df_f.columns else "total")})
    sku_grp = sku_grp.rename(columns={metric_choice: "metric_sum"}).sort_values("metric_sum", ascending=False).reset_index()
    sku_grp["cum_pct"] = sku_grp["metric_sum"].cumsum() / sku_grp["metric_sum"].sum() * 100
    top_n = st.slider("Number of top SKUs to show", min_value=5, max_value=min(100, len(sku_grp)), value=min(20, len(sku_grp)))
    sku_plot = sku_grp.head(top_n)
    fig3 = px.bar(sku_plot, x="sku", y="metric_sum", title=f"Top {top_n} SKUs by {metric_choice}")
    fig3.update_layout(xaxis_tickangle=-45)
    st.plotly_chart(fig3, use_container_width=True)
    fig3b = px.line(sku_grp.head(max(100, top_n)), x="sku", y="cum_pct", title="Cumulative % contribution (Pareto)", markers=True)
    fig3b.update_layout(xaxis_tickangle=-45, yaxis_title="Cumulative %")
    st.plotly_chart(fig3b, use_container_width=True)

##########################
# 3b) SKU margin scatter (new)
##########################
st.subheader("SKU margin scatter (Revenue vs Margin%)")
# Requirements: sku, revenue (total), profit (net_profit), qty optional
if "sku" in df_f.columns and ("total" in df_f.columns or "net_profit" in df_f.columns):
    # compute aggregates
    agg_cols = {}
    if "total" in df_f.columns:
        agg_cols["total"] = ("total", "sum")
    if "net_profit" in df_f.columns:
        agg_cols["net_profit"] = ("net_profit", "sum")
    if "qty" in df_f.columns:
        agg_cols["qty"] = ("qty", "sum")

    # using groupby.agg with tuple mapping:
    grouped = df_f.groupby("sku").agg({k: v[1] for k,v in agg_cols.items()}).rename(columns={c: c for c in agg_cols.keys()})
    # Ensure columns exist
    revenue_col = "total" if "total" in grouped.columns else None
    profit_col = "net_profit" if "net_profit" in grouped.columns else None
    qty_col = "qty" if "qty" in grouped.columns else None

    # compute margin %
    if revenue_col and profit_col:
        grouped = grouped.reset_index()
        # avoid divide by zero
        grouped["margin_pct"] = np.where(grouped[revenue_col] != 0, 100.0 * grouped[profit_col] / grouped[revenue_col], np.nan)
        grouped["revenue"] = grouped[revenue_col]
        grouped["profit"] = grouped[profit_col]
        if qty_col:
            grouped["qty"] = grouped[qty_col]

        min_rev = st.number_input("Min revenue to include SKU", min_value=0, value=0, step=1)
        top_k = st.slider("Max SKUs to plot", min_value=10, max_value=min(500, len(grouped)), value=min(200, len(grouped)))

        plotted = grouped[grouped["revenue"] >= min_rev].sort_values("revenue", ascending=False).head(top_k)
        if plotted.shape[0] == 0:
            st.info("No SKUs meet the revenue / filter criteria.")
        else:
            size_field = "qty" if "qty" in plotted.columns else "revenue"
            hover_data = ["sku", "revenue", "profit", "margin_pct"]
            if "qty" in plotted.columns:
                hover_data.append("qty")
            fig_scatter = px.scatter(plotted, x="revenue", y="margin_pct", size=size_field, hover_data=hover_data,
                                     title="SKU: Revenue vs Margin (%) — point size ~ qty or revenue", labels={"margin_pct":"Margin (%)","revenue":"Revenue"})
            fig_scatter.update_traces(marker=dict(opacity=0.8))
            st.plotly_chart(fig_scatter, use_container_width=True)
    else:
        st.info("SKU margin scatter requires both 'total' (revenue) and 'net_profit' columns.")
else:
    st.info("SKU scatter requires 'sku' and revenue/profit columns.")

##########################
# 4) Return rate by SKU
##########################
st.subheader("Return rate by SKU")
if "sku" in df_f.columns and "order_status" in df_f.columns:
    sku_counts = df_f.groupby("sku").agg(total_sold=("order_id" if "order_id" in df_f.columns else "qty", "count" if "order_id" in df_f.columns else "sum"),
                                         returns=("order_status", lambda s: (s.str.lower()=="returned").sum()))
    sku_counts = sku_counts.reset_index()
    sku_counts["return_rate_pct"] = 100 * sku_counts["returns"] / sku_counts["total_sold"].replace(0, np.nan)
    min_vol = st.number_input("Min total sold to include in this chart", min_value=1, value=5)
    filtered_sku = sku_counts[sku_counts["total_sold"] >= min_vol].sort_values("return_rate_pct", ascending=False).head(30)
    if filtered_sku.shape[0] == 0:
        st.info("No SKUs meet the min volume filter.")
    else:
        fig4 = px.bar(filtered_sku, x="sku", y="return_rate_pct", title="Top SKUs by Return Rate (%)")
        fig4.update_layout(xaxis_tickangle=-45, yaxis_title="Return rate (%)")
        st.plotly_chart(fig4, use_container_width=True)
else:
    st.info("Return rate requires 'sku' and 'order_status' columns.")

##########################
# 5) Payment delay distribution
##########################
st.subheader("Payment delay distribution (days)")
if "payment_delta" in df_f.columns:
    pd_clean = df_f["payment_delta"].dropna().astype(float)
    if len(pd_clean) == 0:
        st.info("No payment delta values present.")
    else:
        fig5 = px.histogram(pd_clean, nbins=30, title="Payment delay distribution (days)")
        st.plotly_chart(fig5, use_container_width=True)
        st.write("Summary:")
        st.write(pd_clean.describe().to_frame().T)
        p90 = np.nanpercentile(pd_clean, 90)
        st.write(f"90th percentile payment delay: {p90:.0f} days")
else:
    st.info("Add 'payment_delta' column (days) to view payment delay distribution.")

st.markdown("---")

##########################
# Cohort heatmap (new)
##########################
st.subheader("Customer Cohort heatmap (by first order month)")
# Cohort calculation requires a customer identifier. We'll look for common candidate columns.
buyer_col = None
for cand in ["buyer_id", "buyer", "customer_id", "user_id"]:
    if cand in df_f.columns:
        buyer_col = cand
        break

if buyer_col is None:
    st.info("Cohort heatmap requires a customer identifier column such as 'buyer_id' or 'customer_id'. Your file has none of the common names.")
else:
    # Build cohort table
    # - For each buyer, find first order month (cohort)
    # - For each order, compute month index (months since cohort)
    try:
        temp = df_f.dropna(subset=["order_date", buyer_col]).copy()
        # ensure order_date is datetime
        temp["order_month"] = temp["order_date"].dt.to_period("M").dt.to_timestamp()
        # first purchase month per buyer
        first = temp.groupby(buyer_col)["order_month"].min().reset_index().rename(columns={"order_month":"cohort_month"})
        temp = temp.merge(first, on=buyer_col, how="left")
        # months difference
        def month_diff(row):
            return (row["order_month"].year - row["cohort_month"].year) * 12 + (row["order_month"].month - row["cohort_month"].month)
        temp["period_index"] = temp.apply(month_diff, axis=1)

        # cohort counts: number of unique buyers active per cohort-period
        cohort_counts = temp.groupby(["cohort_month", "period_index"])[buyer_col].nunique().reset_index().rename(columns={buyer_col:"n_buyers"})

        # pivot
        cohort_pivot = cohort_counts.pivot(index="cohort_month", columns="period_index", values="n_buyers").fillna(0).astype(int)

        if cohort_pivot.shape[0] == 0:
            st.info("Not enough data to build cohort table.")
        else:
            # Optionally normalize each cohort row to show retention %
            normalize = st.checkbox("Show retention % (normalize row-wise)", value=True)
            display_df = cohort_pivot.copy()
            if normalize:
                display_df = (display_df.T / display_df.iloc[:,0]).T * 100
                z_title = "Retention %"
            else:
                z_title = "Active buyers"

            # sort cohorts ascending by cohort_month
            display_df = display_df.sort_index()

            # Prepare x and y labels
            x_labels = [f"Month {int(i)}" for i in display_df.columns]
            y_labels = [str(pd.to_datetime(idx).strftime("%Y-%m")) for idx in display_df.index]

            fig_cohort = px.imshow(display_df.values,
                                   labels=dict(x="Period (months since first order)", y="Cohort (first order month)", color=z_title),
                                   x=x_labels, y=y_labels,
                                   aspect="auto",
                                   text_auto=False)
            fig_cohort.update_xaxes(side="bottom")
            st.plotly_chart(fig_cohort, use_container_width=True)

            st.write("Cohort table (raw numbers):")
            st.dataframe(cohort_pivot.head(100))
    except Exception as e:
        st.error(f"Could not compute cohort heatmap: {e}")

st.markdown("---")

##########################
# Data explorer & download
##########################
st.subheader("Filtered dataset preview & download")
st.write("Rows shown:", df_f.shape[0], "Columns:", df_f.shape[1])
st.dataframe(df_f.head(200), use_container_width=True)

@st.cache_data
def convert_df_to_csv_bytes(dff):
    return dff.to_csv(index=False).encode("utf-8")

csv_bytes = convert_df_to_csv_bytes(df_f)
st.download_button("Download filtered data (CSV)", data=csv_bytes, file_name="filtered_orders.csv", mime="text/csv")

st.markdown("App built from your notebook — contact me if you want more charts or a one-to-one port of every analysis cell.")
