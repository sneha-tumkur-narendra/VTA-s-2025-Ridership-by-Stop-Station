import streamlit as st
import pandas as pd
import plotly.express as px

st.set_page_config(page_title="VTA Ridership Dashboard", layout="wide")

df = pd.read_csv("output.csv", low_memory=False)

# Clean numeric columns
numeric_cols = [
    "BOARDINGS", "ALIGHTINGS", "TRIPS",
    "AVG_BOARDINGS", "AVG_ALIGHTINGS", "AVG_ACTIVITY",
    "PASS_LOAD", "PEAK_LOAD", "AVG_PEAK_LOAD"
]

for col in numeric_cols:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

df = df.dropna(subset=["BOARDINGS"])

st.title("VTA Ridership Dashboard")

# Sidebar filters
st.sidebar.header("Filters")

routes = sorted(df["ROUTE_NAME"].dropna().unique())
selected_routes = st.sidebar.multiselect(
    "Select Routes",
    routes,
    default=routes[:5]
)

service_periods = sorted(df["SERVICE_PERIOD"].dropna().unique())
selected_periods = st.sidebar.multiselect(
    "Select Service Period",
    service_periods,
    default=service_periods
)

cities = sorted(df["CITY"].dropna().unique())
selected_cities = st.sidebar.multiselect(
    "Select City",
    cities,
    default=cities[:10]
)

filtered_df = df[
    df["ROUTE_NAME"].isin(selected_routes)
    & df["SERVICE_PERIOD"].isin(selected_periods)
    & df["CITY"].isin(selected_cities)
]

# KPIs
col1, col2, col3, col4 = st.columns(4)

col1.metric("Filtered Rows", f"{len(filtered_df):,}")
col2.metric("Total Boardings", f"{filtered_df['BOARDINGS'].sum():,.0f}")
col3.metric("Avg Boardings", f"{filtered_df['AVG_BOARDINGS'].mean():,.2f}")
col4.metric("Total Trips", f"{filtered_df['TRIPS'].sum():,.0f}")

# Chart 1: Total boardings by route
route_summary = (
    filtered_df.groupby("ROUTE_NAME", as_index=False)["BOARDINGS"]
    .sum()
    .sort_values("BOARDINGS", ascending=False)
)

fig1 = px.bar(
    route_summary,
    x="ROUTE_NAME",
    y="BOARDINGS",
    title="Total Boardings by Route",
    labels={"ROUTE_NAME": "Route", "BOARDINGS": "Total Boardings"}
)

fig1.update_layout(xaxis_tickangle=-45)

# Chart 2: Top stops by boardings
stop_summary = (
    filtered_df.groupby("STOP_DISPLAY", as_index=False)["BOARDINGS"]
    .sum()
    .sort_values("BOARDINGS", ascending=False)
    .head(15)
)

fig2 = px.bar(
    stop_summary,
    x="BOARDINGS",
    y="STOP_DISPLAY",
    orientation="h",
    title="Top 15 Stops by Boardings",
    labels={"STOP_DISPLAY": "Stop", "BOARDINGS": "Total Boardings"}
)

fig2.update_layout(yaxis={"categoryorder": "total ascending"})

# Chart 3: Boardings by city
city_summary = (
    filtered_df.groupby("CITY", as_index=False)["BOARDINGS"]
    .sum()
    .sort_values("BOARDINGS", ascending=False)
    .head(15)
)

fig3 = px.bar(
    city_summary,
    x="CITY",
    y="BOARDINGS",
    title="Top Cities by Boardings",
    labels={"CITY": "City", "BOARDINGS": "Total Boardings"}
)

fig3.update_layout(xaxis_tickangle=-45)

# Chart 4: Boardings vs alightings
fig4 = px.scatter(
    filtered_df,
    x="BOARDINGS",
    y="ALIGHTINGS",
    color="ROUTE_NAME",
    size="TRIPS",
    hover_data=["STOP_DISPLAY", "CITY", "SERVICE_PERIOD"],
    title="Boardings vs Alightings by Stop",
    labels={
        "BOARDINGS": "Boardings",
        "ALIGHTINGS": "Alightings",
        "ROUTE_NAME": "Route"
    }
)

# Display charts
left, right = st.columns(2)

with left:
    st.plotly_chart(fig1, use_container_width=True)
    st.plotly_chart(fig3, use_container_width=True)

with right:
    st.plotly_chart(fig2, use_container_width=True)
    st.plotly_chart(fig4, use_container_width=True)

st.subheader("Data Preview")
st.dataframe(filtered_df.head(100), use_container_width=True)