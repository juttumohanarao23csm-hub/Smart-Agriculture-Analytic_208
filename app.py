import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import json
from pathlib import Path

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Smart Agriculture Analytics",
    page_icon="🌱",
    layout="wide"
)

# ---------------- THEME ----------------
st.markdown("""
<style>
.stApp {
    background-color: #f4f8f4;
}
[data-testid="stSidebar"] {
    background-color: #12372a;
}
[data-testid="stSidebar"] * {
    color: white !important;
}
.metric-card {
    background: white;
    padding: 20px;
    border-radius: 12px;
    border-left: 5px solid #218c52;
    box-shadow: 0 2px 8px rgba(0,0,0,0.06);
}
.metric-title {
    color: #64748b;
    font-size: 14px;
}
.metric-value {
    color: #12372a;
    font-size: 27px;
    font-weight: bold;
}
h1, h2, h3 {
    color: #12372a;
}
</style>
""", unsafe_allow_html=True)

# ---------------- LOAD DATA ----------------
BASE = Path(__file__).parent
DATA_FILE = BASE / "data" / "agriculture_merged.csv"
CLUSTER_FILE = BASE / "results" / "agriculture-clusters.txt"

@st.cache_data
def load_data():
    df = pd.read_csv(DATA_FILE)

    numeric_cols = [
        "year", "yield_kg_per_ha", "soil_ph",
        "clay_percent", "organic_carbon",
        "rainfall_mm", "temperature",
        "area_thousand_ha"
    ]

    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    return df

df = load_data()

# ---------------- SIDEBAR ----------------
st.sidebar.title("🌱 Smart Agriculture")
st.sidebar.caption("Analytics & Intelligence System")

page = st.sidebar.radio(
    "NAVIGATION",
    [
        "Dashboard Overview",
        "Crop Analysis",
        "Weather Analysis",
        "Soil Analysis",
        "District Analysis",
        "ML Clustering"
    ]
)

st.sidebar.markdown("---")
st.sidebar.info("Agricultural Dataset: 20,000 records")

# ---------------- FILTERS ----------------
st.sidebar.subheader("Data Filters")

crops = ["All"] + sorted(df["crop"].dropna().unique().tolist())
states = ["All"] + sorted(df["state"].dropna().unique().tolist())

selected_crop = st.sidebar.selectbox("Select Crop", crops)
selected_state = st.sidebar.selectbox("Select State", states)

filtered = df.copy()

if selected_crop != "All":
    filtered = filtered[filtered["crop"] == selected_crop]

if selected_state != "All":
    filtered = filtered[filtered["state"] == selected_state]

# ---------------- HEADER ----------------
st.title("🌱 Smart Agriculture Analytics")
st.caption("Data-driven insights for better agricultural decisions")

st.markdown("---")

# ---------------- METRIC FUNCTION ----------------
def metric_card(title, value):
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-title">{title}</div>
            <div class="metric-value">{value}</div>
        </div>
        """,
        unsafe_allow_html=True
    )

# ---------------- OVERVIEW ----------------
if page == "Dashboard Overview":

    st.subheader("Agriculture Overview")

    c1, c2, c3, c4 = st.columns(4)

    with c1:
        metric_card("Total Records", f"{len(filtered):,}")

    with c2:
        metric_card("Total Crops", filtered["crop"].nunique())

    with c3:
        metric_card("States", filtered["state"].nunique())

    with c4:
        avg_yield = filtered["yield_kg_per_ha"].mean()
        metric_card(
            "Average Yield",
            f"{avg_yield:,.2f}" if pd.notna(avg_yield) else "N/A"
        )

    st.markdown("## Crop Distribution")

    crop_count = (
        filtered["crop"]
        .value_counts()
        .reset_index()
    )
    crop_count.columns = ["Crop", "Records"]

    fig = px.bar(
        crop_count,
        x="Crop",
        y="Records",
        color="Records",
        title="Agricultural Records by Crop",
        color_continuous_scale="Greens"
    )
    st.plotly_chart(fig, use_container_width=True)

    col1, col2 = st.columns(2)

    with col1:
        state_count = filtered["state"].value_counts().reset_index()
        state_count.columns = ["State", "Records"]

        fig = px.pie(
            state_count,
            names="State",
            values="Records",
            title="State-wise Distribution",
            hole=0.4
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        yearly = (
            filtered.groupby("year")["yield_kg_per_ha"]
            .mean()
            .reset_index()
        )

        fig = px.line(
            yearly,
            x="year",
            y="yield_kg_per_ha",
            markers=True,
            title="Average Crop Yield Over Years"
        )
        st.plotly_chart(fig, use_container_width=True)


# ---------------- CROP ANALYSIS ----------------
elif page == "Crop Analysis":

    st.subheader("🌾 Crop Performance Analysis")

    crop_yield = (
        filtered.groupby("crop")["yield_kg_per_ha"]
        .mean()
        .reset_index()
        .sort_values("yield_kg_per_ha", ascending=False)
    )

    fig = px.bar(
        crop_yield,
        x="crop",
        y="yield_kg_per_ha",
        color="yield_kg_per_ha",
        title="Average Yield by Crop",
        color_continuous_scale="Greens"
    )
    st.plotly_chart(fig, use_container_width=True)

    fig = px.box(
        filtered,
        x="crop",
        y="yield_kg_per_ha",
        title="Crop Yield Distribution"
    )
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Crop Statistics")
    st.dataframe(crop_yield, use_container_width=True)


# ---------------- WEATHER ANALYSIS ----------------
elif page == "Weather Analysis":

    st.subheader("🌦️ Weather Analysis")

    weather = filtered.dropna(
        subset=["rainfall_mm", "temperature"]
    )

    c1, c2 = st.columns(2)

    with c1:
        metric_card(
            "Average Rainfall",
            f"{weather['rainfall_mm'].mean():.2f} mm"
            if not weather.empty else "N/A"
        )

    with c2:
        metric_card(
            "Average Temperature",
            f"{weather['temperature'].mean():.2f} °C"
            if not weather.empty else "N/A"
        )

    fig = px.scatter(
        weather,
        x="rainfall_mm",
        y="yield_kg_per_ha",
        color="crop",
        title="Rainfall vs Crop Yield"
    )
    st.plotly_chart(fig, use_container_width=True)

    fig = px.scatter(
        weather,
        x="temperature",
        y="yield_kg_per_ha",
        color="crop",
        title="Temperature vs Crop Yield"
    )
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Weather Data Availability")

    if "has_weather_data" in filtered.columns:
        weather_status = (
            filtered["has_weather_data"]
            .astype(str)
            .value_counts()
            .reset_index()
        )
        weather_status.columns = ["Weather Data", "Records"]

        fig = px.pie(
            weather_status,
            names="Weather Data",
            values="Records",
            title="Weather Availability"
        )
        st.plotly_chart(fig, use_container_width=True)


# ---------------- SOIL ANALYSIS ----------------
elif page == "Soil Analysis":

    st.subheader("🪴 Soil Quality Analysis")

    c1, c2 = st.columns(2)

    with c1:
        metric_card(
            "Average Soil pH",
            f"{filtered['soil_ph'].mean():.2f}"
            if filtered["soil_ph"].notna().any() else "N/A"
        )

    with c2:
        metric_card(
            "Average Organic Carbon",
            f"{filtered['organic_carbon'].mean():.2f}"
            if filtered["organic_carbon"].notna().any() else "N/A"
        )

    col1, col2 = st.columns(2)

    with col1:
        fig = px.histogram(
            filtered,
            x="soil_ph",
            nbins=25,
            title="Soil pH Distribution",
            color_discrete_sequence=["#218c52"]
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        fig = px.histogram(
            filtered,
            x="organic_carbon",
            nbins=25,
            title="Organic Carbon Distribution",
            color_discrete_sequence=["#4c956c"]
        )
        st.plotly_chart(fig, use_container_width=True)

    fig = px.scatter(
        filtered,
        x="soil_ph",
        y="yield_kg_per_ha",
        color="crop",
        title="Soil pH vs Crop Yield"
    )
    st.plotly_chart(fig, use_container_width=True)


# ---------------- DISTRICT ANALYSIS ----------------
elif page == "District Analysis":

    st.subheader("📍 District-wise Agriculture Analysis")

    district_data = (
        filtered.groupby("district")
        .agg(
            Records=("crop", "count"),
            Average_Yield=("yield_kg_per_ha", "mean")
        )
        .reset_index()
        .sort_values("Average_Yield", ascending=False)
    )

    fig = px.bar(
        district_data.head(20),
        x="district",
        y="Average_Yield",
        color="Average_Yield",
        title="Top 20 Districts by Average Yield",
        color_continuous_scale="Greens"
    )
    st.plotly_chart(fig, use_container_width=True)

    fig = px.bar(
        district_data.head(20),
        x="district",
        y="Records",
        title="Agricultural Records by District"
    )
    st.plotly_chart(fig, use_container_width=True)

    st.dataframe(district_data, use_container_width=True)


# ---------------- ML CLUSTERING ----------------
elif page == "ML Clustering":

    st.subheader("🤖 Mahout K-Means Clustering")
    st.write("Clustering results generated using Apache Mahout 0.13.0.")

    cluster_data = pd.DataFrame({
        "Cluster": ["Cluster 1", "Cluster 2", "Cluster 3"],
        "Records": [2818, 6582, 10600]
    })

    cluster_data["Percentage"] = (
        cluster_data["Records"] / 20000 * 100
    )

    c1, c2, c3 = st.columns(3)

    with c1:
        metric_card("Cluster 1", "2,818")

    with c2:
        metric_card("Cluster 2", "6,582")

    with c3:
        metric_card("Cluster 3", "10,600")

    fig = px.pie(
        cluster_data,
        names="Cluster",
        values="Records",
        hole=0.45,
        title="K-Means Cluster Distribution",
        color_discrete_sequence=["#1b4332", "#52b788", "#95d5b2"]
    )
    st.plotly_chart(fig, use_container_width=True)

    fig = px.bar(
        cluster_data,
        x="Cluster",
        y="Records",
        color="Cluster",
        title="Number of Records in Each Cluster"
    )
    st.plotly_chart(fig, use_container_width=True)

    st.dataframe(cluster_data, use_container_width=True)

    st.info(
        "K-Means configuration: K = 3, Maximum Iterations = 20, "
        "Convergence Delta = 0.001."
    )

    st.caption(
        "Cluster counts are from the actual Mahout clusteredPoints output."
    )

# ---------------- FOOTER ----------------
st.markdown("---")
st.markdown(
    "<center>🌱 Smart Agriculture Analytics | "
    "Powered by Python, Streamlit, Hadoop & Apache Mahout</center>",
    unsafe_allow_html=True
)
