import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import seaborn as sns
import os
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# ── Page Config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Urban Environmental Intelligence",
    page_icon="🏙️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Custom CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');
    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
    .main { background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%); }
    .block-container { padding-top: 2rem; }
    .stMetric {
        background: rgba(255,255,255,0.05);
        border: 1px solid rgba(255,255,255,0.1);
        border-radius: 12px;
        padding: 16px;
        backdrop-filter: blur(10px);
    }
    h1 { color: #38bdf8 !important; font-weight: 700 !important; }
    h2 { color: #7dd3fc !important; }
    h3 { color: #bae6fd !important; }
    .stSidebar { background: rgba(15, 23, 42, 0.95) !important; }
    </style>
""", unsafe_allow_html=True)


# ── Data Loader ────────────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    parquet_path = "data/final_dataset.parquet"
    meta_path    = "data/target_locations.csv"

    if not os.path.exists(parquet_path):
        return None, None

    df       = pd.read_parquet(parquet_path)
    loc_meta = pd.read_csv(meta_path) if os.path.exists(meta_path) else None
    return df, loc_meta


# ── Task 1 ─────────────────────────────────────────────────────────────────────
def show_task1(df):
    st.header("🔬 Task 1: Dimensionality Reduction (PCA)")
    st.write(
        "Projects 6-dimensional air quality data into 2 principal components "
        "to identify clustering between Industrial and Residential zones."
    )

    with st.expander("📖 Method Justification: Why PCA over t-SNE / UMAP?", expanded=True):
        st.markdown("""
        **PCA was chosen** because:
        1. **Interpretable axes** — loadings map back to physical variables (e.g., PM2.5, NO2).
           t-SNE/UMAP axes have no physical meaning.
        2. **Loading analysis** — the assignment requires loading analysis, which is only
           possible with a linear method.
        3. **Scalability** — PCA handles large sensor datasets efficiently (O(n·p²)).
        4. **Reproducibility** — PCA is deterministic; t-SNE is stochastic.
        """)

    # ── Preprocessing ──────────────────────────────────────────────────────────
    df_pca    = df.copy()
    df_pca['datetime'] = pd.to_datetime(df_pca['datetime']).dt.floor('h')
    wide = df_pca.pivot_table(
        index=['station_id', 'datetime', 'zone'],
        columns='parameter', values='value', aggfunc='mean'
    ).reset_index()

    features = ['pm25', 'pm10', 'no2', 'o3', 'temp', 'humidity']
    features = [f for f in features if f in wide.columns]

    for f in features:
        wide[f] = wide.groupby('station_id')[f].transform(lambda x: x.fillna(x.mean()))
    wide = wide.fillna(wide.mean(numeric_only=True)).dropna(subset=features)

    # ── PCA ────────────────────────────────────────────────────────────────────
    scaler   = StandardScaler()
    x_scaled = scaler.fit_transform(wide[features])
    pca      = PCA(n_components=2)
    pcs      = pca.fit_transform(x_scaled)

    pca_df         = pd.DataFrame(pcs, columns=['PC1', 'PC2'])
    pca_df['Zone'] = wide['zone'].values

    ev1 = pca.explained_variance_ratio_[0] * 100
    ev2 = pca.explained_variance_ratio_[1] * 100

    col1, col2, col3 = st.columns(3)
    col1.metric("PC1 Variance Explained", f"{ev1:.1f}%")
    col2.metric("PC2 Variance Explained", f"{ev2:.1f}%")
    col3.metric("Total Captured", f"{ev1+ev2:.1f}%")

    # ── Scatter Plot ───────────────────────────────────────────────────────────
    fig = px.scatter(
        pca_df.sample(min(5000, len(pca_df))),
        x='PC1', y='PC2', color='Zone',
        color_discrete_map={'Industrial': '#ef4444', 'Residential': '#3b82f6'},
        template='plotly_dark',
        opacity=0.4,
        title=f"PCA Clusters: Industrial vs Residential "
              f"(PC1={ev1:.1f}%, PC2={ev2:.1f}%)",
        labels={
            'PC1': f'PC1 — Pollution Intensity ({ev1:.1f}%)',
            'PC2': f'PC2 — Meteorological Variation ({ev2:.1f}%)'
        }
    )
    fig.update_traces(marker=dict(size=4))
    st.plotly_chart(fig, use_container_width=True)

    # ── Loading Analysis ───────────────────────────────────────────────────────
    loadings_df = pd.DataFrame(
        pca.components_.T, columns=['PC1_loading', 'PC2_loading'], index=features
    )
    st.subheader("📊 PCA Loadings Matrix")
    st.dataframe(loadings_df.round(3), use_container_width=True)

    top_pc1 = loadings_df['PC1_loading'].abs().idxmax()
    top_pc2 = loadings_df['PC2_loading'].abs().idxmax()
    st.info(
        f"**PC1** is most influenced by **{top_pc1.upper()}** — represents pollution "
        f"intensity (Industrial zones shift right).\n\n"
        f"**PC2** is most influenced by **{top_pc2.upper()}** — represents "
        f"meteorological variation (seasonal/geographic spread)."
    )


# ── Task 2 ─────────────────────────────────────────────────────────────────────
def show_task2(df):
    st.header("🌡️ Task 2: High-Density Temporal Analysis")
    st.write(
        "A **Station × Time heatmap** replaces 100 overlapping line charts, "
        "enabling simultaneous comparison of all sensors without overplotting."
    )

    pm25          = df[df['parameter'] == 'pm25'].copy()
    pm25['datetime'] = pd.to_datetime(pm25['datetime']).dt.floor('D')
    daily         = pm25.groupby(['station_id', 'datetime'])['value'].mean().reset_index()
    pivot         = daily.pivot(index='station_id', columns='datetime', values='value')

    fig = px.imshow(
        pivot,
        color_continuous_scale='YlOrRd',
        labels=dict(x="Date", y="Station", color="PM2.5 (μg/m³)"),
        title="Global PM2.5 Temporal Heatmap — 100 Sensors (2025)",
        template='plotly_dark',
        aspect='auto'
    )
    fig.update_layout(coloraxis_colorbar=dict(title="PM2.5 μg/m³"))
    st.plotly_chart(fig, use_container_width=True)

    # ── Signature Plots ────────────────────────────────────────────────────────
    col1, col2 = st.columns(2)

    pm25_full         = df[df['parameter'] == 'pm25'].copy()
    pm25_full['datetime'] = pd.to_datetime(pm25_full['datetime'])
    pm25_full['hour'] = pm25_full['datetime'].dt.hour
    pm25_full['month'] = pm25_full['datetime'].dt.month
    diurnal  = pm25_full.groupby('hour')['value'].mean().reset_index()
    seasonal = pm25_full.groupby('month')['value'].mean().reset_index()

    with col1:
        st.subheader("⏰ 24-Hour Diurnal Cycle")
        fig1 = px.line(
            diurnal, x='hour', y='value',
            template='plotly_dark', color_discrete_sequence=['#06b6d4'],
            labels={'hour': 'Hour of Day', 'value': 'Mean PM2.5 (μg/m³)'}
        )
        fig1.add_hline(y=35, line_dash="dash", line_color="red",
                       annotation_text="Health Threshold")
        st.plotly_chart(fig1, use_container_width=True)

    with col2:
        st.subheader("📅 Monthly Seasonal Pattern")
        month_names = ['Jan','Feb','Mar','Apr','May','Jun',
                       'Jul','Aug','Sep','Oct','Nov','Dec']
        seasonal['month_name'] = seasonal['month'].apply(lambda x: month_names[x-1])
        fig2 = px.bar(
            seasonal, x='month_name', y='value',
            template='plotly_dark', color='value',
            color_continuous_scale='YlOrRd',
            labels={'month_name': 'Month', 'value': 'Mean PM2.5 (μg/m³)'}
        )
        fig2.add_hline(y=35, line_dash="dash", line_color="red")
        st.plotly_chart(fig2, use_container_width=True)

    diurnal_std   = diurnal['value'].std()
    seasonal_amp  = seasonal['value'].max() - seasonal['value'].min()
    peak_hour     = diurnal.loc[diurnal['value'].idxmax(), 'hour']
    peak_month    = seasonal.loc[seasonal['value'].idxmax(), 'month']

    if diurnal_std > seasonal_amp / 3:
        driver = "**Daily (24-hour traffic cycle)** dominates"
    else:
        driver = "**Monthly (seasonal shifts)** dominate"

    st.success(
        f"**Periodic Signature Conclusion:** {driver}. "
        f"Diurnal peak at Hour **{peak_hour}:00**, Seasonal peak in "
        f"Month **{month_names[peak_month-1]}**."
    )


# ── Task 3 ─────────────────────────────────────────────────────────────────────
def show_task3(df):
    st.header("📈 Task 3: Distribution Modeling & Tail Integrity")

    industrial = df[(df['parameter'] == 'pm25') & (df['zone'] == 'Industrial')]
    if industrial.empty:
        st.warning("No Industrial PM2.5 data found.")
        return

    station_options = industrial['station_id'].unique()[:15]
    station         = st.selectbox("Select Industrial Station", station_options)
    data            = industrial[industrial['station_id'] == station]['value'].dropna()

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("📊 Plot A — Peak Optimized (Histogram + KDE)")
        fig = px.histogram(
            data, nbins=60,
            template='plotly_dark',
            color_discrete_sequence=['#3b82f6'],
            title="Focus on Modal Peaks",
            labels={'value': 'PM2.5 (μg/m³)'}
        )
        st.plotly_chart(fig, use_container_width=True)
        st.caption("Best for identifying the most common (modal) pollution level.")

    with col2:
        st.subheader("📉 Plot B — Tail Optimized (ECDF)")
        fig2 = px.ecdf(
            data,
            template='plotly_dark',
            color_discrete_sequence=['#ef4444'],
            title="Focus on Extreme Hazard Tails",
            labels={'value': 'PM2.5 (μg/m³)'}
        )
        fig2.add_vline(x=200, line_dash="dash", line_color="orange",
                       annotation_text="Extreme Hazard (200 μg/m³)")
        fig2.add_vline(x=35, line_dash="dot", line_color="yellow",
                       annotation_text="Health Threshold (35)")
        st.plotly_chart(fig2, use_container_width=True)
        st.caption("Best for reading the probability of rare extreme events.")

    p99          = np.percentile(data, 99)
    prob_extreme = (data > 200).mean()

    col3, col4, col5 = st.columns(3)
    col3.metric("99th Percentile", f"{p99:.2f} μg/m³")
    col4.metric("Prob(PM2.5 > 200)", f"{prob_extreme:.4%}")
    col5.metric("Total Readings", f"{len(data):,}")

    st.info(
        "**Technical Justification:** The ECDF is the more *honest* depiction "
        "of rare hazardous events. A histogram is sensitive to bin width choice — "
        "wide bins can completely hide the extreme tail. The ECDF shows *every* "
        "data point's exact cumulative rank with zero information loss, making "
        "the probability of PM2.5 > 200 μg/m³ directly readable from the y-axis."
    )


# ── Task 4 ─────────────────────────────────────────────────────────────────────
def show_task4(df):
    st.header("🎨 Task 4: Visual Integrity Audit")

    st.error(
        "**DECISION: 3D Bar Chart Proposal REJECTED** ❌\n\n"
        "**Lie Factor Formula:** `Lie Factor = (size of effect in graphic) / "
        "(size of effect in data)`\n\n"
        "3D bar charts produce a Lie Factor of **1.5–2.0** due to perspective "
        "foreshortening — a bar 2× taller appears 3–4× larger. "
        "Tufte's acceptable limit is ≤ 1.05. Additionally, 3D introduces "
        "occlusion (rear bars hidden) and wastes ink on depth with zero data value."
    )

    st.success(
        "**Alternative: Small Multiples Scatter Grid** ✅\n\n"
        "Each facet shows one region. X-axis = Population Density, "
        "Y-axis = Mean PM2.5. Colour = PM2.5 via Viridis (sequential, "
        "perceptually uniform). All 3 variables visible simultaneously."
    )

    if os.path.exists("results/task4_integrity_multiples.png"):
        st.image(
            "results/task4_integrity_multiples.png",
            caption="Small Multiples: Pollution × Population Density × Region "
                    "(Sequential Viridis — no perspective distortion)"
        )
    else:
        st.warning("Run `python src/task4_integrity.py` to generate this plot.")

    with st.expander("🎨 Color Scale Justification: Sequential vs Rainbow"):
        st.markdown("""
        **Selected: Sequential (Viridis)**

        | Criterion | Viridis (Sequential) ✅ | Rainbow/Jet ❌ |
        |---|---|---|
        | Perceptual uniformity | Equal data steps = equal perceived steps | False gradients at cyan/yellow boundaries |
        | Greyscale readable | Yes (monotonic luminance) | No |
        | Colour-blind safe | Yes (deuteranopia/protanopia safe) | No |
        | Appropriate for | Single ordered quantity (PM2.5 low→high risk) | No use case in scientific visualization |

        > **Key principle**: Human luminance perception is monotonic — we judge
        > magnitude by brightness. A sequential scale maps data magnitude directly
        > to brightness, eliminating false visual boundaries.
        """)


# ── Main App ───────────────────────────────────────────────────────────────────
def main():
    st.title("🏙️ Urban Environmental Intelligence Dashboard")
    st.subheader("Smart City Diagnostic Engine — OpenAQ 2025 | 100 Global Sensor Nodes")

    df, locs = load_data()

    if df is None:
        st.error(
            "⚠️ Data files not found at `data/final_dataset.parquet`. "
            "Please run `python main.py` to execute the full ingestion pipeline first."
        )
        return

    menu = [
        "📊 Project Overview",
        "🔬 Task 1: Dimensionality",
        "🌡️ Task 2: Temporal",
        "📈 Task 3: Distribution",
        "🎨 Task 4: Integrity Audit"
    ]
    choice = st.sidebar.radio("Navigation", menu)

    if "Overview" in choice:
        st.write("### Smart City Air Quality Initiative — 2025")
        st.write(
            "This dashboard provides a four-part diagnostic analysis of global "
            "air quality data collected from 100 OpenAQ sensor nodes throughout 2025."
        )
        cols = st.columns(4)
        cols[0].metric("Total Records",     f"{len(df):,}")
        cols[1].metric("Stations Monitored", f"{df['station_id'].nunique()}")
        cols[2].metric("Mean PM2.5 (μg/m³)", f"{df[df['parameter']=='pm25']['value'].mean():.2f}")
        cols[3].metric("Parameters Tracked", f"{df['parameter'].nunique()}")

        st.divider()
        st.subheader("Data Distribution by Zone")
        zone_dist = df.groupby('zone')['value'].count().reset_index()
        fig = px.pie(zone_dist, values='value', names='zone',
                     template='plotly_dark', hole=0.4,
                     color_discrete_sequence=['#ef4444', '#3b82f6'])
        st.plotly_chart(fig, use_container_width=True)

    elif "Task 1" in choice:
        show_task1(df)
    elif "Task 2" in choice:
        show_task2(df)
    elif "Task 3" in choice:
        show_task3(df)
    elif "Task 4" in choice:
        show_task4(df)


if __name__ == "__main__":
    main()
