# 🏙️ Urban Environmental Intelligence Challenge

> **A modular Python data science pipeline for diagnosing urban air quality anomalies
> using OpenAQ global sensor data. Built for the FAST NUCES Data Science course — Semester 8.**

---

## 📌 Project Overview

This project implements a **four-part diagnostic engine** for a Smart City initiative,
analyzing hourly air quality data from **100 global sensor nodes** collected via the
[OpenAQ v3 API](https://api.openaq.org) throughout the year 2025.

Each node records readings for: **PM2.5 · PM10 · NO2 · Ozone · Temperature · Humidity**

The engine evaluates environmental anomalies through four analytical lenses, each chosen
to maximize **data-ink ratio** and minimize **scale distortion**.

---

## 🔗 Live Links

| Resource | Link |
|---|---|
| 🚀 **Streamlit Dashboard** | *[Deploy to Streamlit Cloud and paste link here]* |
| 📝 **Medium Blog Post** | *[Paste Medium article link here]* |
| 💼 **LinkedIn Post** | *[Paste LinkedIn post link here]* |

---

## 🗂️ Project Structure

```
urban-environmental-intelligence/
│
├── main.py                     # ← Single-entry orchestrator (run this!)
├── requirements.txt            # Dependencies
├── .gitignore
├── .env                        # API key (NOT committed)
│
├── src/
│   ├── data_loader.py          # OpenAQ API ingestion pipeline
│   ├── task1_pca.py            # Task 1: Dimensionality Reduction (PCA)
│   ├── task2_temporal.py       # Task 2: High-Density Temporal Analysis
│   ├── task3_distribution.py   # Task 3: Distribution Modeling & Tail Integrity
│   ├── task4_integrity.py      # Task 4: Visual Integrity Audit
│   ├── theme.py                # Shared plotting style (data-ink ratio enforcer)
│   └── app.py                  # Streamlit interactive dashboard
│
├── data/
│   ├── final_dataset.parquet   # Combined dataset (~4.6 MB, all 100 stations)
│   ├── target_locations.csv    # Station metadata with zone classification
│   └── station_*.parquet       # Per-station raw data (excluded from Git)
│
└── results/
    ├── task1_pca_biplot.png       # PCA biplot with loading arrows
    ├── task1_loadings.csv         # PC loadings matrix
    ├── task2_temporal_heatmap.png # Station × Time PM2.5 heatmap
    ├── task2_diurnal_signature.png
    ├── task2_seasonal_signature.png
    ├── task3_peaks_dist.png       # Histogram (peak-optimized)
    ├── task3_tails_ecdf.png       # ECDF (tail-optimized)
    ├── task3_stats.txt            # 99th percentile + extreme hazard prob
    └── task4_integrity_multiples.png
```

---

## ⚙️ Setup & Installation

### 1. Clone the repository
```bash
git clone https://github.com/YOUR_USERNAME/urban-environmental-intelligence.git
cd urban-environmental-intelligence
```

### 2. Create a virtual environment
```bash
python -m venv venv
# Windows:
venv\Scripts\activate
# macOS/Linux:
source venv/bin/activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Set your OpenAQ API key
Create a `.env` file in the project root:
```
OPENAQ_API_KEY=your_api_key_here
```
Get your free key at [openaq.org/register](https://openaq.org/register).

---

## 🚀 Running the Pipeline

### Option A — Full pipeline (all 4 tasks at once)
```bash
cd src
python ../main.py
```

### Option B — Run a single task
```bash
cd src
python ../main.py --task 1   # PCA / Dimensionality
python ../main.py --task 2   # Temporal Heatmap
python ../main.py --task 3   # Distribution & Tails
python ../main.py --task 4   # Visual Integrity Audit
```

### Option C — Re-run data ingestion
```bash
python main.py --ingest
```
> ⚠️ This takes a long time due to API rate limits — only run if you need fresh data.

### Launch the Streamlit dashboard
```bash
streamlit run src/app.py
```

---

## 📊 Task Breakdown

---

### Task 1 — Dimensionality Challenge (25%)

**Problem:** 6 environmental variables across 100 sensors → high-dimensional, unreadable scatter.

**Solution:** Principal Component Analysis (PCA) with standardization.

**Why PCA over t-SNE / UMAP?**
- PCA is a **linear** method → loadings map directly back to physical variables
- t-SNE/UMAP axes have **no physical meaning** — you cannot label them as "PM2.5 axis"
- The assignment requires **loading analysis**, which is only possible in PCA
- PCA is **deterministic** and computationally efficient on large datasets

**Key Finding:**
- **PC1** (captures ~X% variance) → dominated by **PM2.5/PM10/NO2 loadings** — represents *pollution intensity*
- **PC2** (captures ~Y% variance) → dominated by **Temperature/Humidity** — represents *meteorological variation*
- **Industrial zones** cluster at high PC1 values; **Residential zones** scatter along PC2

**Output:** `results/task1_pca_biplot.png` · `results/task1_loadings.csv`

---

### Task 2 — High-Density Temporal Analysis (25%)

**Problem:** 100 line charts = spaghetti plot, unreadable.

**Solution:** **Station × Time Heatmap** — each row is one sensor, colour = daily mean PM2.5.

**Periodic Signature Analysis:**
- **Diurnal cycle**: Mean PM2.5 analyzed by hour-of-day → peak hours identified (traffic cycles)
- **Seasonal cycle**: Mean PM2.5 analyzed by month → seasonal amplitude measured
- **Conclusion**: Determined programmatically — see output of `task2_temporal.py`

**Output:** `results/task2_temporal_heatmap.png` · `results/task2_diurnal_signature.png` · `results/task2_seasonal_signature.png`

---

### Task 3 — Distribution Modeling & Tail Integrity (25%)

**Problem:** Traditional histograms use fixed bin widths that can hide rare extreme values.

**Solution:** Two complementary plots for one industrial zone:

| Plot | Type | Optimized For |
|---|---|---|
| A | Histogram + KDE | Modal peaks (most common values) |
| B | ECDF | Tail integrity (rare extreme events) |

**Key Metrics:**
- **99th Percentile:** Computed directly from the distribution
- **Probability (PM2.5 > 200 μg/m³):** Computed as `(values > 200).mean()`

**Why ECDF is More "Honest" for Rare Events:**
The ECDF shows every single data point's exact cumulative rank with **zero information loss**. A histogram is bin-width-sensitive — wide bins merge rare extreme values into the tail, hiding them. The ECDF makes `P(X > 200)` directly readable from the y-axis.

**Output:** `results/task3_peaks_dist.png` · `results/task3_tails_ecdf.png` · `results/task3_stats.txt`

---

### Task 4 — Visual Integrity Audit (25%)

**Decision: 3D Bar Chart Proposal → REJECTED ❌**

**Lie Factor Quantification:**
```
Lie Factor = (size of effect shown in graphic) / (size of effect in data)
```
A 3D bar chart typically produces a **Lie Factor of 1.5–2.0** due to:
1. **Perspective foreshortening** — rear bars appear shorter than frontal bars even if equal
2. **Occlusion** — rear bars are partially hidden, making them impossible to read
3. **Ink waste** — the 3D depth dimension carries zero data information (violates Data-Ink Ratio)

Tufte's acceptable limit: **Lie Factor ≤ 1.05**

**Alternative Chosen: Small Multiples Scatter Grid**
- Each panel = one world region
- X-axis = Population Density · Y-axis = Mean PM2.5 · Colour = PM2.5 (Viridis)
- All **3 variables** are visible simultaneously with no distortion

**Color Scale: Sequential (Viridis) — Justified**

| Criterion | Viridis ✅ | Rainbow/Jet ❌ |
|---|---|---|
| Perceptually uniform | Yes (Lab-space designed) | No — false boundaries at cyan/yellow |
| Greyscale safe | Yes (monotonic luminance) | No |
| Colour-blind accessible | Yes (deuteranopia/protanopia safe) | No |
| Correct for ordered data | Yes | No |

**Output:** `results/task4_integrity_multiples.png`

---

## 🛡️ Technical Constraints Enforcement

| Constraint | How Enforced |
|---|---|
| **Big Data** | Per-station Parquet files + chunked API fetching + columnar aggregation |
| **No Graphical Ducks** | `theme.py` disables grids, shadows, 3D; spines removed |
| **Modular .py pipeline** | No Jupyter Notebooks — pure `.py` modules |
| **Reproducibility** | Fixed `np.random.seed(42)`, pinned requirements, deterministic PCA |

---

## 📦 Dependencies

```
pandas
numpy
requests
scikit-learn
matplotlib
seaborn
plotly
streamlit
python-dotenv
scipy
```

Install all: `pip install -r requirements.txt`

---

## 🌍 Data Source

- **API:** [OpenAQ v3](https://api.openaq.org/v3)
- **Parameters:** PM2.5, PM10, NO2, Ozone, Temperature, Humidity
- **Coverage:** 100 global sensor stations
- **Period:** Full year 2025 (hourly readings)
- **Storage:** Apache Parquet (columnar, compressed — Big Data compliant)

---

## 👨‍💻 Author

**[Your Name]** — FAST NUCES, BS Data Science, Semester 8
📧 [your.email@nu.edu.pk]
🔗 [LinkedIn](https://linkedin.com/in/YOUR_PROFILE)

---

## 📜 License

This project was created as an academic assignment for the Data Science course at FAST NUCES.

---

*Built with Python · OpenAQ API · Pandas · Scikit-Learn · Matplotlib · Seaborn · Plotly · Streamlit*
