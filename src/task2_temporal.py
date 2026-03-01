import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from theme import set_style, sequential_palette


def run_task2():
    print("--- Running Task 2: High-Density Temporal Analysis ---")
    set_style()

    # ── 1. Load Data ───────────────────────────────────────────────────────────
    data_path = "data/final_dataset.parquet"
    if not os.path.exists(data_path):
        print("Error: Dataset not found. Run data_loader first.")
        return

    df = pd.read_parquet(data_path)
    df = df[df['parameter'] == 'pm25'].copy()
    df['datetime'] = pd.to_datetime(df['datetime'])

    # ── 2. Heatmap: Stations × Time ────────────────────────────────────────────
    # Standard solution for 100-line overplotting — replace 100 line-charts
    # with a single Stations × Time heatmap (each row = 1 sensor).
    print("Aggregating for Heatmap (Station × Day)...")

    top_stations = df['station_id'].unique()[:100]
    sub_df       = df[df['station_id'].isin(top_stations)].copy()
    sub_df['day'] = sub_df['datetime'].dt.floor('D')

    pivot_df = sub_df.pivot_table(
        index='station_id', columns='day', values='value', aggfunc='mean'
    )

    fig, ax = plt.subplots(figsize=(18, 10))
    im = ax.imshow(
        pivot_df.values,
        aspect='auto',
        cmap='YlOrRd',
        interpolation='nearest'
    )
    cbar = fig.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label('Daily Mean PM2.5 (μg/m³)', fontsize=11)

    # Mark the Health Threshold on the colorbar
    cbar.ax.axhline(
        y=(35 - pivot_df.values.min()) /
          (pivot_df.values.max() - pivot_df.values.min() + 1e-9),
        color='red', linewidth=2, linestyle='--'
    )
    cbar.ax.text(1.5, 35, '35 μg/m³\nThreshold', color='red', fontsize=8, va='center')

    # X-axis: monthly tick labels
    n_days = pivot_df.shape[1]
    month_ticks = np.linspace(0, n_days - 1, 12, dtype=int)
    month_labels = [f"Month {i+1}" for i in range(12)]
    ax.set_xticks(month_ticks)
    ax.set_xticklabels(month_labels, rotation=45, ha='right', fontsize=9)
    ax.set_yticks([])
    ax.set_xlabel("Time (Monthly bins, 2025)", fontsize=11)
    ax.set_ylabel(f"Stations (n={len(pivot_df)})", fontsize=11)
    ax.set_title(
        "Task 2: PM2.5 Temporal Heatmap Across 100 Sensors (2025)\n"
        "Each row = one sensor station | Colour = daily mean PM2.5",
        fontsize=13
    )

    plt.savefig("results/task2_temporal_heatmap.png", bbox_inches='tight')
    print("Heatmap saved → results/task2_temporal_heatmap.png")

    # ── 3. Diurnal Signature (24-hour cycle) ───────────────────────────────────
    df['hour_only'] = df['datetime'].dt.hour
    diurnal         = df.groupby('hour_only')['value'].mean()
    diurnal_peak    = diurnal.idxmax()
    diurnal_std     = diurnal.std()

    fig2, ax2 = plt.subplots(figsize=(9, 5))
    ax2.plot(diurnal.index, diurnal.values, color='teal', lw=2.5)
    ax2.fill_between(diurnal.index, diurnal.values, alpha=0.15, color='teal')
    ax2.axhline(35, color='red', linestyle='--', lw=1.5, label='Health Threshold')
    ax2.axvline(diurnal_peak, color='orange', linestyle=':', lw=1.5,
                label=f'Peak Hour: {diurnal_peak}:00')
    ax2.set_title("Periodic Signature: 24-Hour Diurnal Traffic Cycle", fontsize=13)
    ax2.set_xlabel("Hour of Day (UTC)", fontsize=11)
    ax2.set_ylabel("Mean PM2.5 (μg/m³)", fontsize=11)
    ax2.legend()
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    fig2.savefig("results/task2_diurnal_signature.png", bbox_inches='tight')

    # ── 4. Seasonal Signature (monthly) ───────────────────────────────────────
    df['month']  = df['datetime'].dt.month
    seasonal     = df.groupby('month')['value'].mean()
    seasonal_std = df.groupby('month')['value'].std()
    seasonal_peak = seasonal.idxmax()
    seasonal_amplitude = seasonal.max() - seasonal.min()

    month_names = ['Jan','Feb','Mar','Apr','May','Jun',
                   'Jul','Aug','Sep','Oct','Nov','Dec']

    fig3, ax3 = plt.subplots(figsize=(9, 5))
    bars = ax3.bar(
        seasonal.index, seasonal.values,
        color=[plt.cm.YlOrRd(v / seasonal.max()) for v in seasonal.values],
        alpha=0.85, yerr=seasonal_std.values, capsize=3, ecolor='gray'
    )
    ax3.axhline(35, color='red', linestyle='--', lw=1.5, label='Health Threshold')
    ax3.set_xticks(range(1, 13))
    ax3.set_xticklabels(month_names, fontsize=9)
    ax3.set_title("Periodic Signature: Monthly Seasonal Shifts", fontsize=13)
    ax3.set_xlabel("Month (2025)", fontsize=11)
    ax3.set_ylabel("Mean PM2.5 (μg/m³)", fontsize=11)
    ax3.legend()
    ax3.spines['top'].set_visible(False)
    ax3.spines['right'].set_visible(False)
    fig3.savefig("results/task2_seasonal_signature.png", bbox_inches='tight')

    # ── 5. Periodic Signature Conclusion ──────────────────────────────────────
    print()
    print("=" * 60)
    print("PERIODIC SIGNATURE ANALYSIS CONCLUSION:")
    print()
    print(f"  Diurnal cycle peak: Hour {diurnal_peak}:00 UTC")
    print(f"  Diurnal amplitude (std): {diurnal_std:.2f} μg/m³")
    print()
    print(f"  Seasonal cycle peak: Month {seasonal_peak} "
          f"({month_names[seasonal_peak-1]})")
    print(f"  Seasonal amplitude (max−min): {seasonal_amplitude:.2f} μg/m³")
    print()
    if diurnal_std > seasonal_amplitude / 3:
        conclusion = "DAILY (24-hour traffic cycle)"
        reason = (
            f"The diurnal std ({diurnal_std:.2f} μg/m³) is large relative to "
            f"seasonal amplitude ({seasonal_amplitude:.2f} μg/m³), indicating "
            "within-day traffic patterns dominate pollution spikes."
        )
    else:
        conclusion = "MONTHLY (30-day seasonal shift)"
        reason = (
            f"The seasonal amplitude ({seasonal_amplitude:.2f} μg/m³) dominates "
            f"over the diurnal std ({diurnal_std:.2f} μg/m³), indicating "
            "seasonal factors (temperature inversions, heating) drive violations."
        )

    print(f"  PRIMARY DRIVER: {conclusion}")
    print(f"  Reasoning: {reason}")
    print("=" * 60)

    print("All Task 2 plots saved to results/")


if __name__ == "__main__":
    run_task2()
