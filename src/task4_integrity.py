import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import seaborn as sns
from theme import set_style, sequential_palette


def run_task4():
    print("--- Running Task 4: Visual Integrity Audit ---")
    set_style()

    # ── 1. Load Data ───────────────────────────────────────────────────────────
    data_path = "data/final_dataset.parquet"
    meta_path = "data/target_locations.csv"

    if not os.path.exists(data_path):
        print(f"Error: {data_path} not found. Run data_loader first.")
        return

    df = pd.read_parquet(data_path)

    if os.path.exists(meta_path):
        loc_meta = pd.read_csv(meta_path)[['id', 'country']]
        df = df.merge(loc_meta, left_on='station_id', right_on='id', how='left')
    else:
        # Fallback: use station_id as country proxy
        df['country'] = df['station_id'].astype(str).str[:3]

    # ── 2. Decision: REJECT 3D Bar Chart ─────────────────────────────────────
    # Lie Factor = (size of effect shown in graphic) / (size of effect in data)
    # A 3D bar chart can visually inflate a bar that is twice as tall
    # into one that looks ~3-4× taller due to perspective — Lie Factor ≈ 1.5–2.0
    # which far exceeds Tufte's acceptable limit of 1.05.
    print("=" * 60)
    print("DECISION: REJECT 3D Bar Chart Proposal")
    print("=" * 60)
    print("Lie Factor = (visual effect size) / (data effect size)")
    print("A 3D bar chart typically produces a Lie Factor of 1.5–2.0")
    print("due to perspective foreshortening and occlusion.")
    print("Tufte's principle: acceptable Lie Factor ≤ 1.05")
    print("→ 3D bar charts are REJECTED on both Lie Factor AND")
    print("  Data-Ink Ratio grounds (ink spent on depth adds no data value).")
    print("=" * 60)

    # ── 3. Build region-level summary ─────────────────────────────────────────
    pm25_df = df[df['parameter'] == 'pm25'].copy()
    region_poll = (
        pm25_df.groupby('country')['value']
        .agg(mean_pm25='mean', n_readings='count')
        .reset_index()
        .dropna(subset=['country'])
    )

    # Simulate Population Density (log-normal — realistic proxy)
    # NOTE: OpenAQ does not provide population density; this is a simulated
    # proxy used to demonstrate the three-variable comparison as required.
    np.random.seed(42)
    region_poll['pop_density'] = np.random.lognormal(
        mean=6, sigma=1, size=len(region_poll)
    )

    # Pick top regions by pollution for a readable plot
    n_regions = min(16, len(region_poll))
    top_regions = region_poll.nlargest(n_regions, 'mean_pm25')

    # ── 4. Alternative: Small Multiples — Scatter (Pollution vs Pop Density) ──
    # Each facet = one country/region; x = Pop Density, y = Mean PM2.5
    # This shows ALL THREE variables: pollution (y), pop density (x), region (facet)
    facet_data = pm25_df[pm25_df['country'].isin(top_regions['country'].tolist())].copy()
    facet_data = facet_data.merge(
        top_regions[['country', 'pop_density']], on='country', how='left'
    )

    # Compute hourly aggregation per station so each point = one station reading
    hourly_agg = (
        facet_data.groupby(['country', 'station_id', 'pop_density'])['value']
        .mean()
        .reset_index()
        .rename(columns={'value': 'mean_pm25'})
    )

    # ── 4a. Small Multiples grid (faceted scatter) ────────────────────────────
    n_cols  = 4
    n_rows  = int(np.ceil(len(top_regions) / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, n_rows * 4))
    axes = axes.flatten()

    viridis = cm.get_cmap('viridis')

    for i, (_, reg_row) in enumerate(top_regions.iterrows()):
        ax      = axes[i]
        country = reg_row['country']
        sub     = hourly_agg[hourly_agg['country'] == country]

        if sub.empty:
            ax.set_visible(False)
            continue

        # Colour each dot by PM2.5 level using Sequential Viridis
        norm   = plt.Normalize(sub['mean_pm25'].min(), sub['mean_pm25'].max())
        colors = viridis(norm(sub['mean_pm25'].values))

        ax.scatter(
            sub['pop_density'], sub['mean_pm25'],
            c=colors, s=60, alpha=0.8, edgecolors='none'
        )
        ax.axhline(35, color='red', linestyle='--', linewidth=1,
                   label='Health Threshold (35 μg/m³)')
        ax.set_title(f"{country}", fontsize=11, fontweight='bold')
        ax.set_xlabel("Pop. Density (simulated)", fontsize=8)
        ax.set_ylabel("Mean PM2.5 (μg/m³)", fontsize=8)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    # Hide any unused subplots
    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    # Shared colorbar
    sm = cm.ScalarMappable(cmap='viridis')
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=axes, shrink=0.6, pad=0.02)
    cbar.set_label('PM2.5 Level (μg/m³)', fontsize=10)

    fig.suptitle(
        'Task 4: Small Multiples — Pollution vs Population Density by Region\n'
        '(Sequential Viridis scale · No 3D distortion · High Data-Ink Ratio)',
        fontsize=14, y=1.01
    )
    plt.savefig("results/task4_integrity_multiples.png", bbox_inches='tight')
    print("Small Multiples scatter saved → results/task4_integrity_multiples.png")

    # ── 5. Color Scale Justification ─────────────────────────────────────────
    print("\n--- Color Scale Justification ---")
    print("Selected: Sequential (Viridis)")
    print("Reason 1: Human luminance perception is monotonic — darker = more.")
    print("Reason 2: Rainbow/Jet scales introduce false boundaries (pseudo-bands)")
    print("          where equal data steps appear unequal visually.")
    print("Reason 3: Viridis is perceptually uniform (Lab-space designed),")
    print("          remains readable in greyscale, and is accessible to")
    print("          colour-blind viewers (deuteranopia / protanopia safe).")
    print("Reason 4: Sequential scale is appropriate here because PM2.5")
    print("          is a single ordered quantity (low → high risk).")


if __name__ == "__main__":
    run_task4()
