import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from theme import set_style

def run_task3():
    print("--- Running Task 3: Distribution Modeling & Tail Integrity ---")
    set_style()
    
    # 1. Load Data
    data_path = "data/final_dataset.parquet"
    if not os.path.exists(data_path): return
    
    df = pd.read_parquet(data_path)
    df = df[(df['parameter'] == 'pm25') & (df['zone'] == 'Industrial')]
    
    if df.empty:
        print("No industrial data for pm25 found.")
        return
        
    # Select one representative industrial station
    target_station = df['station_id'].iloc[0]
    station_data = df[df['station_id'] == target_station]['value']
    print(f"Analyzing Station ID: {target_station} (Industrial)")

    # 2. Plot 1: Optimized for Peaks (Traditional Histogram + KDE)
    plt.figure(figsize=(10, 6))
    sns.histplot(station_data, kde=True, color='blue', alpha=0.3, element="step")
    plt.title(f"Plot A (Peak Optimized): Distribution of PM2.5 at Station {target_station}")
    plt.xlabel("PM2.5 (μg/m³)")
    plt.ylabel("Frequency")
    plt.xlim(0, station_data.max()) # Standard scale
    plt.savefig("results/task3_peaks_dist.png")
    
    # 3. Plot 2: Optimized for Tails (ECDF)
    plt.figure(figsize=(10, 6))
    sns.ecdfplot(station_data, color='red', lw=2)
    plt.title(f"Plot B (Tail Optimized): ECDF of PM2.5 (Integrity Check)")
    plt.xlabel("PM2.5 (μg/m³)")
    plt.ylabel("Cumulative Probability")
    plt.axvline(200, color='black', linestyle='--', alpha=0.7, label='Extreme Hazard (200)')
    # Probability of Extreme Hazard
    prob_extreme = (station_data > 200).mean()
    plt.annotate(f'Prob(>200) = {prob_extreme:.4f}', xy=(210, 0.5), fontsize=12, color='darkred')
    plt.legend()
    plt.savefig("results/task3_tails_ecdf.png")
    
    # 4. Metrics
    p99 = np.percentile(station_data, 99)
    print(f"\n--- Statistics for Station {target_station} ---")
    print(f"99th Percentile: {p99:.2f} μg/m³")
    print(f"Probability of Extreme Hazard (>200 μg/m³): {prob_extreme:.4%}")
    
    # Save statistics
    with open("results/task3_stats.txt", "w") as f:
        f.write(f"Station: {target_station}\n")
        f.write(f"99th Percentile: {p99:.2f}\n")
        f.write(f"Extreme Hazard Prob: {prob_extreme:.4f}\n")

if __name__ == "__main__":
    run_task3()
