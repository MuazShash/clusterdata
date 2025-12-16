#!/usr/bin/env python3
"""
Fairness Analysis Script for GPU Cluster Scheduler

This script analyzes fairness metrics across users and jobs from CSV results.
It generates:
1. Distribution of fairness across jobs for each user (combined plot)
2. Distribution of average fairness per user
3. Correlation analysis for high sharing incentive jobs (>10)
"""

import os
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Set style for better-looking plots
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (14, 6)

# Paths
RESULTS_DIR = "/home/muaz/clusterdata/cluster-trace-gpu-v2020/results_sjf_oracle"
TRACE_FILE = "/home/muaz/clusterdata/cluster-trace-gpu-v2020/simulator/traces/pai/pai_job_duration_estimate_100K.csv"
OUTPUT_DIR = "/home/muaz/clusterdata/cluster-trace-gpu-v2020/fairness_analysis_output"

# Create output directory
os.makedirs(OUTPUT_DIR, exist_ok=True)

def load_result_csv(filepath):
    """
    Load a result CSV file with columns: job_id, value1, value2, sharing_incentive
    Returns a DataFrame with the user ID extracted from filename.
    """
    df = pd.read_csv(filepath, header=None, on_bad_lines='skip')
    df.drop(columns=[0, 1, 2], inplace=True)  # drop value1, value2
    return df

def load_trace_file(filepath):
    """Load the original trace file with job metadata."""
    df = pd.read_csv(filepath)
    return df

def empirical_cdf(data):
    data = np.asarray(data)
    data = data[~np.isnan(data)]  # drop NaNs if any
    x = np.sort(data)
    y = np.arange(1, len(x) + 1) / len(x)
    return x, y

def analyze_fairness_distributions():
    """
    1) Create a CDF plot showing the distribution of fairness (sharing_incentive) 
       for each user overlayed together.
    """
    print("Analyzing fairness distributions across users...")
    
    # Load all result CSVs
    result_files = glob.glob(os.path.join(RESULTS_DIR, "*.csv"))
    all_data = []
    
    for filepath in result_files:
        try:
            df = load_result_csv(filepath)
            all_data.append(df)
        except Exception as e:
            print(f"Error loading {filepath}: {e}")
    
    # Create figure for CDF plot
    plt.subplots(figsize=(14, 8))
    average_fairness = []
    for df in all_data:
        values = df[3].values
        average_fairness.append(np.mean(values[:-1]))
    
    x, y = empirical_cdf(average_fairness)
    plt.step(x, y, where="post", alpha=0.8, color='blue')

    plt.xlabel("Sharing Incentives")
    plt.xscale('log')
    plt.xlim(1, 100) 
    plt.ylabel("CDF")
    plt.title("Average Sharing Incentives Under Fifo Allocation Across All Users")
    plt.grid(True, which="both")
    plt.legend()
    plt.tight_layout()
    plt.show()

def analyze_average_fairness(combined_df):
    """
    2) Create a plot showing the distribution of average fairness per user.
    """
    print("Analyzing average fairness per user...")
    
    # Calculate average fairness per user
    user_avg_fairness = combined_df.groupby('user')['sharing_incentive'].agg(['mean', 'std', 'count']).reset_index()
    user_avg_fairness.columns = ['user', 'avg_sharing_incentive', 'std', 'job_count']
    user_avg_fairness = user_avg_fairness.sort_values('avg_sharing_incentive', ascending=False)
    
    # Create figure
    fig, axes = plt.subplots(2, 1, figsize=(14, 10))
    
    # Plot 1: Bar plot of average fairness
    ax1 = axes[0]
    colors = ['red' if x > 10 else 'orange' if x > 1 else 'green' for x in user_avg_fairness['avg_sharing_incentive']]
    ax1.barh(range(len(user_avg_fairness)), user_avg_fairness['avg_sharing_incentive'], color=colors, alpha=0.7)
    ax1.set_yticks(range(len(user_avg_fairness)))
    ax1.set_yticklabels(user_avg_fairness['user'], fontsize=8)
    ax1.set_xlabel("Average Sharing Incentive Score")
    ax1.set_title("Average Sharing Incentive (Fairness) Per User\n(Green: Fair <1, Orange: Moderate 1-10, Red: Unfair >10)", fontsize=12, fontweight='bold')
    ax1.axvline(x=1.0, color='black', linestyle='--', linewidth=2, label='Fairness Threshold (1.0)')
    ax1.axvline(x=10.0, color='red', linestyle='--', linewidth=2, label='High Unfairness (10.0)')
    ax1.legend()
    ax1.grid(True, alpha=0.3, axis='x')
    
    # Plot 2: Histogram of average fairness
    ax2 = axes[1]
    ax2.hist(user_avg_fairness['avg_sharing_incentive'], bins=30, color='steelblue', edgecolor='black', alpha=0.7)
    ax2.set_xlabel("Average Sharing Incentive Score")
    ax2.set_ylabel("Number of Users")
    ax2.set_title("Distribution of Average Sharing Incentive Across All Users", fontsize=12, fontweight='bold')
    ax2.axvline(x=user_avg_fairness['avg_sharing_incentive'].mean(), color='red', linestyle='--', linewidth=2, label=f"Mean: {user_avg_fairness['avg_sharing_incentive'].mean():.2f}")
    ax2.axvline(x=user_avg_fairness['avg_sharing_incentive'].median(), color='green', linestyle='--', linewidth=2, label=f"Median: {user_avg_fairness['avg_sharing_incentive'].median():.2f}")
    ax2.legend()
    ax2.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plot_path = os.path.join(OUTPUT_DIR, "2_average_fairness_distribution.png")
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {plot_path}")
    plt.close()
    
    # Save statistics to CSV
    stats_path = os.path.join(OUTPUT_DIR, "user_fairness_statistics.csv")
    user_avg_fairness.to_csv(stats_path, index=False)
    print(f"✓ Saved: {stats_path}")
    
    return user_avg_fairness

def analyze_high_sharing_incentive(combined_df):
    """
    3) Analyze correlation between jobs with high sharing incentive (>10).
    Look at: submit_time, duration, user_dur, group_dur, group_gpu_dur
    """
    print("Analyzing high sharing incentive jobs (>10)...")
    
    # Load the trace file
    trace_df = load_trace_file(TRACE_FILE)
    
    # Filter for high sharing incentive jobs
    high_si = combined_df[combined_df['sharing_incentive'] > 10].copy()
    high_si['job_id'] = high_si['job_id'].astype(int)
    
    print(f"Found {len(high_si)} jobs with sharing_incentive > 10")
    
    if len(high_si) == 0:
        print("No jobs with sharing_incentive > 10 found.")
        return
    
    # Merge with trace data
    trace_df['job_id'] = trace_df['job_id'].astype(int)
    merged = high_si.merge(trace_df, left_on='job_id', right_on='job_id', how='left')
    
    # Get relevant columns
    relevant_cols = ['job_id', 'user_x', 'sharing_incentive', 'submit_time', 'duration', 'user_dur', 'group_dur', 'group_gpu_dur']
    available_cols = [col for col in relevant_cols if col in merged.columns]
    analysis_df = merged[available_cols].copy()
    
    # Rename user_x to user for clarity
    if 'user_x' in analysis_df.columns:
        analysis_df.rename(columns={'user_x': 'user'}, inplace=True)
    
    # Calculate correlations
    numeric_cols = [col for col in analysis_df.columns if col not in ['job_id', 'user']]
    correlation_data = analysis_df[numeric_cols].corr()
    
    print("\nCorrelation Matrix for High Sharing Incentive Jobs (>10):")
    print(correlation_data)
    
    # Create correlation heatmap
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(correlation_data, annot=True, fmt='.2f', cmap='coolwarm', center=0, 
                square=True, ax=ax, cbar_kws={'label': 'Correlation Coefficient'})
    ax.set_title(f"Correlation Matrix for Jobs with High Sharing Incentive (>10)\n({len(high_si)} jobs from {high_si['user'].nunique()} users)", 
                 fontsize=12, fontweight='bold')
    plt.tight_layout()
    plot_path = os.path.join(OUTPUT_DIR, "3_high_si_correlation_heatmap.png")
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {plot_path}")
    plt.close()
    
    # Create scatter plots for key relationships
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Scatter 1: sharing_incentive vs duration
    ax = axes[0, 0]
    ax.scatter(analysis_df['duration'], analysis_df['sharing_incentive'], alpha=0.6, s=50)
    ax.set_xlabel("Duration")
    ax.set_ylabel("Sharing Incentive")
    ax.set_title("Sharing Incentive vs Job Duration")
    ax.grid(True, alpha=0.3)
    
    # Scatter 2: sharing_incentive vs submit_time
    ax = axes[0, 1]
    ax.scatter(analysis_df['submit_time'], analysis_df['sharing_incentive'], alpha=0.6, s=50, color='orange')
    ax.set_xlabel("Submit Time")
    ax.set_ylabel("Sharing Incentive")
    ax.set_title("Sharing Incentive vs Submit Time")
    ax.grid(True, alpha=0.3)
    
    # Scatter 3: user_dur vs group_dur
    ax = axes[1, 0]
    ax.scatter(analysis_df['user_dur'], analysis_df['group_dur'], alpha=0.6, s=50, color='green')
    ax.set_xlabel("User Duration")
    ax.set_ylabel("Group Duration")
    ax.set_title("User Duration vs Group Duration")
    ax.grid(True, alpha=0.3)
    
    # Scatter 4: group_dur vs group_gpu_dur
    ax = axes[1, 1]
    ax.scatter(analysis_df['group_dur'], analysis_df['group_gpu_dur'], alpha=0.6, s=50, color='red')
    ax.set_xlabel("Group Duration")
    ax.set_ylabel("Group GPU Duration")
    ax.set_title("Group Duration vs Group GPU Duration")
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plot_path = os.path.join(OUTPUT_DIR, "3_high_si_scatter_plots.png")
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {plot_path}")
    plt.close()
    
    # Save the high SI data
    high_si_path = os.path.join(OUTPUT_DIR, "high_sharing_incentive_jobs.csv")
    analysis_df.to_csv(high_si_path, index=False)
    print(f"✓ Saved: {high_si_path}")
    
    # Save correlation matrix
    corr_path = os.path.join(OUTPUT_DIR, "high_si_correlation_matrix.csv")
    correlation_data.to_csv(corr_path)
    print(f"✓ Saved: {corr_path}")
    
    # Print summary statistics
    print("\n" + "="*70)
    print("SUMMARY STATISTICS FOR HIGH SHARING INCENTIVE JOBS (>10)")
    print("="*70)
    print(f"Total jobs: {len(analysis_df)}")
    print(f"Unique users: {analysis_df['user'].nunique()}")
    print(f"\nSharing Incentive Statistics:")
    print(analysis_df['sharing_incentive'].describe())
    print(f"\nDuration Statistics:")
    print(analysis_df['duration'].describe())
    print(f"\nUser Duration Statistics:")
    print(analysis_df['user_dur'].describe())
    print(f"\nGroup Duration Statistics:")
    print(analysis_df['group_dur'].describe())
    print(f"\nGroup GPU Duration Statistics:")
    print(analysis_df['group_gpu_dur'].describe())
    print("="*70)

def main():
    print("="*70)
    print("FAIRNESS ANALYSIS FOR GPU CLUSTER SCHEDULER")
    print("="*70)
    print()
    
    # Analysis 1: Distribution of fairness across jobs per user
    combined_df = analyze_fairness_distributions()
    # print()
    
    # # Analysis 2: Average fairness per user
    # user_stats = analyze_average_fairness(combined_df)
    # print()
    
    # # Analysis 3: High sharing incentive correlation
    # analyze_high_sharing_incentive(combined_df)
    # print()
    
    print("="*70)
    print("Analysis complete! All outputs saved to:")
    print(OUTPUT_DIR)
    print("="*70)

if __name__ == "__main__":
    main()
