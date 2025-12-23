import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Load the trace file with job durations
print("Loading trace file...")
trace_df = pd.read_csv('simulator/traces/pai/pai_job_duration_estimate_100K.csv')
trace_dict = dict(zip(trace_df['job_id'], trace_df['duration']))

# Define policies
policies = ['fifo', 'sjf', 'sjg', 'sjgg', 'sju']
policy_names = {
    'fifo': 'FIFO',
    'sjf': 'SJF',
    'sjg': 'SJG',
    'sjgg': 'SJGG',
    'sju': 'SJU'
}

# Collect data for each policy
data_by_policy = {}

for policy in policies:
    results_dir = f'results_{policy}'
    
    if not os.path.exists(results_dir):
        print(f"Warning: Directory {results_dir} does not exist")
        continue
    
    job_ids = []
    durations = []
    sharing_incentives = []
    
    # Read each CSV file in the directory
    for filename in os.listdir(results_dir):
        if filename.endswith('.csv'):
            filepath = os.path.join(results_dir, filename)
            
            # Read the file
            with open(filepath, 'r') as f:
                lines = f.readlines()
                
            # The first line contains the data, last line is the average (to be ignored)
            if len(lines) >= 2:
                # Parse the first line to get job_id and sharing incentive
                first_line = lines[0].strip()
                values = first_line.split(',')
                
                job_id = int(values[0])
                sharing_incentive = float(values[-1])
                
                # Cap sharing incentive at 100
                sharing_incentive = min(sharing_incentive, 100)
                
                # Get duration for this job from trace file
                if job_id in trace_dict:
                    duration = trace_dict[job_id]
                    
                    job_ids.append(job_id)
                    durations.append(duration)
                    sharing_incentives.append(sharing_incentive)
    
    data_by_policy[policy] = {
        'job_ids': job_ids,
        'durations': durations,
        'sharing_incentives': sharing_incentives
    }
    
    print(f"Policy {policy}: {len(job_ids)} jobs matched with trace data")

# Create scatter plots
fig, axes = plt.subplots(2, 3, figsize=(18, 12))
axes = axes.flatten()

for idx, policy in enumerate(policies):
    ax = axes[idx]
    data = data_by_policy[policy]
    
    # Create scatter plot
    ax.scatter(data['durations'], data['sharing_incentives'], 
               alpha=0.5, s=30, edgecolors='black', linewidth=0.3)
    
    # Add horizontal line at y=1 (fairness threshold)
    ax.axhline(y=1, color='red', linestyle='--', linewidth=1.5, alpha=0.7, label='Fair (y=1)')
    
    # Labels and title
    ax.set_xlabel('Job Duration (seconds)', fontsize=12)
    ax.set_ylabel('Sharing Incentive', fontsize=12)
    ax.set_title(f'{policy_names[policy]} Policy', fontsize=14, fontweight='bold')
    ax.set_xscale('log')
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    # Add statistics text
    mean_incentive = np.mean(data['sharing_incentives'])
    median_incentive = np.median(data['sharing_incentives'])
    
    # Calculate correlation
    correlation = np.corrcoef(data['durations'], data['sharing_incentives'])[0, 1]
    
    text = f'Mean SI: {mean_incentive:.3f}\nMedian SI: {median_incentive:.3f}\nCorrelation: {correlation:.3f}'
    ax.text(0.02, 0.98, text, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

# Remove the 6th subplot (we only have 5 policies)
fig.delaxes(axes[5])

plt.tight_layout()
plt.savefig('sharing_incentive_vs_duration_all.png', dpi=300, bbox_inches='tight')
print("\nCombined plot saved as 'sharing_incentive_vs_duration_all.png'")

# Also save individual plots
for policy in policies:
    fig, ax = plt.subplots(figsize=(10, 8))
    data = data_by_policy[policy]
    
    # Create scatter plot
    ax.scatter(data['durations'], data['sharing_incentives'], 
               alpha=0.5, s=50, edgecolors='black', linewidth=0.5)
    
    # Add horizontal line at y=1 (fairness threshold)
    ax.axhline(y=1, color='red', linestyle='--', linewidth=2, alpha=0.7, label='Fair (y=1)')
    
    # Labels and title
    ax.set_xlabel('Job Duration (seconds)', fontsize=14)
    ax.set_ylabel('Sharing Incentive', fontsize=14)
    ax.set_title(f'{policy_names[policy]} Policy - Sharing Incentive vs Job Duration', 
                 fontsize=16, fontweight='bold')
    ax.set_xscale('log')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=12)
    
    # Add statistics text
    mean_incentive = np.mean(data['sharing_incentives'])
    median_incentive = np.median(data['sharing_incentives'])
    min_incentive = np.min(data['sharing_incentives'])
    max_incentive = np.max(data['sharing_incentives'])
    
    # Calculate correlation
    correlation = np.corrcoef(data['durations'], data['sharing_incentives'])[0, 1]
    
    # Statistics for duration
    mean_duration = np.mean(data['durations'])
    median_duration = np.median(data['durations'])
    
    text = f'Sharing Incentive:\n  Mean: {mean_incentive:.3f}\n  Median: {median_incentive:.3f}\n  Min: {min_incentive:.3f}\n  Max: {max_incentive:.3f}\n\n'
    text += f'Duration (sec):\n  Mean: {mean_duration:.1f}\n  Median: {median_duration:.1f}\n\n'
    text += f'Correlation: {correlation:.3f}'
    
    ax.text(0.02, 0.98, text, transform=ax.transAxes, fontsize=11,
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))
    
    plt.tight_layout()
    plt.savefig(f'sharing_incentive_vs_duration_{policy}.png', dpi=300, bbox_inches='tight')
    print(f"Individual plot saved as 'sharing_incentive_vs_duration_{policy}.png'")

# Print summary statistics for all policies
print("\n" + "="*80)
print("SUMMARY: Sharing Incentive vs Duration Analysis")
print("="*80)
for policy in policies:
    data = data_by_policy[policy]
    correlation = np.corrcoef(data['durations'], data['sharing_incentives'])[0, 1]
    print(f"\n{policy_names[policy]}:")
    print(f"  Jobs analyzed: {len(data['job_ids'])}")
    print(f"  Correlation (duration vs SI): {correlation:.4f}")
    print(f"  Mean sharing incentive: {np.mean(data['sharing_incentives']):.3f}")
    print(f"  Median sharing incentive: {np.median(data['sharing_incentives']):.3f}")
    print(f"  Mean duration: {np.mean(data['durations']):.1f} sec")
    print(f"  Median duration: {np.median(data['durations']):.1f} sec")
print("="*80)

plt.show()
