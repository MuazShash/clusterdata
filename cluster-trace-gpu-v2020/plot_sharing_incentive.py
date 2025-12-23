import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Load concurrent users data
concurrent_users_df = pd.read_csv(
    "concurrent_users.csv", header=None, names=["user", "concurrent_users"]
)
concurrent_users_dict = dict(
    zip(concurrent_users_df["user"], concurrent_users_df["concurrent_users"])
)

# Define policies
policies = ["fifo", "sjf", "sjg", "sjgg", "sju"]
policy_names = {
    "fifo": "FIFO",
    "sjf": "SJF",
    "sjg": "SJG",
    "sjgg": "SJGG",
    "sju": "SJU",
}

# Collect data for each policy
data_by_policy = {}

for policy in policies:
    results_dir = f"results_{policy}"

    if not os.path.exists(results_dir):
        print(f"Warning: Directory {results_dir} does not exist")
        continue

    users = []
    concurrent_users_list = []
    sharing_incentives = []

    # Read each CSV file in the directory
    for filename in os.listdir(results_dir):
        if filename.endswith(".csv"):
            user_id = filename[:-4]  # Remove .csv extension
            filepath = os.path.join(results_dir, filename)

            # Read the file
            with open(filepath, "r") as f:
                lines = f.readlines()

            lines = lines[0:-1]
            # The first line contains the data, last line is the average (to be ignored)
            sharing_incentive = 0.0
            for line in lines:
                # Parse the first line to get the sharing incentive (last column)
                first_line = line.strip()
                values = first_line.split(",")
                sharing_incentive += float(values[-1]) / (len(lines))

                # Cap sharing incentive at 100
                # sharing_incentive = min(sharing_incentive, 100)

                # Get concurrent users for this user
            if user_id in concurrent_users_dict:
                concurrent_users_count = concurrent_users_dict[user_id]

                users.append(user_id)
                concurrent_users_list.append(concurrent_users_count)
                sharing_incentives.append(sharing_incentive)

    data_by_policy[policy] = {
        "users": users,
        "concurrent_users": concurrent_users_list,
        "sharing_incentives": sharing_incentives,
    }

    print(f"Policy {policy}: {len(users)} users")

# Create scatter plots
fig, axes = plt.subplots(2, 3, figsize=(18, 12))
axes = axes.flatten()

for idx, policy in enumerate(policies):
    ax = axes[idx]
    data = data_by_policy[policy]

    # Create scatter plot
    ax.scatter(
        data["concurrent_users"],
        data["sharing_incentives"],
        alpha=0.6,
        s=50,
        edgecolors="black",
        linewidth=0.5,
    )

    # Add horizontal line at y=1 (fairness threshold)
    ax.axhline(
        y=1, color="red", linestyle="--", linewidth=1.5, alpha=0.7, label="Fair (y=1)"
    )

    # Labels and title
    ax.set_ylim(0.9, 100)
    # ax.set_yscale("log")
    ax.set_xlabel("Concurrent Users", fontsize=12)
    ax.set_ylabel("Average Sharing Incentive", fontsize=12)
    ax.set_title(f"{policy_names[policy]} Policy", fontsize=14, fontweight="bold")
    ax.grid(True, alpha=0.3)
    ax.legend()

    # Add statistics text
    mean_incentive = np.mean(data["sharing_incentives"])
    median_incentive = np.median(data["sharing_incentives"])
    text = f"Mean: {mean_incentive:.3f}\nMedian: {median_incentive:.3f}"
    ax.text(
        0.02,
        0.98,
        text,
        transform=ax.transAxes,
        fontsize=10,
        verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
    )

# Remove the 6th subplot (we only have 5 policies)
fig.delaxes(axes[5])

plt.tight_layout()
plt.savefig("sharing_incentive_scatter_plots.png", dpi=300, bbox_inches="tight")
print("\nScatter plots saved as 'sharing_incentive_scatter_plots.png'")

# Also save individual plots
for policy in policies:
    fig, ax = plt.subplots(figsize=(10, 8))
    data = data_by_policy[policy]

    # Create scatter plot
    ax.scatter(
        data["concurrent_users"],
        data["sharing_incentives"],
        alpha=0.6,
        s=80,
        edgecolors="black",
        linewidth=0.5,
    )

    # Add horizontal line at y=1 (fairness threshold)
    ax.axhline(
        y=1, color="red", linestyle="--", linewidth=2, alpha=0.7, label="Fair (y=1)"
    )

    # Labels and title
    ax.set_xlabel("Concurrent Users", fontsize=14)
    ax.set_ylabel("Average Sharing Incentive", fontsize=14)
    ax.set_title(
        f"{policy_names[policy]} Policy - Sharing Incentive vs Concurrent Users",
        fontsize=16,
        fontweight="bold",
    )
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=12)

    # Add statistics text
    mean_incentive = np.mean(data["sharing_incentives"])
    median_incentive = np.median(data["sharing_incentives"])
    min_incentive = np.min(data["sharing_incentives"])
    max_incentive = np.max(data["sharing_incentives"])
    text = f"Mean: {mean_incentive:.3f}\nMedian: {median_incentive:.3f}\nMin: {min_incentive:.3f}\nMax: {max_incentive:.3f}"
    ax.text(
        0.02,
        0.98,
        text,
        transform=ax.transAxes,
        fontsize=12,
        verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
    )

    plt.tight_layout()
    plt.savefig(f"sharing_incentive_{policy}.png", dpi=300, bbox_inches="tight")
    print(f"Individual plot saved as 'sharing_incentive_{policy}.png'")

plt.show()
