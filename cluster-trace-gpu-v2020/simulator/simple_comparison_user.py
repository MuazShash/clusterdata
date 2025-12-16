#!/usr/bin/env python3
import csv
import os
import sys
from math import ceil
import subprocess
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from utils import print_fn, ALLOC_POLICY_DICT, PREEMPT_POLICY_DICT
from simulator import Simulator
from pathlib import Path
from collections import defaultdict
from bisect import bisect_left, bisect_right
import pandas as pd

def create_filtered_traces(original_trace, target_user, output_path):
    """Create single-user and multi-user traces for comparison."""

    print(f"Reading original trace: {original_trace}")

    jobs = []
    user_counts = defaultdict(int)

    with open(original_trace, "r") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        for row in reader:
            jobs.append(row)
            user_counts[row["user"]] += 1

    print(f"Found {len(jobs)} jobs from {len(user_counts)} users")

    # Create single-user trace
    user_jobs = [job for job in jobs if job["user"] == target_user]
    # Sort by original submit time and reset to start from 0
    user_jobs.sort(key=lambda x: int(x.get("submit_time", 0)))

    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(user_jobs)

    print(f"Created single-user trace: {len(user_jobs)} jobs")

    return output_path, len(user_jobs)

def get_average_concurrent_users(target_user, trace_filename):
    """
    Computes the average number of users running concurrently with a given user in the trace.
    """

    path = Path(trace_filename)
    with path.open("r", newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        if not rows:
            raise ValueError("Input trace is empty")

    class SubmitDuration:
        def __init__(self, user, start, duration):
            self.user = user
            self.start = start
            self.end = start + max(0.0, duration)

    # map available header names to canonical start/duration keys
    user_key = "user"
    start_key = "submit_time"
    dur_key = "duration"

    # parse intervals
    submits = []
    for r in rows:
        try:
            id = str(r.get(user_key, "unknown"))
        except ValueError:
            raise ValueError(f"User field not found for {r}")
        try:
            s = float(r.get(start_key, 0) or 0)
        except ValueError:
            raise ValueError(f"Start not found for {r}")
        try:
            d = float(r.get(dur_key, 0) or 0)
        except ValueError:
           raise ValueError(f"End not found for {r}")
        submits.append(SubmitDuration(id, s, d))

    # prepare sorted lists for sweep counting
    starts_sorted = sorted([s.start for s in submits])
    ends_sorted = sorted([s.end for s in submits])

    concurrent_users = []
    for submission in submits:
        if(submission.user != target_user):
            continue
        overlapping_users = set()
        num_start_before_end = bisect_left(starts_sorted, submission.end)
        num_end_after_start = bisect_right(ends_sorted, submission.start)
        for overlapping_submission in submits[num_end_after_start:num_start_before_end]:
            overlapping_users.add(overlapping_submission.user)
        concurrent_users.append(len(overlapping_users))
    
    average_concurrent_users = sum(concurrent_users) / len(concurrent_users) if concurrent_users else 0
    print(f"Average concurrent users for {target_user}: {average_concurrent_users}")
    return average_concurrent_users

def run_simulation(trace_filename, num_jobs, num_gpus):
    
    ARRIVAL_RATE = -1
    REPEAT = 1
    SORT_NODE_POLICY = 3  # 0: packing, 3: max-min balancing.
    MAX_TIME = int(1e9)

    VERBOSE = 0
    LOG_LEVEL = logging.WARNING
    NUM_NODES = 1
    NUM_CPUS = round(23.22 * num_gpus)
    HETERO = False
    PATTERN = 0  # Cluster capacity varying pattern

    # GPU_TYPE_MATCHING = 1 # GPU type perfect match
    # GPU_TYPE_MATCHING = 2 # Only V100 cannot compromise
    GPU_TYPE_MATCHING = 0
    
    EXPORT_JOB_STATS = False
    RANDOM_SEED = 42
    NUM_SPARE_NODE = 0
    EXPORT_CLUSTER_UTIL = False
    SORT_BY_JCT = True
    
    DESCRIBE_FILE = None
    describe_file = trace_filename / DESCRIBE_FILE if DESCRIBE_FILE is not None else None
    
    for alloc_policy in [1]:  # just SJF
        for preempt_policy in [2]:  # 2LGF
            key = (alloc_policy, preempt_policy)
            print_key = "(%-4s,%4s)" % (ALLOC_POLICY_DICT.get(key[0]), PREEMPT_POLICY_DICT.get(key[1]))

            simulator = Simulator(
                csv_file=trace_filename,
                alloc_policy=alloc_policy,
                preempt_policy=preempt_policy,
                sort_node_policy=SORT_NODE_POLICY,
                num_nodes=NUM_NODES,
                random_seed=RANDOM_SEED,
                max_time=MAX_TIME,
                num_spare_node=NUM_SPARE_NODE,
                pattern=PATTERN,
                hetero=HETERO,
                num_gpus=num_gpus,
                num_cpus=NUM_CPUS,
                describe_file=describe_file,
                log_file=None,
                export_job_stats=EXPORT_JOB_STATS,
                export_cluster_util=EXPORT_CLUSTER_UTIL,
                arrival_rate=ARRIVAL_RATE,
                num_jobs_limit=num_jobs,
                gpu_type_matching=GPU_TYPE_MATCHING,
                verbose=VERBOSE)
            results = simulator.simulator_go(repeat=REPEAT)

    return simulator.cluster.job_history.job_done_list


def compare_single_and_actual_jct(user, single_results, original_results, num_jobs_in_original):
    sharing_incentives = []
    
    with open(f"results/{user}.csv", "w") as f:
        for job in single_results:
            if(job["job_id"] > num_jobs_in_original):
                break
            jct_single = job["jct"]
            jct_original = next((job_result["jct"] for job_result in original_results if job_result["job_id"] == job["job_id"]), None)
            print(f"Job {job['job_id']} - Single-user JCT: {jct_single}, Original JCT: {jct_original}")
            print(" Sharing incentive = ", jct_original / jct_single)
            sharing_incentives.append(jct_original / jct_single)
            f.write(f"{job['job_id']},{jct_single},{jct_original},{jct_original / jct_single}\n")

        f.write(f"Average sharing incentive for user {user} : {sum(sharing_incentives) / len(single_results)}. (< 1 preferred for fairness)\n")
    print(f"Average sharing incentive for user {user} : {sum(sharing_incentives) / len(single_results)}. (< 1 preferred for fairness)")

def process_user_task(target_user, original_trace_path, original_results, num_jobs, num_gpus):
    """
    Process a single user task: compute concurrent users, filter trace, run simulation, and compare.
    This function is designed to be run in a thread pool.
    """
    try:
        cluster_denominator = get_average_concurrent_users(target_user, original_trace_path)
        
        single_output_trace = Path(f"simulator/traces/pai/single_user_temp_{target_user}.csv")
        
        single_trace, num_jobs_per_user = create_filtered_traces(
            original_trace_path, target_user, single_output_trace
        )
        # Run simulations
        num_gpus_for_user = ceil(num_gpus / cluster_denominator)  
        print(f"Running simulations for user {target_user} with {num_gpus_for_user} GPUs...")
        
        single_results = run_simulation(single_trace, num_jobs_per_user, num_gpus_for_user)
        compare_single_and_actual_jct(target_user, single_results, original_results, num_jobs)
        
        # Clean up temp file
        if single_output_trace.exists():
            single_output_trace.unlink()
        
        return target_user, True
    except Exception as e:
        print(f"Error processing user {target_user}: {e}")
        import traceback
        traceback.print_exc()
        return target_user, False


def main(user):
    num_jobs = 100000
    num_gpus = 6500

    original_results_path = Path("results/original_results.csv")
    original_results = pd.read_csv(original_results_path).to_dict(orient='records')
    original_trace_path = Path("simulator/traces/pai/pai_job_duration_estimate_100K.csv")
    # breakpoint()
    # Create results directory if it doesn't exist
    Path("results").mkdir(exist_ok=True)
    process_user_task(user, original_trace_path, original_results, num_jobs, num_gpus)
    
    return 0
    
if __name__ == "__main__":
    # sys.exit(main())
    sys.exit(main(sys.argv[1]))
