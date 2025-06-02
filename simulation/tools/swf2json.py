from enum import Enum
import pandas as pd
import numpy as np
import argparse
import progressbar
from sortedcontainers import SortedSet
from data_preprocessing import GetWorkloadData
from json_generator import generateJSON, generateJobData


def loadData(inputFile, workload_name, platform_size):
    user_estimated_jobs = list()
    exact_jobs = list()

    profile_set = SortedSet()
    profiles = {}

    job_info_data = []

    earliest_recorded_submission_time = 0

    workload_data = GetWorkloadData(inputFile, workload_name, platform_size)
    dataset = workload_data.to_numpy()

    job_index = 0

    bar = progressbar.ProgressBar(max_value=len(dataset))

    """
    ["jobID", 
    "submission_time",
    "execution_time",
    "proc_req",
    "user_est",
    "uid",
    "estimation_deviation",]
    """
    print("--------Generating Profile---------")

    for job in dataset:
        [
            jobID,
            submission_time,
            waiting_time,
            execution_time,
            proc_alloc,
            cpu_used,
            mem_used,
            res,
            user_est,
            uid,
            gid,
            exe_num,
            queue,
        ] = job

        if job_index == 0:
            earliest_recorded_submission_time = max(0, int(submission_time))

        profile_item = int(execution_time)
        profile_set.add(profile_item)

        job_profile = {
            "id": job_index,
            "subtime": max(0, int(submission_time)) - earliest_recorded_submission_time,
            "walltime": int(user_est),
            "res": int(res),
            "profile": str(profile_item),
            "uid": int(uid),
            "gid": int(gid),
            "exe_num": int(exe_num),
            "queue": int(queue),
        }

        job_info_data.append(job)

        user_estimated_jobs.append(job_profile)

        job_profile = job_profile.copy()
        job_profile["walltime"] = int(execution_time)
        exact_jobs.append(job_profile)

        job_index += 1
        bar.update(job_index)

    bar.finish()

    for profile_item in profile_set:
        profiles[str(profile_item)] = {"type": "delay", "delay": profile_item}

    print("--------Generated Successfully---------")

    # Return list of jobs and profiles
    return (exact_jobs, user_estimated_jobs, job_info_data, profiles)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate Batsim JSON workload from Parallel Workload Archive SWF file"
    )
    parser.add_argument(
        "-i",
        "--inputFile",
        type=str,
        required=True,
        help="Input swf file",
    )

    parser.add_argument(
        "-o", "--outputFile", type=str, required=True, help="Output JSON file"
    )

    parser.add_argument(
        "-p", "--platformSize", type=int, required=True, help="System platform size"
    )
    parser.add_argument(
        "-w",
        "--workloadName",
        type=str,
        required=False,
        help="Workload name for custom functions",
    )

    args = parser.parse_args()

    # Read SWF File
    (
        exact_jobs,
        user_estimated_jobs,
        uid_list,
        profiles,
    ) = loadData(args.inputFile, args.workloadName, args.platformSize)

    # Then save to JSON
    generateJSON(
        exact_jobs,
        profiles,
        args.platformSize,
        "../workload/exact/{}".format(args.outputFile),
    )
    generateJSON(
        user_estimated_jobs,
        profiles,
        args.platformSize,
        "../workload/user-estimate/{}".format(args.outputFile),
    )

    # Save uid json
    # generateJobData(uid_list, "../extra/{}".format(args.outputFile))

    print("==========Export Done===========")
