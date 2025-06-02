from evalys.workload import Workload
from workload_filter import WorkloadCharacteristics


def GetWorkloadData(path, workload_name, platform_size):
    print("Loading data of {}".format(workload_name))
    workload = Workload.from_csv(path).df.head(5000)
    print("Done load Dataframe")
    workload["estimation_deviation"] = workload.user_est / workload.execution_time

    (condition, normalize) = WorkloadCharacteristics(workload_name, platform_size)

    print("Done Normalization Function & Filtering Condition")

    filtered_workload = workload[workload.apply(condition, axis=1)]

    print("Done Filtering")

    converted_workload = filtered_workload.apply(normalize, axis=1)

    print("Done Normalizing")

    workload_data = converted_workload[
        [
            "jobID",
            "submission_time",
            "waiting_time",
            "execution_time",
            "proc_alloc",
            "cpu_used",
            "mem_used",
            "res",
            "user_est",
            "uid",
            "gid",
            "exe_num",
            "queue",
        ]
    ]

    print("Done load data")
    return workload_data
