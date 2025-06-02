### Custom filter and data normalization for each workload
def WorkloadCharacteristics(workload_name, platform_size):
    # default filter
    condition = (
        lambda job: job.proc_alloc > 0
        and job.execution_time > 0
        and job.submission_time >= 0
        and job.user_est > 0
    )
    normalize = NormalizeJob(platform_size, 1)

    # in case SDSC-DS
    if workload_name == "SDSC-DS":
        condition = (
            lambda job: job.proc_alloc > 0
            and job.execution_time > 0
            and job.submission_time >= 0
            and job.partition == 2  # BATCH Partition
        )
        normalize = NormalizeJob(platform_size, 8)

    elif workload_name == "ANL-Intrepid":
        normalize = NormalizeJob(platform_size, 256)

    elif workload_name == "SDSC-BLUE":
        condition = (
            lambda job: job.proc_alloc > 0
            and job.execution_time > 0
            and job.submission_time >= 0
            and job.queue != 0
        )
        normalize = NormalizeJob(platform_size, 8)
    else:
        print("No special normalization for {}".format(workload_name))
    return (condition, normalize)


def NormalizeJob(platform_size, node_cpu_count):
    def job_normalize_func(job):
        job["res"] = min(int(job.proc_alloc / node_cpu_count), platform_size)
        return job

    return job_normalize_func