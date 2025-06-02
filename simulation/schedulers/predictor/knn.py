import numpy as np
from numpy import linalg as LA
from timeit import default_timer as timer

NUMBER_NEIGHBORS = 5

def FindNearestNeighbors(
    dataset, current_job, number_of_neighbors
):

    extra_dataset = np.concatenate((dataset, [current_job]))
    # Columns: res, request_time, exe_num, uid, gid, queue
    categorical_indices = np.arange(len(current_job))[2:]
    nominal_indices = np.arange(len(current_job))[:2]

    nominal_data = extra_dataset[:, nominal_indices]
    nominal_ranges = np.max(nominal_data).flatten() -  np.min(nominal_data).flatten()

    nominal_ranges[nominal_ranges == 0] = 1

    nearest_neighbors = []
    idx = len(dataset) - 1

    # From latest finished jobs
    for job in reversed(dataset):
        distance = CalculateDistance(job, current_job, categorical_indices, nominal_indices, nominal_ranges)

        if distance < 100:
            if len(nearest_neighbors) < number_of_neighbors:
                nearest_neighbors.append({"index": idx, "distance": distance})
                nearest_neighbors.sort(key=lambda job: job["distance"])

            elif distance < nearest_neighbors[-1]["distance"]:
                nearest_neighbors.append({"index": idx, "distance": distance})
                nearest_neighbors.sort(key=lambda job: job["distance"])
                del nearest_neighbors[number_of_neighbors]
            
        idx -= 1
    return nearest_neighbors


def CalculateDistance(finish_job, current_job, categorical_indices, nominal_indices, nominal_ranges):
    categorical_distance = np.count_nonzero(finish_job[categorical_indices] != current_job[categorical_indices]) * 100
    nominal_distances = (finish_job[nominal_indices] - current_job[nominal_indices]) / nominal_ranges

    return np.sqrt(np.sum(np.square(categorical_distance)) + np.sum(np.square(nominal_distances)))

def CalculateWeight(distance):
    alpha = 1
    beta = 1
    return np.exp(-alpha * (distance ** beta))


def KnnWalltimePredictor(finished_jobs, job, data_size=1000, use_user_estimate=False):

    neighbor_space = finished_jobs[-data_size:]

    if (len(neighbor_space) < 2):
        return job.requested_time

    dataset = []
    for finished_job in neighbor_space:
        dataset.append(
            [
                finished_job.requested_resources,
                finished_job.requested_time,
                finished_job.json_dict["exe_num"],
                finished_job.json_dict["uid"],
                # finished_job.json_dict["gid"],
                # finished_job.json_dict["queue"],
            ]
        )

    job_info = np.array([
                job.requested_resources,
                job.requested_time,
                job.json_dict["exe_num"],
                job.json_dict["uid"],
                # job.json_dict["gid"],
                # job.json_dict["queue"],
            ])
    predict = job.requested_time

    nearest_neighbors = FindNearestNeighbors(np.array(dataset), job_info, NUMBER_NEIGHBORS)
    estimations = []
    runtimes = []
    weights = []

    if len(nearest_neighbors) > 0:
        for neighbor in nearest_neighbors:
            weight = CalculateWeight(neighbor["distance"])
            weights.append(weight)

            job_index = neighbor["index"]
            neighbor_info = neighbor_space[job_index]

            estimations.append(neighbor_info.requested_time / int(neighbor_info.profile) * weight)
            runtimes.append(int(neighbor_info.profile) * weight)
        
        if (use_user_estimate):
            calibration_factor = np.sum(estimations) / np.sum(weight)
            predict = max(job.requested_time // calibration_factor , 1)
        else:
            predict = max(np.sum(runtimes) // np.sum(weights), 1)

    return predict
