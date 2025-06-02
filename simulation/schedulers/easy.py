from batsim.batsim import BatsimScheduler
import numpy as np
from sortedcontainers import SortedSet, SortedList, SortedListWithKey
from procset import ProcSet
import logging
import json
from pprint import pprint
from predictor.knn import KnnWalltimePredictor


EXTEND_BACKFILLING = False
USE_BUFFER_BACKFILLING = False


class Easy(BatsimScheduler):
    def __init__(self, options):
        self.options = options
        self.logger = logging.getLogger(__name__)

    def onAfterBatsimInit(self):
        self.total_nodes = self.bs.nb_resources
        self.idleNodes = SortedList(range(self.total_nodes))
        self.listRunningJobs = SortedListWithKey(
            key=lambda job: job.estimate_finish_time
        )
        self.jobWaitTimes = [0]
        self.listWaittingJobs = []
        self.predictionStatistics = []
        self.finishedJobs = []

    def onJobSubmission(self, job):
        # Check whether submitted job is bigger than system
        if job.requested_resources > self.bs.nb_compute_resources:
            # This job requests more resources than the cluster size
            self.bs.reject_jobs([job])

        job.is_backfilled = False

        current_time = self.bs.time()
        # If no remaining nodes, then add to waiting list
        # # if len(self.idleNodes) == 0:
        # self.waitingJobs.append(job)

        # self.executeSomeJobs(current_time)
        if len(self.idleNodes) < job.requested_resources:
            if len(self.listWaittingJobs) == 0:
                self.findShadowTimeAndExtraNodes(job)
            self.listWaittingJobs.append(job)
        else:
            job.estimate_finish_time = current_time + job.requested_time
            if (
                len(self.listWaittingJobs) == 0
                or job.estimate_finish_time <= self.shadowTime
            ):
                self.listRunningJobs.add(job)
                self.executeJob(job, current_time)
            elif (
                len(self.listWaittingJobs) > 0
                and self.shadowTime < job.estimate_finish_time
                and job.requested_resources <= self.extraNodes
            ):
                self.extraNodes -= job.requested_resources
                self.listRunningJobs.add(job)
                self.executeJob(job, current_time)
            else:
                # This job can not be backfilled
                self.listWaittingJobs.append(job)

    def onJobCompletion(self, job):
        current_time = self.bs.time()
        job.finish_time = current_time
        self.freeComputeNodes(job)
        self.listRunningJobs.remove(job)
        # print(current_time, job)
        if len(self.listWaittingJobs) > 0:
            self.executeSomeJobs(current_time)

        job_id = int(job.id.split("!")[1])

        self.finishedJobs.append(job)

        self.predictionStatistics.append([job_id, job.is_backfilled])

    def findShadowTimeAndExtraNodes(self, job):
        nbFreeNodes = len(self.idleNodes)
        for j in self.listRunningJobs:
            nbFreeNodes += j.requested_resources
            if job.requested_resources <= nbFreeNodes:
                self.extraNodes = nbFreeNodes - job.requested_resources
                # Shadowtime is the earliest time that the job can start
                if EXTEND_BACKFILLING:
                    if USE_BUFFER_BACKFILLING:
                        job_predict_runtime = KnnWalltimePredictor(
                            self.finishedJobs, job
                        )
                        self.shadowTime = j.estimate_finish_time + max(
                            0, job.requested_time - job_predict_runtime
                        )
                    else:
                        self.shadowTime = j.estimate_finish_time + int(
                            np.average(self.jobWaitTimes)
                        )
                else:
                    self.shadowTime = j.estimate_finish_time
                break

    def executeSomeJobs(self, current_time):
        if self.listWaittingJobs[0].requested_resources <= len(self.idleNodes):
            self.executeHeadOfList(self.listWaittingJobs, current_time)
            first_queued_job_alloc = False
        else:
            first_queued_job_alloc = True

        if first_queued_job_alloc == False and len(self.listWaittingJobs) > 0:
            self.findShadowTimeAndExtraNodes(self.listWaittingJobs[0])

        if len(self.listWaittingJobs) > 1:
            self.backFillJobs(self.listWaittingJobs, current_time)

    def executeHeadOfList(self, L, current_time):
        while len(L) > 0 and L[0].requested_resources <= len(self.idleNodes):
            job = L.pop(0)
            job.estimate_finish_time = current_time + job.requested_time
            self.listRunningJobs.add(job)
            self.executeJob(job, current_time)

    def backFillJobs(self, L, current_time):
        i = 1
        for j in range(len(L) - 1):
            if len(self.idleNodes) == 0:
                break
            job = L[i]
            if (
                job.requested_resources <= len(self.idleNodes)
                and current_time + job.requested_time <= self.shadowTime
            ):
                job.estimate_finish_time = current_time + job.requested_time
                self.listRunningJobs.add(job)
                del L[i]
                job.is_backfilled = True
                self.executeJob(job, current_time)
            elif (
                self.shadowTime < current_time + job.requested_time
                and job.requested_resources <= min(self.extraNodes, len(self.idleNodes))
            ):
                self.extraNodes -= job.requested_resources
                job.estimate_finish_time = current_time + job.requested_time
                self.listRunningJobs.add(job)
                del L[i]
                job.is_backfilled = True
                self.executeJob(job, current_time)
            else:
                i += 1

    def freeComputeNodes(self, job):
        self.idleNodes += job.nodes
        del job.nodes[:]

    def executeJob(self, job, current_time):
        # Get available resources
        allocated_nodes = self.idleNodes[0 : job.requested_resources]
        job.nodes = allocated_nodes
        del self.idleNodes[0 : job.requested_resources]

        # And create a corresponding ProcSet
        resources_allocation = ProcSet(*allocated_nodes)
        job.allocation = resources_allocation

        self.jobWaitTimes.append(current_time - job.submit_time)

        # Send an exercute msg (Need only job_id and allocation)
        self.bs.execute_jobs([job])

    def onSimulationEnds(self):
        outputFileName = "prediction-statistic.json"
        try:
            outputJSONFile = open(outputFileName, "w")
            json.dump(self.predictionStatistics, outputJSONFile, indent=2)
            print("Generated prediction statistic to {}".format(outputFileName))
        except IOError as ex:
            print(ex)
