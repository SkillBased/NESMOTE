import numpy as np

from sklearn.datasets import make_moons

from RQ.neighbors import RingQuery, SortedSetQuery
from NESMOTE.util import std_euclid_distance, IndexedCoverTree
from swg import HNSW

from sklearn.neighbors import BallTree

from time import time

class Timer:
    def __init__(self, name : str, timers : list[str] = []) -> None:
        self.name = name
        self.timers = {timer : [] for timer in timers}
        self.timestamp = 0
    
    def start(self):
        if self.timestamp != 0:
            raise(Exception("timer already started"))
        self.timestamp = time()
    
    def stop(self, timer_id : str) -> None:
        result = time() - self.timestamp
        self.timestamp = 0
        if self.timers.get(timer_id) is None:
            self.timers[timer_id] = []
        self.timers[timer_id].append(result)
    
    def stop_start(self, timer_id : str) -> None:
        self.stop(timer_id)
        self.start()
    
    def get(self, timer_id : str) -> float:
        if self.timers.get(timer_id) is None:
            raise(ValueError)
        srt = sorted(self.timers[timer_id])
        avg = np.mean(srt)
        disp = max(avg - srt[0], srt[-1] - avg)
        return avg, disp

def run_trial(nruns : int = 16, k : int = 5, sizes : list[int] = [1024]):
    timers = []
    head = "{:<20}".format("running trial") + "-" * nruns
    print(head)
    for size in sizes:
        oX, oy = make_moons(size * 2, shuffle=True, noise=.1, random_state=42)
        X = oX[oy == 0]
        timer = Timer("nsamples = {:<8} ".format(size))
        print(timer.name, end="", flush=True)

        for _ in range(nruns):

            timer.start()
            bt = BallTree(X)
            timer.stop_start("BT - construct")
            bt.query(X, k=k)
            timer.stop("BT - query all")

            timer.start()
            rg = RingQuery(std_euclid_distance)
            rg.fit(X)
            timer.stop_start("RQ - construct")
            for x in X:
                rg.query(x, k=k)
            timer.stop("RQ - query all")

            timer.start()
            rg = HNSW(std_euclid_distance)
            rg.fit(X)
            timer.stop_start("SW - construct")
            for x in X:
                rg.query(x, k=k)
            timer.stop("SW - query all")

            timer.start()
            rg = SortedSetQuery(std_euclid_distance)
            rg.fit(X)
            timer.stop_start("SS - construct")
            for x in X:
                rg.query(x, k=k)
            timer.stop("SS - query all")

        
            print("#", end="", flush=True)
        
        timers.append(timer)
        print()
    
    return timers

sizes = [128, 256, 512, 1024, 2048, 4096, 8192, 16384]
requests = ["BT - construct", "BT - query all", "RQ - construct", "RQ - query all", "SW - construct", "SW - query all", "SS - construct", "SS - query all"]

timers = run_trial(sizes=sizes)

outfile = open("moons.out", "w")

headline = "{:<20}".format("run details")
for req in requests:
    headline += "{:>20}".format(req)
headline += "\n"
outfile.write(headline)

for timer in timers:
    line = timer.name
    for req in requests:
        avg, disp = timer.get(req)
        line += " {:8.4f} +- {:6.4f}s".format(avg, disp)
    line += "\n"
    outfile.write(line)

outfile.close()

