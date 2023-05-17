import numpy as np

from imblearn.datasets import fetch_datasets

from NESMOTE.neighbors import RingQuery
from NESMOTE.util import std_euclid_distance

from sklearn.neighbors import BallTree

from time import time


class Score:
    def __init__(self) -> None:
        self.n = 0
        self._sum = 0
        self._max = 0
        self._min = 1e9
    
    def add(self, val):
        self.n += 1
        self._sum += val
        if val < self._min:
            self._min = val
        if val > self._max:
            self._max = val
    
    def get(self):
        mean = self._sum / self.n
        mindiff = mean - self._min
        maxdiff = self._max - mean
        return "{:8.4f} +- {:6.4f}".format(mean, max(mindiff, maxdiff))


def run_trial(X, nruns=16, k=5):

    scores = [Score(), Score(), Score(), Score()]

    for _ in range(nruns):
        
        timer = time()
        bt = BallTree(X)
        scores[0].add(time() - timer)
        timer = time()
        bt.query(X[0].reshape(1, -1), k=k)
        scores[1].add(time() - timer)
    
        timer = time()
        rg = RingQuery(std_euclid_distance)
        rg.fit(X)
        scores[2].add(time() - timer)
        timer = time()
        rg.query(X[0], k=k)
        scores[3].add(time() - timer)
    
    return scores[0].get(), scores[1].get(), scores[2].get(), scores[3].get()



datasets = fetch_datasets()

ds_names = ["ecoli", "optical_digits", "abalone", "sick_euthyroid", "spectrometer", "car_eval_34",
            "us_crime", "yeast_ml8", "car_eval_4", "thyroid_sick", "wine_quality", "solar_flare_m0",
            "yeast_me2", "ozone_level", "abalone_19"]

outfile = open("timers.out", "w")

outfile.write("{:15} {:>18} {:>18} {:>18} {:>18}\n".format("dataset", "ball - prep", "ball - query", "ring - prep", "ring - query"))
for name in ds_names:
    X, y = datasets[name].data, datasets[name].target
    print(f"running trial {name}: ", end="", flush=True)
    bt_construct, bt_query, ring_cnstruct, ring_query = run_trial(X[y==-1])
    outfile.write("{:15} {} {} {} {}\n".format(name, bt_construct, bt_query, ring_cnstruct, ring_query))
    print(f"trial {name} completed", flush=True)

outfile.close()