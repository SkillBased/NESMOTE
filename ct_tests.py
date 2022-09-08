import numpy as np

from time import time

from imblearn.datasets import fetch_datasets

from NESMOTE.util import std_euclid_distance
from NESMOTE.base import RingNNG, NeighborhoodGraph


dataset = fetch_datasets()["yeast_ml8"]
data, target = dataset.data, dataset.target
X = data[target == -1]

ng = NeighborhoodGraph(std_euclid_distance)
rg = RingNNG(std_euclid_distance, n_jobs=-1)

timer = time()
ng.wrap_around(X)
print(f"done in {round(time() - timer, 3)}")
timer = time()
rg.construct(X)
print(f"done in {round(time() - timer, 3)}")

