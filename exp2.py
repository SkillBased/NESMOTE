import numpy as np

from imblearn.datasets import fetch_datasets
from imblearn.under_sampling import AllKNN

from NESMOTE.base import NeighborhoodGraph
from NESMOTE.util import IndexedCoverTree, NaiveNNG, std_euclid_distance

from time import time


def run_trial(data, target, goal=-1):

    print("imblearn ann .. ", end="", flush=True)

    timer = time()
    imb_knn = AllKNN()
    imb_knn.fit_resample(data, target)
    imb_time = time() - timer

    print("done, ring method .. ", end="", flush=True)

    timer = time()
    ring = NeighborhoodGraph(std_euclid_distance, data[target == goal])
    ring.neighbor_split()
    ring_time = time() - timer

    print("done, cover tree .. ", end="", flush=True)

    timer = time()
    ct = IndexedCoverTree(std_euclid_distance, data[target == goal])
    ct.execute(data[target == goal])
    ct_time = time() - timer

    print("done, naive search .. ", end="", flush=True)

    timer = time()
    naive = NaiveNNG(std_euclid_distance, data[target == goal])
    naive.execute(data[target == goal])
    naive_time = time() - timer

    print("done", flush=True)

    return imb_time, ring_time, ct_time, naive_time



datasets = fetch_datasets()

ds_names = ["ecoli", "optical_digits", "abalone", "sick_euthyroid", "spectrometer", "car_eval_34",
            "us_crime", "yeast_ml8", "car_eval_4", "thyroid_sick", "wine_quality", "solar_flare_m0",
            "yeast_me2", "ozone_level", "abalone_19"]

ds_names = ["ecoli"]

outfile = open("timers.out", "w")

outfile.write("{:15}    {:12}    {:11}      {:10}    {:12}\n".format("dataset", "imblearn ann", "ring method", "cover tree", "naive search"))
for name in ds_names:
    X, y = datasets[name].data, datasets[name].target
    print(f"running trial {name}: ", end="", flush=True)
    ann, ring, ct, naive = run_trial(X, y)
    outfile.write("{:15} {:15.4f} {:15.4f} {:15.4f} {:15.4f}\n".format(name, ann, ring, ct, naive))
    print(f"trial {name} completed in {round(ann + ring + ct + naive, 3)}s", flush=True)

outfile.close()