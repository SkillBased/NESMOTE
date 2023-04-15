import numpy as np
from swg import HNSW
from swg2 import HeuristicHNSW
from NESMOTE.util import std_euclid_distance
from random import randint, choice
from math import e, sqrt
from time import time

tasks = [2 ** x for x in range(7, 9)]
nruns = 5

res1 = []
res2 = []

for i, task in enumerate(tasks):
    res1.append([])
    res2.append([])
    constr, query = 0, 0
    for _ in range(nruns):
        start = time()
        data = np.array([np.array([randint(-1024, 1024), randint(-1024, 1024)]) for _ in range(task)])
        graph = HNSW(std_euclid_distance)
        graph.fit(data)
        constr += round(time() - start, 3)
        start = time()
        for pt in data:
            graph.query(pt)
        query += round(time() - start, 3)
        res1[i].append(constr)
        res2[i].append(query)
        start = time()
        data = np.array([np.array([randint(-1024, 1024), randint(-1024, 1024)]) for _ in range(task)])
        graph = HeuristicHNSW(std_euclid_distance)
        graph.fit(data)
        constr += round(time() - start, 3)
        start = time()
        for pt in data:
            graph.descend_search(pt)
        query += round(time() - start, 3)
        res1[i].append(constr)
        res2[i].append(query)
    print(f"run {task} done")

print("{:10} ".format("construct"), end="")
factors = ["old", "new"]
for j, factor in enumerate(factors):
    print("{:>10} ".format(factor), end="")
print()
for i, task in enumerate(tasks):
    print("{:10} ".format(task), end="")
    for j, factor in enumerate(factors):
        print("{:10.3f} ".format(res1[i][j]), end="")
    print()

print()

print("{:10} ".format("query"), end="")
for j, factor in enumerate(factors):
    print("{:>10} ".format(factor), end="")
print()
for i, task in enumerate(tasks):
    print("{:10} ".format(task), end="")
    for j, factor in enumerate(factors):
        print("{:10.3f} ".format(res2[i][j]), end="")
    print()