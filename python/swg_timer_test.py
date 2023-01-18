import numpy as np
from swg import HNSW
from NESMOTE.util import std_euclid_distance
from random import randint, choice
from math import e, sqrt
from time import time

tasks = [128, 2048, 65536]
factors = [2, e, 4]
nruns = 3

res1 = []
res2 = []

for i, task in enumerate(tasks):
    res1.append([])
    res2.append([])
    for factor in factors:
        constr, query = 0, 0
        for _ in range(nruns):
            start = time()
            data = np.array([np.array([randint(6, 1018), randint(6, 506)]) for _ in range(task)])
            graph = HNSW(std_euclid_distance)
            graph.fit(data)
            constr += round(time() - start, 3)
            start = time()
            for pt in data:
                graph.query(pt)
            query += round(time() - start, 3)
        res1[i].append(constr)
        res2[i].append(query)
        print(f"run {task}/#/{round(factor, 3)} done")

print("{:10} ".format(""), end="")
for j, factor in enumerate(factors):
    print("{:10.4f} ".format(factor), end="")
print()
for i, task in enumerate(tasks):
    print("{:10} ".format(task), end="")
    for j, factor in enumerate(factors):
        print("{:10.3f} ".format(res1[i][j]), end="")
    print()

print()

print("{:10} ".format(""), end="")
for j, factor in enumerate(factors):
    print("{:10.4f} ".format(factor), end="")
print()
for i, task in enumerate(tasks):
    print("{:10} ".format(task), end="")
    for j, factor in enumerate(factors):
        print("{:10.3f} ".format(res2[i][j]), end="")
    print()