from RQ.base import Origin, TargertedPoint
from random import shuffle
from sortedcontainers.sortedset import SortedSet

n = 256

res = [i for i in range(n)]
tps = [TargertedPoint(pid, pid / 2) for pid in range(n)]
shuffle(tps)

O = Origin(TargertedPoint(-1, 0.))
for tp in tps:
    O.add(tp)

target = TargertedPoint(-2, 117.0) # pid 234

nn_cands = O.query(target)
print(type(nn_cands), type(O.data))