import numpy as np
from tkinter import Tk, Canvas

from swg import SmallWorld, HNSW
from NESMOTE.util import std_euclid_distance

from RQ.neighbors import RingQuery, SortedSetQuery, TargertedPoint

from random import randint, choice

root = Tk()

c = Canvas(width=1024, height=512, bg="#1e1e1e")
c.pack()

nsamples = 1024
data = np.array([np.array([randint(6, 1018), randint(6, 506)]) for _ in range(nsamples)])


ssq = SortedSetQuery(std_euclid_distance)
ssq.fit(data)

for point in data:
    x, y = point
    c.create_oval(x - 2, y - 2, x + 2, y + 2, fill="#007e00", outline="")


lines = []

def step(event):
    print("stepping -- ", end="")
    global current_idx, lines, c, graph

    for line in lines:
        c.delete(line)
    lines = []
 
    point = choice(data)
    x, y = point
    origin = ssq.origins[0]
    target = TargertedPoint(-1, std_euclid_distance(point, origin.point))
    origin_cands = set(origin.query(target, extend=50))
    for pid in origin_cands:
        x1, y1 = data[pid]
        lines.append(c.create_oval(x1 - 2, y1 - 2, x1 + 2, y1 + 2, fill="#00007e", outline=""))
    res = ssq.query(point)
    print(len(res))
    for x1, y1 in res:
#        x1, y1 = data[pid]
        lines.append(c.create_oval(x1 - 2, y1 - 2, x1 + 2, y1 + 2, fill="#007e7e", outline=""))
    lines.append(c.create_oval(x - 2, y - 2, x + 2, y + 2, fill="#7e0000", outline=""))
    return
    

c.bind("<Button-1>", step)
root.mainloop()