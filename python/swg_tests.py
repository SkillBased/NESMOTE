import numpy as np
from tkinter import Tk, Canvas

from swg2 import HeuristicHNSW
from NESMOTE.util import std_euclid_distance

from RQ.neighbors import RingQuery, SortedSetQuery, TargertedPoint

from random import randint, choice

root = Tk()

c = Canvas(width=1024, height=512, bg="#1e1e1e")
c.pack()

nsamples = 1024
data = np.array([np.array([randint(6, 1018), randint(6, 506)]) for _ in range(nsamples)])

nsw = HeuristicHNSW(std_euclid_distance)

for idx, point in enumerate(data):
    nsw.add(point)
    x, y = point
    c.create_oval(x - 2, y - 2, x + 2, y + 2, fill="#007e00", outline="")


lines = []
idx_ = -1

def step(event):
    global idx_
    print("stepping -- ", end="")
    global current_idx, lines, c, graph

    for line in lines:
        c.delete(line)
    lines = []
 
    point = data[idx_]
    idx_ -= 1
    x, y = point
    print(x, y)
    res = [data[pt] for pt in nsw.descend_search(point, k=12)]
    for x1, y1 in res:
#        x1, y1 = data[pid]
        lines.append(c.create_oval(x1 - 2, y1 - 2, x1 + 2, y1 + 2, fill="#00aeae", outline=""))
    lines.append(c.create_oval(x - 2, y - 2, x + 2, y + 2, fill="#ae0000", outline=""))
    return
    

c.bind("<Button-1>", step)
root.mainloop()