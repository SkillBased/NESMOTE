import numpy as np
from tkinter import Tk, Canvas

from NESMOTE.util import std_euclid_distance

from NESMOTE.neighbors import RingQuery

from random import randint, choice

root = Tk()

c = Canvas(width=1024, height=512, bg="#1e1e1e")
c.pack()

nsamples = 1024
data = np.array([np.array([randint(6, 1018), randint(6, 506)]) for _ in range(nsamples)])

nsw = RingQuery(std_euclid_distance)
nsw.fit(data)
xnt = 0
for idx in range(nsamples):
    if len(nsw.query(data[idx])) < 1:
        xnt += 1
print(xnt)

for idx, point in enumerate(data):
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
    res = [data[pt] for pt in nsw.query(point, k=8)]
    for x1, y1 in res:
#        x1, y1 = data[pid]
        lines.append(c.create_oval(x1 - 2, y1 - 2, x1 + 2, y1 + 2, fill="#00aeae", outline=""))
    lines.append(c.create_oval(x - 2, y - 2, x + 2, y + 2, fill="#ae0000", outline=""))
    return
    

c.bind("<Button-1>", step)
root.mainloop()