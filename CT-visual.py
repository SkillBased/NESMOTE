import numpy as np

from tkinter import Tk, Canvas

from NESMOTE.util import CoverTree, std_euclid_distance

from random import randint

from time import time

root = Tk()

c = Canvas(width=1024, height=512, bg="#1e1e1e")
c.pack()

c.create_line(512, 0, 512, 512, fill="yellow", width=4)

nsamples = 512
left = []
right = []
shift = np.array([512, 0])
for _ in range(nsamples):
    lpt = np.array([randint(6, 506), randint(6, 506)])
    rpt = lpt + shift
    left.append(lpt)
    right.append(rpt)

for x, y in left:
    c.create_oval(x - 2, y - 2, x + 2, y + 2, fill="#007e00", outline="")
for x, y in right:
    c.create_oval(x - 2, y - 2, x + 2, y + 2, fill="#007e00", outline="")

ntargets = 8
targets = [randint(0, nsamples - 1) for _ in range(ntargets)]

timer = time()
LCT = CoverTree(std_euclid_distance)
LCT.limit = 128
idx = 0
for pt in left:
    parent = LCT.add(pt)
    if parent != -1:
        x0, y0 = left[parent]
        x1, y1 = pt
        c.create_line(x0, y0, x1, y1, fill="#005e00", width=1)
    idx += 1
print(f"regular constructed in {round(time() - timer, 2)}")
timer = time()
for target in targets:
    knn = LCT.get_neighbours(target)
    for pid in knn:
        if pid != target:
            x0, y0 = left[target]
            x1, y1 = left[pid]
            c.create_line(x0, y0, x1, y1, fill="#00005e", width=1)
print(f"regular done in {round(time() - timer, 2)}")

'''
timer = time()
RCT = CoverTree(std_euclid_distance)
RCT.limit = 128
idx = 0
for pt in right:
    parent = RCT.descend_add(pt)
    if parent != -1:
        x0, y0 = right[parent]
        x1, y1 = pt
        c.create_line(x0, y0, x1, y1, fill="#005e00", width=1)
    idx += 1
print(f"descend constructed in {round(time() - timer, 2)}")
timer = time()
knn = RCT.descend_search(target)
for pid in knn:
    if pid != target:
        x0, y0 = right[target]
        x1, y1 = right[pid]
        c.create_line(x0, y0, x1, y1, fill="#00005e", width=1)
print(f"descend done in {round(time() - timer, 2)}")
'''

timer = time()
RCT = CoverTree(std_euclid_distance)
RCT.limit = 128
idx = 0
for pt in right:
    parent = RCT.add(pt)
    if parent != -1:
        x0, y0 = right[parent]
        x1, y1 = pt
        c.create_line(x0, y0, x1, y1, fill="#005e00", width=1)
    idx += 1
print(f"mixed constructed in {round(time() - timer, 2)}")
timer = time()
for target in targets:
    knn = RCT.descend_search(target)
    for pid in knn:
        if pid != target:
            x0, y0 = right[target]
            x1, y1 = right[pid]
            c.create_line(x0, y0, x1, y1, fill="#00005e", width=1)
print(f"mixed done in {round(time() - timer, 2)}")




root.mainloop()