import numpy as np
from tkinter import Tk, Canvas

from NESMOTE.base import RingNNG
from NESMOTE.util import std_euclid_distance

from random import randint, choice

root = Tk()

c = Canvas(width=1024, height=512, bg="#1e1e1e")
c.pack()

nsamples = 1024
data = np.array([np.array([randint(6, 1018), randint(6, 506)]) for _ in range(nsamples)])


graph = RingNNG(std_euclid_distance)
graph.fit(data)
graph.clique_split()
print(graph.cliques)

for point in data:
    x, y = point
    c.create_oval(x - 2, y - 2, x + 2, y + 2, fill="#007e00", outline="")

lines = []

def step(event):
    print("stepping")
    global current_idx, lines, c, graph

    for line in lines:
        c.delete(line)
    lines = []

    i = randint(0, len(data) - 1)
 
    point = data[i]
    x, y = point
    lines.append(c.create_oval(x - 2, y - 2, x + 2, y + 2, fill="#7e0000", outline=""))
    neighbors = graph.cliques[i][0]
    for neighbor in neighbors:
        x1, y1 = data[neighbor]
        lines.append(c.create_line(x, y, x1, y1, fill="#00007e"))
    return
    

c.bind("<Button-1>", step)
root.mainloop()