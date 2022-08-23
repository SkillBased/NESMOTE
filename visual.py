from tkinter import *

from time import time
from sklearn.datasets import make_moons
from imblearn.datasets import make_imbalance

from NESMOTE.util import std_euclid_distance, std_euclid_wavg
from NESMOTE.oversampler import NESMOTE

root = Tk()
root.title("NESMOTE augmentation strategies")

c = Canvas(width=1200, height=800, bg="#1e1e1e")
c.pack()

c.create_line(600, 0, 600, 800, fill="white")
c.create_line(0, 400, 1200, 400, fill="white")

c.create_text(300, 20, anchor="center", text="original", font="Courier 12", fill='White')
c.create_text(300, 35, anchor="center", text="make_moons + make_imbalance 4:1", font="Courier 10", fill='White')

c.create_text(900, 20, anchor="center", text="rebalanced", font="Courier 12", fill='White')
c.create_text(900, 35, anchor="center", text="same points total, balanced classes", font="Courier 10", fill='White')

c.create_text(300, 420, anchor="center", text="resampled", font="Courier 12", fill='White')
c.create_text(300, 435, anchor="center", text="same points total, same classes", font="Courier 10", fill='White')

c.create_text(900, 420, anchor="center", text="upscaled", font="Courier 12", fill='White')
c.create_text(900, 435, anchor="center", text="more points total, balanced classes", font="Courier 10", fill='White')

def transfer_point(x, y, quarter):
    px = (x + 2) * 120
    py = (y + 1.5) * 120
    px += 600 * (quarter % 2)
    py += 400 * int(quarter / 2)
    return px, py


count = 1000
oX, oy = make_moons(count, shuffle=True, noise=.1, random_state=42)
pts, y = make_imbalance(oX, oy, sampling_strategy={0 : 400, 1 : 100})

rebalance = {
    "strategy" : "rebalance",
}
resample = {
    "strategy" : "resample",
}
upscale = {
    "strategy" : "upscale",
}

reb = NESMOTE(std_euclid_distance, std_euclid_wavg, rebalance)
rebX, reby = reb.fit_resample(pts, y)

res = NESMOTE(std_euclid_distance, std_euclid_wavg, resample)
resX, resy = res.fit_resample(pts, y)

ups = NESMOTE(std_euclid_distance, std_euclid_wavg, upscale)
upsX, upsy = ups.fit_resample(pts, y)

pts_colors = ["#5e0000", "#005e00", "#00005e"]
aux_colors = ["#ae0000", "#00ae00", "#0000ae"]

for point, pt_class in zip(pts, y):
    x, y = transfer_point(point[0], point[1], 0)
    c.create_oval(x-2, y-2, x+2, y+2, fill=aux_colors[pt_class], outline="")

for point, pt_class in zip(rebX, reby):
    x, y = transfer_point(point[0], point[1], 1)
    c.create_oval(x-2, y-2, x+2, y+2, fill=aux_colors[pt_class], outline="")

for point, pt_class in zip(resX, resy):
    x, y = transfer_point(point[0], point[1], 2)
    c.create_oval(x-2, y-2, x+2, y+2, fill=aux_colors[pt_class], outline="")

for point, pt_class in zip(upsX, upsy):
    x, y = transfer_point(point[0], point[1], 3)
    c.create_oval(x-2, y-2, x+2, y+2, fill=aux_colors[pt_class], outline="")

root.mainloop()