import numpy as np

from time import time
from NESMOTE.oversampler import FastNESMOTE, NESMOTE
from sklearn.datasets import make_moons

from imblearn.datasets import fetch_datasets

from NESMOTE.util import std_euclid_distance, std_euclid_wavg
from imblearn.over_sampling import SMOTE

from tkinter import Tk, Canvas

nsamples = 400

sets = fetch_datasets()
dataset = sets["scene"]
X, y = dataset.data, dataset.target
print(len(X))

os = FastNESMOTE(std_euclid_distance, std_euclid_wavg)
timer = time()
os.fit(X, y)
print(f"constuction in {round(time() - timer, 2)}")
timer = time()
nX, ny = os.resample(X, y)  
print(f"generation in {round(time() - timer, 2)}")

os = NESMOTE(std_euclid_distance, std_euclid_wavg)
timer = time()
os.fit(X, y)
print(f"constuction in {round(time() - timer, 2)}")
timer = time()
nX, ny = os.resample(X, y)  
print(f"generation in {round(time() - timer, 2)}")
