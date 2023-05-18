from geomstats.geometry.hypersphere import Hypersphere
from geomstats.learning.frechet_mean import FrechetMean
import geomstats.visualization as visualization

import numpy as np

from NESMOTE.oversample import Pipeline, MCProcessor, StandardSampler
from NESMOTE.neighbors import RingQuery

sphere = Hypersphere(dim=2)

batch = 100
data = sphere.random_uniform(n_samples=batch)
y = np.array([0] * batch)

metric = sphere.metric
mean = FrechetMean(metric)

def dist_func(a, b):
    global metric
    return metric.dist(a, b)

def wavg_func(weights, points):
    global mean
    mean.fit(X=points, weights=weights)
    return mean.estimate_

params = {
    "strategy" : "resample"
}

augmenter = Pipeline(RingQuery(dist_func), MCProcessor(), StandardSampler(wavg_func))
nX, ny = augmenter.fit_sample(data, y)

visualization.plot(data, space="S2")