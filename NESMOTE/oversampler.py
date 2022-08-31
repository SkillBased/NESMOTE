import numpy as np
from numpy.random import choice, randint
from NESMOTE.base import NeighborhoodGraph, SortedArray
from NESMOTE.util import IndexedCoverTree

from random import random, gammavariate

from joblib import Parallel, delayed


#
from time import time
#

'''

parametrers - augmentation:

-> strategy
-- "resample"  : regenerate all classes with same amounts of points as in the given dataset
-- "rebalance" : regenerate all classes with equal amount of points in each
-- "upscale"   : keep all points and generate new to equalize amoint of points in each class

-> weights
-- "standard" : more centered distribution
-- "gamma"    : more even distribution

parametrers - neighborhood graph:

-> k-neighbors
-- int, number of closest neighbors in the graph construction

-> distance (optional)
-- float > 0, maximal distance for a point to be considered a neighbor

parametrers - augmentation groups:

-> groupby
-- "neighbors" : all closest neighbors for each point form a group
-- "cliques"   : find maximal cliques

-> wrap
-- all : all maximal cliques are used
-- max : only one greatest clique is used

'''

class NESMOTE:
    def __init__(self, dist_func, sampling_func, params={}):
        self.distance = dist_func
        self.group_sampler = sampling_func

        self.parameters = params
        self.samplers = []
        self.input_sizes = {}
        self.dominant_size = 0

    def fit(self, X, y):
        '''
            X : array-like of shape ..., n
            y : array-like of shape n

            fit the augmenter to the data given
        '''
        self.samplers = []
        for value, count in np.transpose(np.unique(y, return_counts=True)):
            index = (y == value)
            class_pts = X[index]
            self.input_sizes[value] = count
            if count > self.dominant_size:
                self.dominant_size = count
            # construct class sampler
            smp = ClassSampler(value, class_pts, self.parameters, self.distance, self.group_sampler)
            self.samplers.append(smp)

    def resample(self, X, y):
        '''
            X : array-like of shape ..., n
            y : array-like of shape n

            create new points
        '''
        nX = None
        ny = None

        strategy = self.parameters.get("strategy") if self.parameters.get("strategy") is not None else "upscale"

        weighter = self.parameters.get("weights") if self.parameters.get("weights") is not None else "standard"

        if strategy == "resample":
            for sampler in self.samplers:
                sampler.prep()
                n = self.input_sizes[sampler.pt_class]
                sX = sampler.generate(n, weighter)
                sy = np.array([sampler.pt_class] * n)
                if nX is None:
                    nX = sX
                    ny = sy
                else:
                    nX = np.vstack([nX, sX])
                    ny = np.hstack([ny, sy])

        if strategy == "rebalance":
            n = int(y.shape[0] / len(self.samplers))
            for sampler in self.samplers:
                sampler.prep()
                sX = sampler.generate(n, weighter)
                sy = np.array([sampler.pt_class] * n)
                if nX is None:
                    nX = sX
                    ny = sy
                else:
                    nX = np.vstack([nX, sX])
                    ny = np.hstack([ny, sy])

        if strategy == "upscale":
            nX = X
            ny = y
            for sampler in self.samplers:
                n = self.dominant_size - self.input_sizes[sampler.pt_class]
                if (n < 1):
                    continue
                sampler.prep()
                sX = sampler.generate(n, weighter)
                sy = np.array([sampler.pt_class] * n)
                if nX is None:
                    nX = sX
                    ny = sy
                else:
                    nX = np.vstack([nX, sX])
                    ny = np.hstack([ny, sy])

        return nX, ny

    def fit_resample(self, X, y):
        self.fit(X, y)
        P, q = self.resample(X, y)
        return P, q


class ClassSampler:
    def __init__(self, pt_class, points, parameters, dist_func, sampler_func):
        self.groups = []

        self.points = points
        self.pt_class = pt_class
        self.count = self.points.shape[0]

        self.parameters = parameters

        self.distance = dist_func
        self.sampler = sampler_func
    
    def prep(self):
        # setup graph args
        k_neighbors = self.parameters.get("k-neighbors") if self.parameters.get("k-neighbors") is not None else 5
        dist_restriction = self.parameters.get("distance")
        # fit a graph to class and split it
        class_ng = NeighborhoodGraph(self.distance, self.points, k_neighbors, dist_restriction)
        split_type = self.parameters.get("groupby") if self.parameters.get("groupby") is not None else "cliques"
        class_ng.split(how=split_type)
        cutoff = self.parameters.get("group-cut") if self.parameters.get("group-cut") is not None else 0
        wrap = self.parameters.get("wrap") if self.parameters.get("wrap") is not None else "max"
        if wrap == "all":
            self.full_wrap(class_ng.get_groups(), cutoff)
        else:
            self.max_wrap(class_ng.get_groups())
    
    def max_wrap(self, cliques):
        for pid in range(self.count):
            p_cliques = cliques[pid]
            if len(p_cliques) > 0:
                maximal = max(p_cliques).values
                maximal += [pid]
                self.groups.append(maximal)
            else:
                self.groups.append([pid])

    def full_wrap(self, cliques, cutoff=3):
        maxes = []
        for pid in range(self.count):
            p_cliques = cliques[pid]
            if len(p_cliques) > 0:
                for clique in p_cliques:
                    if len(clique.values) <= cutoff:
                        continue
                    inferior = False
                    remove = [False] * len(maxes)
                    full_clique = SortedArray([pid]) + clique
                    for i, candidate in enumerate(maxes):
                        if full_clique <= candidate:
                            inferior = True
                        if candidate <= full_clique:
                            remove[i] = True
                    new_maxes = []
                    for i, elem in enumerate(maxes):
                        if not remove[i]:
                            new_maxes += [elem]
                    if not inferior:
                        new_maxes += [full_clique]
                    maxes = new_maxes
        for clique in maxes:
            body = clique.values
            self.groups.append(body)

    def generate(self, n_points, weighter="standard"):
        res = []
        group_ids = np.random.choice(np.arange(0, len(self.groups)), size=n_points, replace=True)
        for idx in group_ids:
            aug_points = self.points[self.groups[idx]]
            if weighter == "gamma":
                aug_weights = np.array(adaptive_gamma_picker(len(aug_points)))
            else:
                aug_weights = np.array(standard_picker(len(aug_points)))
            x = self.sampler(aug_weights, aug_points)
            res.append(x)
        return np.array(res)

def adaptive_gamma_picker(n):
    weights = []
    weight_sum = 0
    for i in range(n):
        new_weight = abs(gammavariate(2 / n, 2 ** n))
        weights.append(new_weight)
        weight_sum += new_weight
    for i in range(n):
        weights[i] = weights[i] / weight_sum
    return weights

def standard_picker(n):
    weights = []
    weight_sum = 0
    for i in range(n):
        new_weight = random()
        weights.append(new_weight)
        weight_sum += new_weight
    for i in range(n):
        weights[i] = weights[i] / weight_sum
    return weights

class Oversampler:
    def __init__(self, data, pt_class, dist, sample, params={}) -> None:
        self.data = data
        self.pt_class = pt_class

        self.dist = dist
        self.sample = sample

        self.params = params
        self.mode = params.get("mode") if params.get("mode") is not None else "ND-SMOTE"

        self.cover_tree = IndexedCoverTree(self.dist, self.data)
    
    def generate_smote(self, n_points, n_jobs):
        points = Parallel(n_jobs=n_jobs)(delayed(self.get_smote_point)() for _ in range(n_points))
        return points
    
    def get_smote_point(self, edge=False):
        origin = self.data[randint(0, len(self.data))]
        k = self.params.get("k-neighbors") if self.params.get("k-neighbors") is not None else 5
        knn = self.cover_tree.knn(origin, k)
        group = [self.data[choice(knn)], origin]

        if not edge:
            group = self.data[knn]
        
        weights = standard_picker(len(group))

        return self.sample(weights, group)


class FastNESMOTE:
    def __init__(self, dist, sampler):
        self.distance = dist
        self.sample = sampler

        self.input_sizes = {}
        self.majority_size = 0

    def fit(self, X, y):
        '''
            X : array-like of shape ..., n
            y : array-like of shape n

            fit the augmenter to the data given
        '''
        self.samplers = []
        for value, count in np.transpose(np.unique(y, return_counts=True)):
            timer = time()
            class_pts = X[y == value]
            self.input_sizes[value] = count
            if count > self.majority_size:
                self.majority_size = count
            # construct class sampler
            smp = Oversampler(class_pts, value, self.distance, self.sample)
            self.samplers.append(smp)
        
    def resample(self, X, y):
        nX = X
        ny = y
        for sampler in self.samplers:
            n = self.majority_size - self.input_sizes[sampler.pt_class]
            if (n < 1):
                continue
            sX = sampler.generate_smote(n, n_jobs=8)
            sy = np.array([sampler.pt_class] * n)
            if nX is None:
                nX = sX
                ny = sy
            else:
                nX = np.vstack([nX, sX])
                ny = np.hstack([ny, sy])
        return nX, ny
   
    def fit_resample(self, X, y):
        self.fit(X, y)
        P, q = self.resample(X, y)
        return P, q 

    
