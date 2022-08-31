import numpy as np

from numpy.random import choice
from heapq import heapify, heappush

from time import time

EPS = 1e-6

class SortedArray:
    '''
        a set-like class that mantains sorted state when performing operations
        used in base.NeighborhoodGraph to provide fast set functional
    '''
    def __init__(self, values=[], trusted=False):
        '''
            values  : list
            trusted : boolean

            construct on values, sort by default
        '''
        self.values = values
        if not trusted:
            if (self.check()):
                self.values.sort()

    def check(self):
        '''
            check if the values are sorted; internal
        '''
        prev = -1
        for value in self.values:
            if value < prev:
                return True
            prev = value
        return False

    def push(self, value):
        '''
            value : comaprable to all stored values
            add a value to the end; internal
            use insert with trusted=True instead
        '''
        self.values += [value]
    
    def pop(self):
        '''
            removes the smallest element and returns it
        '''
        if len(self.values) == 0:
            return None
        return self.values.pop(0)
    
    def insert(self, value):
        '''
            value   : comaprable to all stored values

            add a value to the array maintaining order
            checks for new value being greater than all others
        '''
        if self.empty() or value > self.values[-1]:
            self.push(value)
            return
        new_values = []
        ins = False
        for v in self.values:
            if (v < value):
                new_values += [v]
            elif (v > value):
                if not ins:
                    new_values += [value]
                new_values += [v]
            else:
                return

    def empty(self):
        '''
            chesks if no elements are stored
        '''
        return len(self.values) == 0

    def reset(self):
        '''
            removes all values
        '''
        self.values = []

    def __add__(self, other):
        '''
            operator "+" override as set union
        '''
        result = SortedArray()
        result.reset()
        i, j = 0, 0
        while i < len(self.values) and j < len(other.values):
            if self.values[i] < other.values[j]:
                result.push(self.values[i])
                i += 1
            elif self.values[i] > other.values[j]:
                result.push(other.values[j])
                j += 1
            else:
                result.push(self.values[i])
                i += 1
                j += 1
        while i < len(self.values):
            result.push(self.values[i])
            i += 1
        while j < len(other.values):
            result.push(other.values[j])
            j += 1
        return result
    
    def __iadd__(self, other):
        '''
            operator "+=" override as set union
        '''
        self = self + other
        return self

    def __sub__(self, other):
        '''
            operator "-" override as left part set difference
        '''
        result = SortedArray()
        result.reset()
        i, j = 0, 0
        while i < len(self.values) and j < len(other.values):
            if self.values[i] < other.values[j]:
                result.push(self.values[i])
                i += 1
            elif self.values[i] > other.values[j]:
                j += 1
            else:
                i += 1
                j += 1
        while i < len(self.values):
            result.push(self.values[i])
            i += 1
        return result

    def __isub__(self, other):
        '''
            operator "-=" override as left part set difference
        '''
        self = self - other
        return self
    
    def __mul__(self, other):
        '''
            operator "*" override as set intersection
        '''
        result = SortedArray()
        result.reset()
        i, j = 0, 0
        while i < len(self.values) and j < len(other.values):
            if self.values[i] < other.values[j]:
                i += 1
            elif self.values[i] > other.values[j]:
                j += 1
            else:
                result.push(self.values[i])
                i += 1
                j += 1
        return result

    def __imul__(self, other):
        '''
            operator "*" override as set intersection
        '''
        self = self * other
        return self
    
    def __le__(self, other):
        '''
            operator "<=" override as subset check
        '''
        i = 0
        for value in other.values:
            if self.values[i] == value:
                i += 1
                if i >= len(self.values):
                    return True
        return False
    
    def __lt__(self, other):
        '''
            operator "<" override as size comparison
        '''
        return len(self.values) < len(other.values)
    
    def __eq__(self, other):
        '''
            operator "=" override as set equality
        '''
        return self.values == other.values


'''
    included support for R^n spaces
'''

def std_euclid_distance(a, b):
    return np.sum((a - b) ** 2) ** 0.5

def std_euclid_wavg(weights, points):
    x = None
    if np.sum(weights) != 1:
        weights = weights / np.sum(weights)
    for wi, pi in zip(weights, points):
        if x is None:
            x = pi * wi
        else:
            x += pi * wi
    return x


class IndexedCoverTree:
    def __init__(self, dist_func, data=None, max_depth=None, base=None, factor=2, fast=False):
        self.dist = dist_func
        self.max_depth = max_depth
        self.base = base
        self.factor = factor

        self.points = []
        self.tree = {-1: set()}

        if data is not None:
            if self.base is None:
                self.base = self.guess_base(data)
            if self.max_depth is None:
                self.max_depth = int(np.log2(len(data))) + 1
            for pt in data:
                if fast:
                    self.fast_insert(pt)                
                else:
                    self.insert(pt)
        
    def guess_base(self, data, nsamples=5):
        ids = choice(np.arange(len(data), dtype=int), size=min(nsamples, len(data)), replace=False)
        samples = data[ids, :]
        guess = 0
        for pt in samples:
            df = lambda x: self.dist(pt, x)
            guess += np.mean(np.apply_along_axis(df, 1, data)) / nsamples
        return guess * 1.618
            
    
    def get_children(self, parents, keep_parents=True):
        children = set()
        for parent in parents:
            children.update(self.tree[parent])
        if keep_parents:
            children.update(parents)
        children.discard(-1)
        return children

    def get_cutoff(self, parents, reference, level, keep_parents=True):
        children = set()
        for parent in parents:
            for child in self.tree[parent]:
                if self.dist(reference, self.points[child]) < (self.base / (self.factor ** level)):
                    children.add(child)
        if keep_parents:
            for parent in parents:
                if self.dist(reference, self.points[parent]) < (self.base / (self.factor ** level)):
                    children.add(parent)
        children.discard(-1)
        return children
    
    def closest_child(self, parent, reference, level, keep_parent=True):
        children = self.tree[parent]
        if keep_parent:
            children.add(parent)
        closest = self.closest_point(reference, children)
        if self.dist(reference, closest) < (self.base / (self.factor ** level)):
            return closest
        return None
    
    def cutoff(self, reference, points, level):
        cut = set()
        for pt in points:
            if self.dist(reference, self.points[pt]) < (self.base / (self.factor ** level)):
                cut.add(pt)
        return cut
    
    def closest_point(self, reference, points):
        point, dist = None, 1e9
        for pt in points:
            d = self.dist(reference, self.points[pt])
            if d < dist:
                point, dist = pt, d
        return point

    def insert(self, pt):
        pid = len(self.points)
        self.points.append(pt)
        self.tree[pid] = set()
        
        parents = [-1]
        point_parent = -1
        level = 1
        while level < self.max_depth:
#            children = self.get_children(parents)
            passed = self.get_cutoff(parents, pt, level)
            if not passed:
                point_parent = self.closest_point(pt, parents)
                break
            else:
                parents = passed
                level += 1
        else:
            point_parent = self.closest_point(pt, parents)

        self.tree[point_parent].add(pid)
        return point_parent

    def fast_insert(self, pt):
        pid = len(self.points)
        self.points.append(pt)
        self.tree[pid] = set()

        parent = -1
        level = 1
        while level < self.max_depth:
            closest = self.closest_child(parent, pt, level)
            if closest is None:
                break
            parent = closest

        self.tree[parent].add(pid)
        return parent

    def knn(self, pt, k=5):
        ok, res = self.step_knn([-1], pt, 0, 0, k + 1)
        if not ok:
            print("failed to locate knn")
        for pid in res:
            if self.dist(pt, self.points[pid]) < EPS:
                res.discard(pid)
                break
        ret = []
        for _ in range(k):
            closest = self.closest_point(pt, res)
            ret.append(closest)
            res.discard(closest)
        return ret
        
    def step_knn(self, parents, pt, cutoff_level, depth, k):

        if len(parents) > k and cutoff_level < depth:
            next_step = self.cutoff(pt, parents, cutoff_level)
            if next_step:
                ok, res = self.step_knn(next_step, pt, cutoff_level + 1, depth, k)
                if ok:
                    return ok, res
        
        if depth < self.max_depth:
            next_step = self.get_children(parents)
            ok, res = self.step_knn(next_step, pt, cutoff_level, depth + 1, k)
            if ok:
                return ok, res
        
        if len(parents) > k:
            return True, parents
        return False, set()
    
    def execute(self, data, k=5):
        for pt in data:
            self.knn(pt, k)

        

class NaiveNNG:
    def __init__(self, dist, data):
        self.distance = dist
        self.points = data
    
    def execute(self, data):
        for a in self.points:
            for b in self.points:
                self.distance(a, b)