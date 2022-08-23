import numpy as np
from random import choice

from heapq import heapify, heappush

from time import sleep

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


class CTINdexedPoint:
    def __init__(self, pid=-1, distance=0):
        self.pid = pid
        self.distance = distance
    
    def __lt__(self, other):
        return self.distance < other.distance

    def __le__(self, other):
        return self.distance <= other.distance


class CoverTree:
    def __init__(self, dist_func=None):
        self.points = []
        self.dist = dist_func
        self.covers = {-1: []}

        self.limit = 1
        self.factor = 2
    
    def distance(self, a, b):
        return self.dist(self.points[a], self.points[b])
    
    def check_terminal(self, points):
        for pt in points:
            if len(self.covers[pt]) > 1:
                return False
        return True

    def descend(self, current):
        children = []
        for parent in current:
            children.extend(self.covers[parent])
        return children
    
    def descend_to_indexed(self, current, target):
        indexed_children = []
        for parent in current:
            for child in self.covers[parent]:
                indexed_children.append(CTINdexedPoint(child, self.distance(target, child)))
        # heapq to speed up I guess
        indexed_children.sort()
        return indexed_children

    
    def run_descent(self, parents, target, cutoff, request, force_terminate=False):
        # going with linear for now, binsearch will do faster
        if self.check_terminal(parents) or force_terminate:
            if len(parents) >= request:
                return True, parents[:request]
            else:
                return False, []
        
        if len(parents) == 1 and parents[0] == target:
            return False, []

        indexed_children = self.descend_to_indexed(parents, target)
        valid = []
        full = []
        for ctip in indexed_children:
            if cutoff == 0 or ctip.distance < cutoff:
                if ctip.pid not in valid:
                    valid.append(ctip.pid)
            if ctip.pid not in full:
                full.append(ctip.pid)
        ok, res = self.run_descent(valid, target, cutoff / self.factor, request, (parents == valid))
        if ok:
            return ok, res
        else:
            ok, res = self.run_descent(full, target, 0, request, (parents == full))
            if ok:
                return ok, res
            return False, []
        
    def descend_add(self, point):
        p = len(self.points)
        self.points.append(point)
        self.covers[p] = [p]
        ok, nn = self.run_descent([-1], p, self.limit, 1)
        if ok:
            parent = nn[0]
            self.covers[parent].append(p)
            return parent
        else:
            self.covers[-1].append(p)
            return -1
    
    def descend_search(self, target, k=5):
        ok, knn = self.run_descent([-1], target, self.limit, k + 1)
        if ok:
            return knn[1:]
        else:
            return []

    def add(self, point, verbose=0):
        # set a point's id, add it to the pool and set its children list
        p = len(self.points)
        self.points.append(point)
        self.covers[p] = [p]

        # setup root as initial parent
        parents = [-1]
        # setup cutoff distance (copy hack)
        limit = 0 + self.limit

        while True:
            # descend the tree
            children = self.descend(parents)
            # cutoff the further points
            valid = []
            for c in children:
                if c != -1:
                    if self.distance(c, p) < limit and c not in valid:
                        valid.append(c)
            
            # check for loop end conditions:
            # all points are terminal (leaves)
            if (len(valid)):
                parents = valid
                if self.check_terminal(parents):
                    break
            # or there are no points meeting the cutoff limit
            else:
                break

            # reduce the cutoff limit
            limit /= self.factor
        
        # at this point any point in parents list qualifies to be new point's parent
        # choosing a random one should reduce three complexity in dense regions
        # due to the construction method this does not affect the correctness
        parent = choice(parents)
        self.covers[parent].append(p)
        return parent
        
    def get_neighbours(self, point_id, k=5, verbose=0):

        # setup root as initial parent
        parents = [-1]
        # setup cutoff distance (copy hack)
        limit = 0 + self.limit
        # setup a heap for all visited points
        step = 0
        cutoff_points = {}
        used = [False for _ in range(len(self.points))]

        while True:
            # descend the tree
            children = self.descend(parents)
            # update the point heap for all children
            cutoff_points[step] = []
            # also cutoff the further points
            valid = []
            for c in children:
                if c != -1:
                    if self.distance(c, point_id) >= limit:
                        if not used[c]:
                            cutoff_points[step].append(c)
                            used[c] = True
                    else:
                        if c not in valid:
                            valid.append(c)
            
            # check for loop end conditions:
            # all points are terminal (leaves)
            if (len(valid)):
                parents = valid
                if self.check_terminal(parents):
                    break
            # or there are no points meeting the cutoff limit
            else:
                break

            # reduce the cutoff limit
            limit /= self.factor
            pts = sorted([CTINdexedPoint(cutpt, self.distance(cutpt, point_id)) for cutpt in cutoff_points[step]])
            cutoff_points[step] = pts
            step += 1
        
        # by this point heap contains all visited points sorted from closest to furthest
        # while this may not be entirely accurate for select few occurances
        # most of the time first k + 1 heap elements are in fact given point and its KNN
        cutoff_points[step] = sorted([CTINdexedPoint(cutpt, self.distance(cutpt, point_id)) for cutpt in parents])
         
        knn = []
        while step > 0 and len(knn) < k:
            for ctip in cutoff_points[step]:
                pid = ctip.pid
                if pid != point_id:
                    knn.append(pid)
                if len(knn) >= k:
                    break
            step -= 1

        return knn

    