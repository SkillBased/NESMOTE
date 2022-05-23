import numpy as np

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
