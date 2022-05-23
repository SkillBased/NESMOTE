import numpy as np
from NESMOTE.util import SortedArray

from math import floor, ceil

from copy import deepcopy

from time import time

class NeighborhoodGraph:
    def __init__(self, dist_func, dataset=None, k_neighbors=5, dist_restriction=None):
        '''
            dist_func        : callable, accepts two rows of a dataset and returns float
            dataset          : np.array(), points to construct on
            k_neighbors      : int, closests points cut, 0  for no cut
            dist_restriction : float, a maximum distance for points to be considered close to each other
            
            constructs the graph automatically if dataset is provided in class constructor
        '''
        self.distance = dist_func
        self.adj_list = {}
        self.cliques = {}
        self.points = dataset
        self.neighbor_cut = k_neighbors
        self.limit = dist_restriction

        if self.points is not None:
            self.construct()
        
    def wrap_around(self, dataset, k_neighbors=5, dist_restriction=None):
        '''
            dataset          : np.array(), points to construct on
            k_neighbors      : int, closests points cut, 0  for no cut
            dist_restriction : float, a maximum distance for points to be considered close to each other
            
            constructs the graph around the dataset provided
        '''
        self.points = dataset
        self.neighbor_cut = k_neighbors
        self.limit = dist_restriction
        self.construct()
    
    def construct(self):
        '''
            constructs the graph and returns it as a dict of adjacency lists in self.adj_list
        '''
        if self.points is None:
            print("dataset is not specified")
            return
        if self.limit is None:
            self.adapt_limit()
        self.ring_construct()
        
    def ring_construct(self):
        '''
            use a ring metod to split all points into rings of self.limit width
            this lowers the amount of negative calculations significantly
            effectively bringing time complexity to O(n) precount + O(n * k) count
            where k stands for graph density and normally would be assumed O(1)
        '''
        nrings = self.points.shape[1] + 1
        # choose random points as ring origins
        origins = np.random.choice(np.arange(self.points.shape[0]), nrings)
        splits = {}
        for oid in origins:
            O = self.points[oid]
            # use lambda-apply to count distances a bit faster
            f_O = lambda A: self.distance(A, O) / self.limit
            ring_dists = np.apply_along_axis(f_O, 1, self.points)
            ring_split = {}
            # linear index run to use fast inserts 
            idx = 0
            for dist in ring_dists:
                low = floor(dist)
                high = ceil(dist)
                if ring_split.get(low) is None:
                    ring_split[low] = SortedArray()
                    ring_split[low].reset()
                if ring_split.get(high) is None:
                    ring_split[high] = SortedArray()
                    ring_split[high].reset()
                ring_split[low].insert(idx)
                ring_split[high].insert(idx)
                idx += 1
            splits[oid] = ring_split
        # linear index run for graph construction
        cnt = 0
        for A in self.points:
            # use rings to cut far away points
            candidates = None
            for origin_id in origins:
                possible = SortedArray()
                possible.reset()
                d = round(self.distance(A, self.points[origin_id]) / self.limit)
                if candidates is None:
                    candidates = splits[origin_id].get(d)
                else:
                    candidates *= splits[origin_id].get(d)
            # use lambda-apply to count distances a bit faster
            record = {"left": SortedArray(), "right": SortedArray()}
            record["left"].reset()
            record["right"].reset()
            counted_cands = []
            f_A = lambda B_id: (self.distance(A, self.points[B_id]), B_id)
            for B_id in candidates.values:
                counted_cands += [f_A(B_id)]
            counted_cands.sort()
            if self.neighbor_cut != 0 and self.neighbor_cut < len(counted_cands):
                counted_cands = counted_cands[:self.neighbor_cut]
            # cut k closest points
            # for sparse sections it is intented to not fill sometimes
            for _, pid in counted_cands:
                if pid < cnt:
                    record["left"].insert(pid)
                else:
                    record["right"].insert(pid)
            self.adj_list[cnt] = record
            cnt += 1

    
    def adapt_limit(self, nsamples=8, cut=.025):
        '''
            nsamples : int, number of points to sample
            cut      : float, 0 < cut < 1, nearby percentage

            adapts distance limit to a graph by taking samples
            for each sample all distances are count and cut percentage is taken
            average of last in lists is the adapted limit
        '''
        edge = min(self.neighbor_cut * 2, max(5, int(self.points.shape[0] * cut)))
        origins = np.random.choice(np.arange(self.points.shape[0]), nsamples)
        mean_cut = 0
        for O in self.points[origins]:
            f_O = lambda A: self.distance(A, O)
            dists = np.apply_along_axis(f_O, 1, self.points)
            mean_cut += sorted(dists)[edge]
        self.limit = mean_cut / nsamples
    
    def get_groups(self):
        '''
            safe return of constructed cliques
        '''
        return deepcopy(self.cliques)
    
    def split(self, how="cliques"):
        if how == "cliques":
            self.clique_split()
        elif how == "neighbors":
            self.neighbor_split()

    def neighbor_split(self):
        '''
            make a group of all neighbors
        '''
        for curr_vert in self.adj_list.keys():
            self.cliques[curr_vert] = [deepcopy(self.adj_list[curr_vert]["left"]) + deepcopy(self.adj_list[curr_vert]["right"])]

    def clique_split(self):
        '''
            using a method provided by other research
            TBA
        '''
        for curr_vert in self.adj_list.keys():
            self.cliques[curr_vert] = []
            left_list = deepcopy(self.adj_list[curr_vert]["left"])
            while not left_list.empty():
                header = left_list.pop()
                candidates = self.adj_list[header]["right"]
                self.update_cliques(header, curr_vert, left_list * candidates)
    
    def update_cliques(self, header, vertice, renovation):
        '''
            using a method provided by other research
            TBA
        '''
        if (len(self.cliques[header]) == 0 or renovation.empty()):
            self.cliques[header] += [SortedArray([vertice])]
            return
        perfect_match = False
        partial_set = []
        full_set = []
        for i in range(len(self.cliques[header])):
            clique = self.cliques[header][i]
            if clique == renovation:
                self.cliques[header][i] = clique + SortedArray([vertice])
                perfect_match = True
                break
            elif renovation <= clique:
                self.cliques[header] += [renovation + SortedArray([vertice])]
                perfect_match = True
                break
            elif clique <= renovation:
                self.cliques[header][i] = clique + SortedArray([vertice])
                full_set += [clique * renovation]
            elif not (clique * renovation).empty():
                partial_set += [clique * renovation]
        if not perfect_match:
            for candidate in partial_set:
                not_inferior = True
                for pretendent in partial_set:
                    if ((not candidate == pretendent) and candidate <= pretendent):
                        not_inferior = False
                        break
                for pretendent in full_set:
                    if candidate <= pretendent:
                        not_inferior = False
                        break
                if not_inferior:
                    self.cliques[header] += [candidate + SortedArray([vertice])]

