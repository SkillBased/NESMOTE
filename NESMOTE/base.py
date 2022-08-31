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

        self.origins = []
        self.rings = {}

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

    
    '''
    The idea is that in naive implementation ~99% of distance calculations are done to claim
    that points are far away. Using rings we can simply remove great portion of these calculations,
    therefore significantly reducing overall complexity. Formally, the time complexity
    of all knn searches is still O(n*n), however heavy distance calculations are mostly replaced 
    by fast set operations. In euclidean spaces this can be taken one step further with use of
    grid structure instead of rings --- this removes the problem of ring population ineqality.
    '''

    def split_to_rings(self):
        # get enough origins to claim that all-rings intersection is a single connected area
        nrings = self.points.shape[1] + 1
        # choose random points for origins
        # ctatistically it is unlikely to get a bad case here
        # more importantly, getting just a bad case will not affect performance
        # problems start with about a quarter of all points within small distance
        self.origins = np.random.choice(np.arange(self.points.shape[0]), nrings)
        self.rings = {}
        # for all chosen origins
        for oid in self.origins:
            # setup an empty ring
            self.rings[oid] = {}
            # find distances from origin to all points
            O = self.points[oid]
            f_O = lambda A: self.distance(A, O) / self.limit
            origin_dists = np.apply_along_axis(f_O, 1, self.points)
            # for each point and respective distance
            for idx, distance in enumerate(origin_dists):
                # determine two rings for the point
                # this optimisation allows to skip ring joins later
                inner = floor(distance)
                outer = ceil(distance)
                # init rings if needed
                if self.rings[oid].get(inner) is None:
                    self.rings[oid][inner] = set()
                if self.rings[oid].get(outer) is None:
                    self.rings[oid][outer] = set()
                # add point to rings
                self.rings[oid][inner].add(idx)
                self.rings[oid][outer].add(idx)
                # note: it is highly inlikely to land exactly on ring border
                # even if that happens, float inacuracy almost guarrantees correctness


        
    def ring_construct(self):
        # get all rings ready
        self.split_to_rings()
        # for each point in data
        for idx, A in enumerate(self.points):
            # determine all candidates for nearest neighbors
            candidates = None
            # for all ring origins
            for origin_id in self.origins:
                # get the point ring number
                d = round(self.distance(A, self.points[origin_id]) / self.limit)
                # update candidates --- intrersect with new ring
                if candidates is None:
                    candidates = self.rings[origin_id].get(d)
                else:
                    candidates.intersection_update(self.rings[origin_id].get(d))
            # init the record
            record = {"left": SortedArray(), "right": SortedArray()}
            record["left"].reset()
            record["right"].reset()
            # process the candidates
            counted_cands = []
            f_A = lambda B_id: (self.distance(A, self.points[B_id]), B_id)
            for B_id in candidates:
                counted_cands += [f_A(B_id)]
            counted_cands.sort()
            # cut k closest points
            # for sparse sections it is intented to not fill sometimes
            if self.neighbor_cut != 0 and self.neighbor_cut < len(counted_cands):
                counted_cands = counted_cands[:self.neighbor_cut]
            for _, pid in counted_cands:
                if pid < idx:
                    record["left"].insert(pid)
                else:
                    record["right"].insert(pid)
            self.adj_list[idx] = record

    
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
            edge = min(edge, dists.shape[0] - 1)
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
        elif how == "smote":
            self.smote_split()

    def neighbor_split(self):
        '''
            make a group of all neighbors
        '''
        for curr_vert in self.adj_list.keys():
            self.cliques[curr_vert] = [deepcopy(self.adj_list[curr_vert]["left"]) + deepcopy(self.adj_list[curr_vert]["right"])]

    def smote_split(self):
        '''
            make a smote-like edge groups
        '''
        for curr_vert in self.adj_list.keys():
            self.cliques[curr_vert] = []
            for neighbor in self.adj_list[curr_vert]["left"].values:
                self.cliques[curr_vert] += [SortedArray(values=[neighbor, curr_vert], trusted=True)]

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

