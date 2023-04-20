import numpy as np
from NESMOTE.util import SortedArray

from math import floor, ceil

from copy import deepcopy

from typing import Callable, Optional

from heapq import nsmallest

class RingQuery:
    def __init__(self, distance : Callable[..., float]) -> None:
        self.distance = distance
        self.data = None
        self.ring_width = None
        self.origin_ids = set()
        self.rings = {}
    
    def _adapt_width(self, nsamples : int = 8, cut : float = .01) -> float:
        if self.data is None:
            return
        edge = min(self.data.shape[0], max(16, int(self.data.shape[0] * cut)))
        origins = np.random.choice(np.arange(self.data.shape[0]), nsamples)
        edges = [self._adapt_single(origin_id, edge) for origin_id in origins]
        return np.mean(edges)
    
    def _adapt_single(self, origin_id : int, edge : int) -> Optional[float]:
        origin_distance = lambda point: self.distance(self.data[origin_id], point)
        distances = np.apply_along_axis(origin_distance, 1, self.data)
        smallest = nsmallest(edge, distances)
        return smallest[-1]

    def fit(self, X, n_origins : Optional[int] = None, width : Optional[float] = None, n_jobs : int = 1) -> None:
        self.data = X
        origin_count = n_origins if n_origins is not None else self.data.shape[1] + 1
        self.origin_ids = np.random.choice(np.arange(self.data.shape[0]), origin_count)
        self.ring_width = width if width is not None else self._adapt_width()
        self.rings = {}
        for origin_id in self.origin_ids:
            self.rings[origin_id] = self._process_single(origin_id)

    def _process_single(self, origin_id : int) -> dict[int, set[int]]:
        origin_distance = lambda point: self.distance(self.data[origin_id], point)
        ring = {}
        distances = np.apply_along_axis(origin_distance, 1, self.data)
        for idx, dist in enumerate(distances):
            inner, outer = floor(dist / self.ring_width), ceil(dist / self.ring_width)
            if inner == outer:
                outer += 1
            if ring.get(inner) is None:
                ring[inner] = set()
            if ring.get(outer) is None:
                ring[outer] = set()
            ring[inner].add(idx)
            ring[outer].add(idx)
        return ring


    def query(self, point, k : int = 5, return_raw : bool = False):
        point_distance = lambda point_id: (self.distance(self.data[point_id], point), point_id)
        candidates = None
        for origin_id in self.origin_ids:
            d = round(self.distance(point, self.data[origin_id]) / self.ring_width)
            if candidates is None:
                candidates = self.rings[origin_id].get(d)
            else:
                candidates.intersection_update(self.rings[origin_id].get(d))
        processed = []
        for point_id in candidates:
            processed.append(point_distance(point_id))
        processed.sort()
        if k != 0 and k < len(processed):
            processed = processed[:k]
        result = [neighbor[1] for neighbor in processed]
        if return_raw:
            return self.data[result]
        return result

class NNG:
    def __init__(self, dist_func : Callable[..., float], k_neighbors : int = 5, n_jobs : int = 1) -> None:
        self.distance = dist_func
        self.k_neighbors = k_neighbors

        self.data = None
        self.adj_list = {}
        self.cliques = {}

        self.rq = RingQuery(self.distance)
        self.n_jobs = n_jobs
    
    def construct(self, X) -> None:
        self.data = X
        self.rq.fit(self.data, n_jobs=self.n_jobs)
        for point_id in range(self.data.shape[0]):
            self.adj_list[point_id] = self._query_record(point_id)

    def _query_record(self, point_id : int) -> None:
        neighbor_ids = self.rq.query(self.data[point_id], k=self.k_neighbors + 1)
        record = {"left": SortedArray(), "right": SortedArray()}
        record["left"].reset()
        record["right"].reset()
        for neighbor_id in neighbor_ids:
            if neighbor_id < point_id:
                record["left"].insert(neighbor_id)
            elif neighbor_id > point_id:
                record["right"].insert(neighbor_id)
        return record
    
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