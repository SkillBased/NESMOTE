import numpy as np
from RQ.ring_query import RingQuery

from typing import Callable
from sortedcontainers import SortedSet

class RingNNG:
    def __init__(self, dist_func : Callable[..., float], k_neighbors : int = 5, n_jobs : int = 1) -> None:
        self.distance = dist_func
        self.k_neighbors = k_neighbors

        self.data = None
        self.adj_list = {}
        self.cliques = {}

        self.rq = RingQuery(self.distance)
        self.n_jobs = n_jobs
    
    def fit(self, X) -> None:
        self.data = X
        self.rq.fit(self.data, n_jobs=self.n_jobs)
        for point_id in range(self.data.shape[0]):
            self.adj_list[point_id] = self._query_record(point_id)

    def _query_record(self, point_id : int) -> None:
        neighbor_ids = self.rq.query(self.data[point_id], k=self.k_neighbors + 1)
        record = {"left": SortedSet(), "right": SortedSet()}
        for neighbor_id in neighbor_ids:
            if neighbor_id < point_id:
                record["left"].add(neighbor_id)
            elif neighbor_id > point_id:
                record["right"].add(neighbor_id)
        return record

    def clique_split(self):
        '''
            using a method provided by other research
            TBA
        '''
        for curr_vert in self.adj_list.keys():
            self.cliques[curr_vert] = []
            left_list = self.adj_list[curr_vert]["left"].copy()
            while len(left_list):
                header = left_list.pop(0)
                candidates = self.adj_list[header]["right"]
                self.update_cliques(header, curr_vert, SortedSet.intersection(left_list, candidates))
    
    def update_cliques(self, header, vertice, renovation):
        '''
            using a method provided by other research
            TBA
        '''
        if (len(self.cliques[header]) == 0 or len(renovation) == 0):
            self.cliques[header] += [SortedSet([vertice])]
            return
        perfect_match = False
        partial_set = []
        full_set = []
        for idx, clique in enumerate(self.cliques[header]):
            if clique == renovation:
                self.cliques[header][idx] = SortedSet.union(clique, SortedSet([vertice]))
                perfect_match = True
                break
            elif renovation <= clique:
                self.cliques[header] += [SortedSet.union(renovation, SortedSet([vertice]))]
                perfect_match = True
                break
            elif clique <= renovation:
                self.cliques[header][idx] = SortedSet.union(clique, SortedSet([vertice]))
                full_set += [SortedSet.intersection(clique, renovation)]
            elif len(SortedSet.intersection(clique, renovation)):
                partial_set += [SortedSet.intersection(clique, renovation)]
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
                    self.cliques[header] += [SortedSet.union(candidate, SortedSet([vertice]))]