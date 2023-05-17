import numpy as np
from NESMOTE.util import SortedArray

from math import floor, ceil

from copy import deepcopy

from typing import Callable, Optional

from heapq import nsmallest

from NESMOTE.neighbors import RingQuery


class NNG:
    '''
    This is the nearest neighbor graph implementation based on RingQuery.
    It is also structured specifically to be used in the fast clique search as the method requires it.
    Wierdly enough, running heapq for sorted arrays is slower that custom class.
    '''
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
        Python is guilty of returning links instead of values for complex structures.
        `deepcopy` overrides this behavior for a data-safe return.
        '''
        return deepcopy(self.cliques)
    
    def split(self, how="cliques"):
        '''
        This function returns augmentation groups based on the constructed NNG and the method of choice.
        Note that this does not perform augmentations, just returns all edges.
        '''
        if how == "cliques":
            self.clique_split()
        elif how == "neighbors":
            self.neighbor_split()
        elif how == "smote":
            self.smote_split()

    def smote_split(self):
        '''
        The standard SMOTE-like approach that returns NNG edges.
        Both fastest and least quality.
        '''
        for curr_vert in self.adj_list.keys():
            self.cliques[curr_vert] = []
            for neighbor in self.adj_list[curr_vert]["left"].values:
                self.cliques[curr_vert] += [SortedArray(values=[neighbor, curr_vert], trusted=True)]

    def neighbor_split(self):
        '''
        Returns KNN groups for each vertice.
        A compromise approach that allows for better accuracy than SMOTE-lile and better complexity that cliques.
        '''
        for curr_vert in self.adj_list.keys():
            self.cliques[curr_vert] = [deepcopy(self.adj_list[curr_vert]["left"]) + deepcopy(self.adj_list[curr_vert]["right"])]


    def clique_split(self):
        '''
        Returns maximal cliques from NNG as augmentation groups.
        Hihgest accuracy and complexity.
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
        This is taken from the algorithm you can read about at
        https://www.researchgate.net/publication/317130248_A_linear_time_algorithm_for_maximal_clique_enumeration_in_large_sparse_graphs
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