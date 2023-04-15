from typing import Callable, List, Dict, Tuple, Optional
from math import floor, log, e
from heapq import heapify, heappush, heappop, heappushpop
from random import choice
from copy import deepcopy

class HeuristicHNSW:
    class ClosestOut:
        def __init__(self, point_id : int, target_distance : float) -> None:
            self.point_id = point_id
            self.target_distance = target_distance

        def __lt__(self, other) -> bool:
            if self.target_distance == other.target_distance:
                return self.point_id < other.point_id
            return self.target_distance < other.target_distance

    class FurthestOut:
        def __init__(self, point_id : int, target_distance : float) -> None:
            self.point_id = point_id
            self.target_distance = target_distance

        # operator is reversed to work with heappushpop
        def __lt__(self, other) -> bool:
            if self.target_distance == other.target_distance:
                return self.point_id > other.point_id
            return self.target_distance > other.target_distance

    class Layer:
        def __init__(self, max_size : int, max_neighbors : int = 12) -> None:
            self.points = {}
            self.neighbors = {}
            self.max_size = max_size
            self.max_neighbors = max_neighbors
        
        def empty(self):
            return not len(self.points)

        def full(self):
            return len(self.points) >= self.max_size

        def add_point(self, point_id, neighbors):
            self.points[point_id] = True
            self.neighbors[point_id] = []
            heapify(self.neighbors[point_id])
            for pt in neighbors:
                heappush(self.neighbors[point_id], pt)
        
        def contest_point(self, new, target_id):
            if len(self.neighbors[target_id]) < self.max_neighbors:
                heappush(self.neighbors[target_id], new)
                return
            heappushpop(self.neighbors[target_id], new)


    def __init__(self,
                 distance : Callable[..., float],
                 max_neighbors : int = 12,
                 base : int = 64,
                 factor : float = 32.0) -> None:

        self.distance = distance
        self.max_neighbors = max_neighbors
        self.base = base
        self.factor = factor
        self.points = {}
        self.layers = []
    
    def fit(self, X):
        for pt in X:
            self.add(pt)

    def search_layer(self, point, layer_id: int, k : int = 5,
                     init: Optional[List] = None,
                     used: Optional[Dict[int, bool]] = None) -> List[int]:
        if self.layers[layer_id].empty():
            return []
        if init is None or not len(init):
            init = [choice(list(self.layers[layer_id].points.keys()))]
        if used is None:
            used = {}
        queue = []
        heapify(queue)
        for pt_idx in init:
            heappush(queue, HeuristicHNSW.ClosestOut(pt_idx, self.distance(point, self.points[pt_idx])))
            used[pt_idx] = True
        prev = -1
        res = []
        while len(queue):
            closest = heappop(queue)
            pt_idx = closest.point_id
            if prev == pt_idx:
                res.append(pt_idx)
                if len(res) >= k:
                    break
                continue
            prev = pt_idx
            if self.layers[layer_id].neighbors.get(pt_idx) is None:
                continue
            for neighbor_pt in self.layers[layer_id].neighbors[pt_idx]:
                neighbor_idx = neighbor_pt.point_id
                if used.get(neighbor_idx) is None:
                    used[neighbor_idx] = True
                    heappush(queue, HeuristicHNSW.ClosestOut(neighbor_idx, self.distance(point, self.points[neighbor_idx])))
            heappush(queue, closest)
        return res
    
    def descend_search(self, point, k : int = 5):
        init = []
        used = {}
        for idx, layer in enumerate(self.layers):
            init = self.search_layer(point, idx, k, init, used)
            used = deepcopy(layer.points)
        return init

    def add(self, point):
        point_id = len(self.points.keys())
        self.points[point_id] = point
        if not len(self.layers):
            self.layers.append(HeuristicHNSW.Layer(self.base, self.max_neighbors))
        if self.layers[-1].full():
            self.layers.append(deepcopy(self.layers[-1]))
            self.layers[-1].max_size = round(self.layers[-1].max_size * self.factor)
        neighbors = self.descend_search(point, self.max_neighbors)
        self.layers[-1].points[point_id] = True
        self.layers[-1].neighbors[point_id] = []
        heapify(self.layers[-1].neighbors[point_id])
        for pt_idx in neighbors:
            heappush(self.layers[-1].neighbors[point_id], HeuristicHNSW.FurthestOut(pt_idx, self.distance(point, self.points[pt_idx])))
            self.layers[-1].contest_point(HeuristicHNSW.FurthestOut(point_id, self.distance(point, self.points[pt_idx])), pt_idx)
        
    def stat(self):
        if not len(self.layers):
            return
        for pt_idx in self.layers[-1].neighbors.keys():
            print(pt_idx, end=" -- ")
            for pt in self.layers[-1].neighbors[pt_idx]:
                print(pt.point_id, end=" ")
            print()
        
