import numpy as np
import concurrent.futures
from typing import Callable, List, Optional
from math import floor, ceil, log, e
from heapq import heapify, heappush, heappop, heappushpop, nsmallest
from numpy.random import choice
from copy import deepcopy

class NENN:
    '''
    A template class for nearest neighbor search within NESMOTE module.
    Builtin components rely on avaliability of `fit`, `query` and `size` functions with exact arguments.
    Also they expect the ability to address points of the dataset as `NENN.data[idx]`.
    '''
    def __init__(self, distance_metric : Callable[..., float]) -> None:
        self.distance = distance_metric
    
    def fit(self, X) -> None:
        return
    
    def query(self, point, k : int = 5, raw : bool = False) -> List:
        return []

    def size(self) -> int:
        return 0

class RingQuery(NENN):
    '''
    RingQuery is a nearest neighbor search class that supports non-Euclidean spaces.
    The processing algorithm minimizes the amount of distance computations using a linear precomputation.
    In total O(n) distances are computed while alternatives make at least O(n log n) computations.
    With higher complexity of spaces and thus distance computations this optimization proves highly beneficial.
    RingQuery is not expandable naturally and has to be refit, but it supports querying for non-dataset points.
    RingQuery is technically a probabilistic algorithm but for reasonable applications it can be considered precise.
    '''
    def __init__(self, distance_metric: Callable[..., float]) -> None:
        super().__init__(distance_metric)
        self.data = None
        self.ring_width = None
        self.origin_ids = set()
        self.rings = {}
        
    def _adapt_width(self, nsamples : int = 8, cut : float = .01) -> float:
        '''
        The magic of assesing the dataset parameters and finding optimal ring width.
        Default parameters are a safe bet, but for extra large sets it is recommended to lower the `cut` value.
        NOTE: this may or may not be fixed in the new approach.
        '''
        if self.data is None:
            return
        edge = min([self.data.shape[0], max(16, int(self.data.shape[0] * cut)), 64])
        origins = np.random.choice(np.arange(self.data.shape[0]), nsamples)
        edges = [self._adapt_single(origin_id, edge) for origin_id in origins]
        return np.mean(edges)
    
    def _adapt_single(self, origin_id : int, edge : int) -> Optional[float]:
        '''
        Basically a greedy knn that allows to estimate the average largest neighbor distance with a just-right overshot:
        Not too little to miss any neighbors, but not too large to slow down the algorithm
        '''
        origin_distance = lambda point: self.distance(self.data[origin_id], point)
        distances = np.apply_along_axis(origin_distance, 1, self.data)
        smallest = nsmallest(edge, distances)
        return smallest[-1]

    def fit(self, X, n_origins : Optional[int] = None, width : Optional[float] = None, n_jobs : Optional[int] = None) -> None:
        '''
        TBA: summary
        '''
        self.data = X
        origin_count = n_origins if n_origins is not None else self.data.shape[1] + 1
        self.origin_ids = np.random.choice(np.arange(self.data.shape[0]), origin_count)
        self.ring_width = width if width is not None else self._adapt_width()
        self.rings = {}
        with concurrent.futures.ThreadPoolExecutor(max_workers=n_jobs) as exectutor:
            promised = {exectutor.submit(self._process_single, origin_id): origin_id for origin_id in self.origin_ids}
            for future in concurrent.futures.as_completed(promised):
                origin_id = promised[future]
                self.rings[origin_id] = future.result()

    def _process_single(self, origin_id : int) -> dict[int, set[int]]:
        '''
        TBA: summary
        '''
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

    def query(self, point, k : int = 5, raw : bool = False):
        '''
        TBA: summary
        '''
        point_distance = lambda point_id: (self.distance(self.data[point_id], point), point_id)
        candidates = None
        for origin_id in self.origin_ids:
            d = round(self.distance(point, self.data[origin_id]) / self.ring_width)
            if candidates is None:
                candidates = deepcopy(self.rings[origin_id].get(d))
            else:
                candidates.intersection_update(self.rings[origin_id].get(d))
        processed = []
        for point_id in candidates:
            processed.append(point_distance(point_id))
        processed.sort()
        if k != 0 and k < len(processed):
            processed = processed[:k]
        result = [neighbor[1] for neighbor in processed]
        if raw:
            return self.data[result]
        return result

    def size(self) -> int:
        return self.data.shape[0]
    
class SmallWorldLayer:

    # Point with distance to target. In sorted first point is furthest
    class FurthestOut:
        def __init__(self, point_id : int, target_distance : float) -> None:
            self.point_id = point_id
            self.target_distance = target_distance

        # operator is reversed to work with heappushpop
        def __lt__(self, other) -> bool:
            if self.target_distance == other.target_distance:
                return self.point_id > other.point_id
            return self.target_distance > other.target_distance

    # Point with distance to target. In sorted first point is closest
    class ClosestOut:
        def __init__(self, point_id : int, target_distance : float) -> None:
            self.point_id = point_id
            self.target_distance = target_distance

        def __lt__(self, other) -> bool:
            if self.target_distance == other.target_distance:
                return self.point_id < other.point_id
            return self.target_distance < other.target_distance

    def __init__(self, distance : Callable[..., float], min_neighbors : int = 8, max_neighbors : int = 16) -> None:
        self.distance = distance

        self.points = {}
        self.edges = {}

        self.min_neighbors = min_neighbors
        self.max_neighbors = max_neighbors
    
    def empty(self) -> bool:
        return len(self.points) == 0
    
    def get_FurthestOut(self, point_id : int, target):
        return SmallWorldLayer.FurthestOut(point_id, self.distance(self.points[point_id], target))

    def get_ClosestOut(self, point_id : int, target):
        return SmallWorldLayer.ClosestOut(point_id, self.distance(self.points[point_id], target))

    def insert(self, point_id : int, point, neighbors : List[int]) -> None:
        self.points[point_id] = point
        self.edges[point_id] = []
        for neighbor_id in neighbors:
            self.prep_point(neighbor_id)
            self.edges[neighbor_id].append(point_id)
            self.edges[point_id].append(neighbor_id)
    
    def prep_point(self, point_id : int) -> None:
        if len(self.edges[point_id]) >= self.max_neighbors:
            furthest_id = self.edges[point_id][0]
            for neighbor_id in self.edges[point_id]:
                if self.distance(self.points[point_id], self.points[neighbor_id]) > self.distance(self.points[point_id], self.points[furthest_id]):
                    furthest_id = neighbor_id
            self.edges[point_id].pop(self.edges[point_id].index(furthest_id))

    # scan all neighbors and advance to closest until noone is closer
    def get_closest(self, target, origin_id : Optional[int] = None) -> int:
        current = origin_id if origin_id is not None else choice(list(self.points.keys()))
        improving = True
        while improving:
            improving = False
            for neighbor_id in self.edges[current]:
                if self.distance(self.points[neighbor_id], target) < self.distance(self.points[current], target):
                    improving = True
                    current = neighbor_id
        return current

    # note that this still works even if theres less than k points in graph
    def get_k_closest(self, target, start_from : Optional[List[int]] = None, k : int = 5) -> List[int]:
        start = start_from if start_from is not None else [choice(list(self.points.keys()))]
#        start = start_from if start_from is not None else [self.get_closest(target)]
        query_queue = []
        heapify(query_queue)
        k_closest = []
        heapify(k_closest)
        used = {}
        for point_id in start:
            heappush(query_queue, self.get_ClosestOut(point_id, target))
            heappush(k_closest, self.get_FurthestOut(point_id, target))
            used[point_id] = True
        while len(query_queue):
            next_point = heappop(query_queue)
            current, distance = next_point.point_id, next_point.target_distance
            if distance > k_closest[0].target_distance:
                break
            for neighbor_id in self.edges[current]:
                if used.get(neighbor_id) is None:
                    used[neighbor_id] = True
                    if len(k_closest) < k:
                        heappush(k_closest, self.get_FurthestOut(neighbor_id, target))
                        heappush(query_queue, self.get_ClosestOut(neighbor_id, target))
                    else:
                        furthest = heappushpop(k_closest, self.get_FurthestOut(neighbor_id, target))
                        if furthest != neighbor_id:
                            heappush(query_queue, self.get_ClosestOut(neighbor_id, target))
        return [targeted.point_id for targeted in k_closest]

# Hierarchical Navigable Small World
class HNSW(NENN):
    def __init__(self, distance_metric : Callable[..., float], height : int = 10, min_neighbors : int = 8, max_neighbors : int = 12, factor : float = e) -> None:
        super().__init__(distance_metric)

        self.data = []
        self.layers = []
        self.height = height

        self.min_neighbors = min_neighbors
        self.max_neighbors = max_neighbors

        self.base = factor
    
    def fit(self, X):
        self.data = []
        self.height = floor(log(X.shape[0], self.base)) - 3
        self.layers = [SmallWorldLayer(self.distance, self.min_neighbors, self.max_neighbors) for _ in range(self.height)]
        for point in X:
            self.add(point)

    # randomise top layer, get k closest 
    # NOTE: maybe instead set top layer as log(point_id) allowing for more balanced structure
    # next layer (n-1) get closest from the closest of previous layer
    # should work OK
    def add(self, point) -> None:
        new_point_id = len(self.data)
        point_height = max(min(self.height - 1 - floor(log(new_point_id + 1, self.base)) + 3, self.height - 1), 0)
        neighbors = None
        for level in range(point_height, -1, -1):
            if self.layers[level].empty():
                self.layers[level].insert(new_point_id, point, [])
            else:
                neighbors = self.layers[level].get_k_closest(point, neighbors, self.min_neighbors)
                self.layers[level].insert(new_point_id, point, neighbors)
        self.data.append(point)

    def query(self, target, k : int = 5) -> List[int]:
        start_from = None
        for level in range(self.height - 1, 0, -1):
            if self.layers[level].empty():
                continue
            start_from = self.layers[level].get_k_closest(target, start_from, 1)
        neighbors = self.layers[0].get_k_closest(target, start_from, k)
        return neighbors

    def stat(self):
        print(f"HNSW contsains {len(self.data)} points")
        for layer in range(self.height):
            print(f"layer {layer} has {len(self.layers[layer].data)} points")
    
    def size(self) -> int:
        return len(self.data)