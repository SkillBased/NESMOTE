from typing import Callable, List, Dict, Tuple, Optional
from numpy.random import choice, randint
from heapq import heapify, heappush, heappop, heappushpop
from collections import deque
from math import floor, log, e


class PyLogger:
    def __init__(self, filename : Optional[str] = None) -> None:
        self.outfile = open(filename if filename is not None else "logger.out", "w")
        self.closed = False

    def log(self, string : str) -> None:
        if not self.closed:
            self.outfile.write(string + "\n")
    
    def close(self) -> None:
        self.closed = True
        self.outfile.close()

global_logger = PyLogger()
global_logger.close()

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

        global_logger.log(f"SWL.get_closest() clalled")

        current = origin_id if origin_id is not None else choice(list(self.points.keys()))
        improving = True
        while improving:

            global_logger.log(f"scanning {len(self.edges[current])} nodes")

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

        global_logger.log(f"SWL.get_k_closest(..., {k}) is returning, {len(used.keys())} nodes are marked used")

        return [targeted.point_id for targeted in k_closest]


class SmallWorld(SmallWorldLayer):
    def __init__(self, distance: Callable[..., float], min_neighbors: int = 8, max_neighbors: int = 16) -> None:
        super().__init__(distance, min_neighbors, max_neighbors)

    def fit(self, X):
        for idx, point in enumerate(X):
            self.add(idx, point)
       
    def add(self, point_id : int, point, start_from : Optional[List[int]] = None) -> List[int]:
        if len(self.points) == 0:
            self.insert(point_id, point, [])
            return []
        neighbors = self.get_k_closest(point, start_from, k=self.min_neighbors)[0]
        self.insert(point_id, point, neighbors)
        return neighbors
    
    def query(self, target, k : int = 5) -> List[int]:
        return self.get_k_closest(target, k=k)[0]


# Hierarchical Navigable Small World
class HNSW:
    def __init__(self, distance : Callable[..., float], height : int = 10, min_neighbors : int = 8, max_neighbors : int = 12, factor : float = e) -> None:
        self.distance = distance

        self.points = []
        self.layers = []
        self.height = height

        self.min_neighbors = min_neighbors
        self.max_neighbors = max_neighbors

        self.base = factor
    
    def fit(self, X):
        self.height = floor(log(X.shape[0], self.base)) - 3
        self.layers = [SmallWorldLayer(self.distance, self.min_neighbors, self.max_neighbors) for _ in range(self.height)]
        for point in X:
            self.add(point)

    # randomise top layer, get k closest 
    # NOTE: maybe instead set top layer as log(point_id) allowing for more balanced structure
    # next layer (n-1) get closest from the closest of previous layer
    # should work OK
    def add(self, point) -> None:

        global_logger.log(f"HNSW.add() called")

        new_point_id = len(self.points)
        point_height = max(min(self.height - 1 - floor(log(new_point_id + 1, self.base)) + 3, self.height - 1), 0)
        '''
        start_from = None
        for level in range(self.height - 1, 0, -1):
            if self.layers[level].empty():
                continue
            start_from = self.layers[level].get_closest(point, start_from)
        if start_from is not None:
            neighbors = [start_from]
        '''
        neighbors = None
        for level in range(point_height, -1, -1):
            if self.layers[level].empty():
                self.layers[level].insert(new_point_id, point, [])
            else:
                neighbors = self.layers[level].get_k_closest(point, neighbors, self.min_neighbors)
                self.layers[level].insert(new_point_id, point, neighbors)
        self.points.append(point)

        global_logger.log(f"HNSW.add() call ended")
        global_logger.log("")

    
    # get k closest on level 0
    # IDK on speed - O(mu * log(n)) propably as we descend though log(n) levels and single level should be linear-ish
    # NOTE: in case of NNG construction this is actually almost redundant
    def query(self, target, k : int = 5) -> List[int]:

        global_logger.log(f"HNSW.query(..., {k}) called")

        start_from = None
        for level in range(self.height - 1, 0, -1):
            if self.layers[level].empty():
                continue
            start_from = self.layers[level].get_k_closest(target, start_from, 1)
        neighbors = self.layers[0].get_k_closest(target, start_from, k)

        global_logger.log(f"HNSW.query() call ended")
        global_logger.log("")

        return neighbors

    def stat(self):
        print(f"HNSW contsains {len(self.points)} points")
        for layer in range(self.height):
            print(f"layer {layer} has {len(self.layers[layer].points)} points")
