import numpy as np
from typing import Callable, Optional
from heapq import nsmallest
from math import floor, ceil, log2

from RQ.base import TargertedPoint, Origin

class SortedSetQuery:
    def __init__(self, distance : Callable[..., float]) -> None:
        self.distance = distance
        self.origins = []
        self.data = None
    
    def fit(self, data) -> None:
        self.data = data
        origin_count = self.data.shape[1] + 1
        origin_ids = np.random.choice(np.arange(self.data.shape[0]), origin_count)
        for origin_id in origin_ids:
            self.origins.append(Origin(self.data[origin_id]))
        point_id = 0
        for point in data:
            self.add(point_id, point)
            point_id += 1

    def add(self, point_id : int, point) -> None:
        for i in range(len(self.origins)):
            self.origins[i].add(TargertedPoint(point_id, self.distance(point, self.origins[i].point)))

    def query(self, point, k=5):
        spread = round((log2(self.data.shape[0])) ** 2)
        candidates = None
        for origin in self.origins:
            target = TargertedPoint(-1, self.distance(point, origin.point))
            origin_cands = set(origin.query(target, extend=spread))
            if candidates is None:
                candidates = origin_cands
            else:
                candidates.intersection_update(origin_cands)
        processed = []
        for point_id in candidates:
            processed.append(TargertedPoint(point_id, self.distance(point, self.data[point_id])))
        processed.sort()
        if k != 0 and k + 1 < len(processed):
            processed = processed[:k + 1]
        result = [neighbor.pid for neighbor in processed]
        return self.data[result]


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

    def stat(self, file="stat.out"):
        outfile = open(file, "w")
        outfile.write(f"RQ has {len(self.origin_ids)} origins\n")
        for origin_id in self.origin_ids:
            outfile.write(f"{origin_id} -- ")
            for ring in self.rings[origin_id].keys():
                outfile.write(f"{len(self.rings[origin_id][ring])} ")
            outfile.write("\n")
        outfile.close()