import numpy as np
import concurrent.futures
from typing import Callable, List, Tuple, Optional, Any
from math import floor, ceil, log, e
from heapq import heapify, heappush, heappop, heappushpop, nsmallest
from copy import deepcopy
from random import gammavariate, random, choice
from sortedcontainers.sortedset import SortedSet

from NESMOTE.neighbors import NENN


class BaseProcessor:
    def __init__(self, k) -> None:
        self.augmentation_groups = []
        self.neighbor_limit = k
    
    def split(self, neighbors : NENN) -> None:
        return

    def get(self, n : int) -> List[Any]:
        return [choice(self.augmentation_groups) for _ in range(n)]

class SMOTEProcessor(BaseProcessor):
    def __init__(self, k : int = 5) -> None:
        super().__init__(k)

    def split(self, neighbors : NENN, n_jobs : Optional[int] = None) -> None:
        self.augmentation_groups = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=n_jobs) as exectutor:
            promised = {exectutor.submit(neighbors.query, neighbors.data[point_id], self.neighbor_limit + 1): point_id for point_id in np.arange(neighbors.size())}
            for future in concurrent.futures.as_completed(promised):
                point_id = promised[future]
                ann = future.result()
                if point_id in ann:
                    ann.pop(ann.index(point_id))
                for neighbor_id in ann:
                    self.augmentation_groups.append([point_id, neighbor_id])


class ANNProcessor(BaseProcessor):
    def __init__(self, k : int = 5) -> None: 
        super().__init__(k)
    
    def split(self, neighbors : NENN, n_jobs : Optional[int] = None) -> None:
        self.augmentation_groups = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=n_jobs) as exectutor:
            promised = {exectutor.submit(neighbors.query, neighbors.data[point_id], self.neighbor_limit + 1): point_id for point_id in np.arange(neighbors.size())}
            for future in concurrent.futures.as_completed(promised):
                point_id = promised[future]
                ann = future.result()
                ann.pop(ann.index(point_id))
                self.augmentation_groups.append(ann)

class MCProcessor(BaseProcessor):

    class SplitPoint:
        def __init__(self, point_id : int, knn : List[int]) -> None:
            self.id = point_id
            self.left = SortedSet()
            self.right = SortedSet()
            for neighbor in knn:
                if neighbor < point_id:
                    self.left.add(neighbor)
                if neighbor > point_id:
                    self.right.add(neighbor)
        
        def get(self, side : str):
            if side == "left":
                return deepcopy(self.left)
            if side == "right":
                return deepcopy(self.right)

    def __init__(self, k : int = 5) -> None:
        super().__init__(k)
        self.processed = {}
        self.cliques = {}
    
    def _process(self, neighbors : NENN, n_jobs : Optional[int] = None) -> None:
        self.processed = {}
        with concurrent.futures.ThreadPoolExecutor(max_workers=n_jobs) as exectutor:
            promised = {exectutor.submit(neighbors.query, neighbors.data[point_id], self.neighbor_limit + 1): point_id for point_id in np.arange(neighbors.size())}
            for future in concurrent.futures.as_completed(promised):
                point_id = promised[future]
                ann = future.result()
                ann.pop(ann.index(point_id))
                self.processed[point_id] = MCProcessor.SplitPoint(point_id, ann)

    def split(self, neighbors: NENN, n_jobs : Optional[int] = None) -> None:
        self._process(neighbors, n_jobs)
        self.cliques = {}
        for point_id in np.arange(neighbors.size()):
            self.cliques[point_id] = []
            left_list = self.processed[point_id].get("left")
            while len(left_list):
                header = left_list.pop(0)
                candidates = self.processed[header].get("right")
                self._update(header, point_id, left_list.intersection(candidates))
        self.augmentation_groups = []
        for point_id in np.arange(neighbors.size()):
            max_size, res = 0, SortedSet()
            for clique in self.cliques[point_id]:
                if len(clique) > max_size:
                    max_size, res = len(clique), clique
            self.augmentation_groups.append([idx for idx in iter(res)])
        
    def _update(self, header, vertice, renovation) -> None:
        if (len(self.cliques[header]) == 0 or not len(renovation)):
            self.cliques[header] += [SortedSet([vertice])]
            return
        perfect_match = False
        partial_set = []
        full_set = []
        for i in range(len(self.cliques[header])):
            clique = self.cliques[header][i]
            if not len(renovation.symmetric_difference(clique)):  # aka sets are equal
                self.cliques[header][i].add(vertice)
                perfect_match = True
                break
            elif not len(renovation.symmetric_difference(clique)):  # aka renovation is subset for clique
                renovation.add(vertice)
                self.cliques[header] += renovation
                perfect_match = True
                break
            elif not len(clique.symmetric_difference(renovation)):  # aka clique is subset for renovation
                self.cliques[header][i].add(vertice)
                full_set += [clique.intersection(renovation)]
            elif len(clique.intersection(renovation)):
                partial_set += [clique.intersection(renovation)]
        if not perfect_match:
            for candidate in partial_set:
                not_inferior = True
                for pretendent in partial_set:
                    if ((len(candidate.symmetric_difference(pretendent))) and len(pretendent.difference(candidate))):
                    # aka candidate is a strict subset for pretendent
                        not_inferior = False
                        break
                for pretendent in full_set:
                    if len(pretendent.difference(candidate)):
                        not_inferior = False
                        break
                if not_inferior:
                    candidate.add(vertice)
                    self.cliques[header] += [candidate]


class BaseSampler:
    def __init__(self, sampler_func : Callable[[List[float], List[Any]], Any]) -> None:
        self.sampler = sampler_func
    
    def make_samples(self, groups : List[Any]) -> List[Any]:
        return [], []
    
class StandardSampler(BaseSampler):
    def __init__(self, sampler_func: Callable[[List[float], List[Any]], Any]) -> None:
        super().__init__(sampler_func)
    
    def make_samples(self, groups: List[Any]) -> List[Any]:
        synthetic = []
        with concurrent.futures.ThreadPoolExecutor() as exectutor:
            promised = [exectutor.submit(self.sampler, standard_picker(len(group)), group) for group in groups]
            for future in concurrent.futures.as_completed(promised):
                x = future.result()
                synthetic.append(x)
        return synthetic

class UniformSampler(BaseSampler):
    def __init__(self, sampler_func: Callable[[List[float], List[Any]], Any]) -> None:
        super().__init__(sampler_func)
    
    def make_samples(self, groups: List[Any]) -> List[Any]:
        synthetic = []
        with concurrent.futures.ThreadPoolExecutor() as exectutor:
            promised = [exectutor.submit(self.sampler, adaptive_gamma_picker(len(group)), group) for group in groups]
            for future in concurrent.futures.as_completed(promised):
                x = future.result()
                synthetic.append(x)
        return synthetic


class Pipeline:
    def __init__(self, neighbors : Optional[NENN] = None, processor : Optional[BaseProcessor] = None, sampler : Optional[Callable[[List[float], List[Any]], Any]] = None) -> None:
        self.neighbors = None
        self.processor = None
        self.sampler = None

        if neighbors is not None and processor is not None and sampler is not None:
            self.setup(neighbors, processor, sampler)
    
    def reset(self) -> None:
        self.neighbors = None
        self.processor = None
        self.sampler = None
    
    def setup(self, neighbors : NENN, processor : BaseProcessor, sampler : BaseSampler):
        self.neighbors = neighbors
        self.processor = processor
        self.sampler = sampler
    
    def fit_resample(self, X, y):
        nX, ny = [], []
        if self.neighbors is None or self.processor is None:
            raise ZeroDivisionError  # placeholder
        classes = {}
        majority = 0
        for value, count in np.transpose(np.unique(y, return_counts=True)):
            class_pts = X[(y == value)]
            majority = max(majority, count)
            classes[value] = class_pts
        for class_value in classes.keys():
            # select all data from a single class
            class_pts = classes[class_value]
            # determine the oversampling volume
            targets = majority - len(class_pts)
            if targets < 1:
                continue
            # fit nearest neighbor search
            self.neighbors.fit(class_pts)
            # construct groups based on KNN graph
            self.processor.split(self.neighbors)
            # extract groups
            groups = self.processor.get(targets)
            # preprocessing before generation
            augment = []
            for group in groups:
                raw = [class_pts[idx] for idx in group]
                if len(raw):
                    augment.append(raw)
            # synthetic data generation
            nX = self.sampler.make_samples(augment)
            ny = [class_value] * len(nX)
        resX, resy = X, y
        if len(nX):
            resX = np.vstack([X, np.array(nX)])
            resy = np.hstack([y, np.array(ny)])
        return resX, resy


def adaptive_gamma_picker(n):
    weights = []
    weight_sum = 0
    for i in range(n):
        new_weight = abs(gammavariate(2 / n, 2 ** n))
        weights.append(new_weight)
        weight_sum += new_weight
    for i in range(n):
        weights[i] = weights[i] / weight_sum
    return weights

def standard_picker(n):
    weights = []
    weight_sum = 0
    for i in range(n):
        new_weight = random()
        weights.append(new_weight)
        weight_sum += new_weight
    for i in range(n):
        weights[i] = weights[i] / weight_sum
    return weights 
