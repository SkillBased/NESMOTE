import numpy as np
from typing import Callable, Optional, Tuple
from heapq import nsmallest
from math import floor, ceil, sin, pi
from sortedcontainers.sortedset import SortedList

class TargertedPoint:
    def __init__(self, point_id : int, distance : float):
        self.pid = point_id
        self.dist = distance
    
    def __lt__(self, other) -> bool:
        if type(other) == TargertedPoint:
            if self.dist == other.dist:
                return self.pid < other.pid
            return self.dist < other.dist
        raise(TypeError)
    
    def __eq__(self, other) -> bool:
        if type(other) == TargertedPoint:
            return self.pid == other.pid
        raise(TypeError)
    
    def __hash__(self) -> int:
        return hash(self.dist)


class Origin:
    def __init__(self, point) -> None:
        self.point = point
        self.data = SortedList()
    
    def add(self, targeted_point : TargertedPoint) -> None:
        self.data.add(targeted_point)
    
    def query(self, targeted_point : TargertedPoint, extend : int = 64) -> list[TargertedPoint]:
        low = max(self.data.bisect_left(targeted_point) - extend, 0)
        high = min(self.data.bisect_right(targeted_point) + 1 + extend, len(self.data) - 1)
        res = [point.pid for point in self.data[low: high]]
        return res


class QueryTreeNode:
    def __init__(self, point : Optional[TargertedPoint] = None) -> None:
        self.point = point
        self.height = 0
        self.width = 0 if self.point is None else 1
        self.left = None
        self.right = None
    
    def get_heights(self) -> Tuple[int, int]:
        left_height, right_height = 0, 0
        if self.left is not None:
            left_height = self.left.height
        if self.right is not None:
            right_height = self.right.height
        return left_height, right_height

    def get_widths(self) -> Tuple[int, int]:
        left_width, right_width = 0, 0
        if self.left is not None:
            left_width = self.left.width
        if self.right is not None:
            right_width = self.right.width
        return left_width, right_width
    
    def recount(self) -> None:
        self.height = max(self.get_heights()) + 1
        self.width = sum(self.get_widths()) + 1
        
    def add(self, new_point : TargertedPoint) -> None:
        if self.point is None:
            self.point = new_point
            return 0
        if new_point < self.point:
            if self.left is None:
                self.left = QueryTreeNode(new_point)
            else:
                self.left.add(new_point)
                self.left = rebalance_node(self.left)
        else:
            if self.right is None:
                self.right = QueryTreeNode(new_point)
            else:
                self.right.add(new_point)
                self.right = rebalance_node(self.right)
        self.recount()
    
    def traverse(self) -> list[int]:
        res = []
        if self.left is not None:
            res.extend(self.left.traverse())
        res.append(self.point.pid)
        if self.right is not None:
            res.extend(self.right.traverse())
        return res

def rebalance_node(node : QueryTreeNode) -> QueryTreeNode:
    left_height, right_height = node.get_heights()
    if left_height - right_height > 1:
        root = node.left
        node.left = root.right
        root.right = node
        return root
    if right_height - left_height > 1:
        root = node.right
        node.right = root.left
        root.left = node
        return root
    return node
