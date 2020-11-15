import copy
from enum import Enum

import numpy as np


class Type(Enum):
    UNCHANGED = 0
    MERGE = 1
    SPLIT = 2


class TreeStructureEvent(object):

    def __init__(self, type: Type, node=None):
        self.type = type
        self.node = node

    def __str__(self):
        return self.type


class HyperRect(object):

    def __init__(self, maxes, mins):
        self.maxes = np.maximum(maxes, mins).astype(float)
        self.mins = np.minimum(maxes, mins).astype(float)
        self.m, = self.maxes.shape

    def area(self):
        return np.prod(self.maxes - self.mins)

    def contains(self, point):

        for dim in range(0, self.m):
            if not self.mins[dim] <= point[dim] <= self.maxes[dim]:
                return False

        return True

    def split(self, d, split):
        mid = np.copy(self.maxes)
        mid[d] = split
        less = HyperRect(self.mins, mid)
        mid = np.copy(self.mins)
        mid[d] = split
        greater = HyperRect(mid, self.maxes)
        return less, greater


class TreeContext:

    def __init__(self):
        self._log = []

    def log(self, event: TreeStructureEvent):
        self._log.append(event)

    def state(self):
        if not self._log:
            return [TreeStructureEvent(Type.UNCHANGED)]
        else:
            return self._log


class KDQ:

    def __init__(self, k, split_size=5, depth=0, axis=0, hyperrect: HyperRect = None, points=None):
        if hyperrect is None:
            hyperrect = HyperRect([1.0] * k, [0.0] * k)
        if points is None:
            points = np.empty((0, k), dtype=float)

        assert len(hyperrect.maxes) == k
        assert len(hyperrect.mins) == k

        self.k = k
        self.split_size = split_size
        self.depth = depth
        self.axis = axis
        self.hyperrect = hyperrect
        self.points = points
        self.nodes = []
        self.weight = 0

        self.left: KDQ or None = None
        self.right: KDQ or None = None

        self.divided = False

    def insert(self, point, context: TreeContext = None):

        if not self.hyperrect.contains(point):
            return False

        if self.weight < self.split_size:
            self.points = np.vstack((self.points, point))
            self.weight = self.points.shape[0]
            return True

        if not self.divided:
            self.split()
            if context:
                context.log(TreeStructureEvent(Type.SPLIT, self))

        self.weight = self.weight + 1

        return self.left.insert(point, context) or self.right.insert(point, context)

    def delete(self, point, context: TreeContext = None):

        if not self.hyperrect.contains(point):
            return False

        # If I'm a leaf
        if not self.divided:
            # Then try and delete the point from my list
            mask = np.equal(point, self.points).all(axis=1)
            self.points = np.delete(np.array(self.points), mask, axis=0)

            deleted = mask.any()
        else:
            deleted = self.left.delete(point, context) or self.right.delete(point, context)

        if deleted:
            self.weight = self.weight - 1

        # If the deletion means we no longer break the split threshold, merge the child nodes
        if self.divided and self.weight <= self.split_size:
            self.merge()
            if context:
                context.log(TreeStructureEvent(Type.MERGE, self))

        return deleted

    def _insert_in_structure(self, point):

        if not self.hyperrect.contains(point):
            return False

        if not self.divided:
            np.vstack((self.points, point))
            return True

        return self.left._insert_in_structure(point) or self.right._insert_in_structure(point)

    def split(self):
        axis = self.depth % self.k

        midpoint = (self.hyperrect.mins[axis] + self.hyperrect.maxes[axis]) / 2
        l_hr, r_hr = self.hyperrect.split(axis, midpoint)

        self.left = KDQ(self.k, self.split_size, self.depth + 1, axis, l_hr)
        self.right = KDQ(self.k, self.split_size, self.depth + 1, axis, r_hr)

        points = self.points.copy()
        self.points = np.empty((0, self.k), dtype=float)
        for point in points:
            if point[axis] < midpoint:
                self.left.insert(point)
            else:
                self.right.insert(point)

        self.divided = True

        return self.left, self.right

    def merge(self):
        """
        Merges two child nodes into this one.
        """
        self.points = np.vstack((self.left.points, self.right.points))
        self.weight = self.points.shape[0]
        self.left = None
        self.right = None
        self.divided = False

    @staticmethod
    def dft(node, action):
        if node:
            action(node)
            KDQ.dft(node.left, action)
            KDQ.dft(node.right, action)

    def clear(self):
        def clear_points(node):
            node.points = np.empty((0, self.k), dtype=float)
            node.weight = 0

        KDQ.dft(self, clear_points)

    def clear_copy(self):
        copy_tree = copy.deepcopy(self)
        copy_tree.clear()
        return copy_tree

    def in_structure_hist(self, points):
        copy_tree = self.clear_copy()
        for point in points:
            copy_tree._insert_in_structure(point)

        return copy_tree.to_hist()

    def to_hist(self):
        nodes = []
        KDQ.dft(self, lambda node: nodes.append(node))
        hist = [len(node.points) for node in nodes]
        return np.array(hist)
