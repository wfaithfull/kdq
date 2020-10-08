from enum import Enum

from kdq.tree import KDQ, HyperRect, TreeContext


class KDQTree:

    def __init__(self, k, split_size=5, bounds=None, max_size=None):
        if bounds is not None:
            assert bounds.shape == (2, k), f"Bounds should be a numpy array of [maxes; mins], shape [2, {k}]"
            self.root = KDQ(k=k, split_size=split_size, hyperrect=HyperRect(bounds[0, :], bounds[1, :]))
        else:
            self.root = KDQ(k=k, split_size=split_size)

        self.k = k
        self.max_size = max_size
        self.points = []

    def insert(self, point):

        context = TreeContext()

        if self.max_size:
            if self.root.weight >= self.max_size:
                oldest_point = self.points.pop(0)
                self.root.delete(oldest_point, context)

        inserted = self.root.insert(point, context)

        if inserted:
            self.points.append(point)

        return inserted, context.state()

    def insert_batch(self, points):
        results = []
        i = 0
        for point in points:
            result = self.insert(point)
            results.append(result)

        return results

    def to_hist(self):
        return self.root.to_hist()

    def in_structure_hist(self, points):
        return self.root.in_structure_hist(points)

    def clear(self):
        self.root.clear()
