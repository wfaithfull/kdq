import unittest

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import patches

from kdq import KDQTree
from kdq.ekdq import EKDQDetector
from kdq.kl import KLDetector


class KDQTreeTests(unittest.TestCase):

    @unittest.skip(reason="Intended for manual inspection only")
    def test_plot_2d(self):
        tree = KDQTree(2, split_size=5, bounds=np.vstack(([5.0] * 2, [-5.0] * 2)))
        coords = np.random.randn(1000, 2)
        tree.insert_batch(coords)

        fig = plt.figure()
        ax = fig.add_subplot()

        def walk_and_plot_2d(node):

            def to_coords(mins, maxes):
                anchor = mins
                dimensions = [maxes[i] - mins[i] for i in range(0, node.k)]

                return anchor, dimensions

            anchor, dimensions = to_coords(node.hyperrect.mins, node.hyperrect.maxes)

            if node.divided:
                walk_and_plot_2d(node.left)
                walk_and_plot_2d(node.right)
            else:
                rect = patches.Rectangle(anchor, dimensions[0], dimensions[1], edgecolor='b', facecolor="none")
                ax.add_patch(rect)
                if node.points.size > 0:
                    points = np.array(node.points)
                    ax.scatter(points[:, 0], points[:, 1])

        walk_and_plot_2d(tree.root)
        plt.show()

    def test_2d(self):
        k = 2
        n = 1000

        self._test_nd(k, n)

    def test_3d(self):
        k = 3
        n = 1000

        self._test_nd(k, n)

    def test_4d(self):
        k = 4
        n = 1000

        self._test_nd(k, n)

    def test_5d(self):
        k = 5
        n = 1000

        self._test_nd(k, n)

    def test_10d(self):
        k = 10
        n = 1000

        self._test_nd(k, n)

    def test_25d(self):
        k = 25
        n = 1000

        self._test_nd(k, n)

    def test_50d(self):
        k = 50
        n = 1000

        self._test_nd(k, n)

    def test_100d(self):
        k = 100
        n = 1000

        self._test_nd(k, n)

    @staticmethod
    def _test_nd(k, n):
        tree = KDQTree(k, split_size=5, bounds=np.vstack(([5.0] * k, [-5.0] * k)))
        coords = np.random.randn(n, k)
        tree.insert_batch(coords)

        assert sum(tree.to_hist()) == n

        return tree

    @staticmethod
    def _null_generator(n, k):
        def gen():
            w1 = np.random.randn(n, k)
            w2 = np.random.randn(n, k)
            return w1, w2

        return gen

    @staticmethod
    def _alternative_generator(n, k):
        def gen():
            w1 = np.random.randn(n, k)

            half = round(k / 2)
            change_half_features = np.hstack((5 * np.ones((1, half)), np.ones((1, half))))

            w2 = np.random.randn(n, k) * change_half_features
            return w1, w2

        return gen

    def test_ekdq(self):
        """
        Tests the EKDQ detector on some fairly easy test cases it should ace...

        50 dimensions, change in 50% of features
        """

        k = 50
        n = 500

        np.random.seed(0)

        bootstrap = np.random.randn(n, k)
        detector = EKDQDetector(k, 2, n, bounds=np.vstack(([-10] * k, [10] * k)))
        detector.bootstrap(bootstrap)

        c_n, st_n = self._run_n_times(detector, 5, self._null_generator(n, k))
        c_a, st_a = self._run_n_times(detector, 5, self._alternative_generator(n, k))

        assert not any(c_n), "False positive on easy problem"
        assert all(c_a), "False negative on easy problem"

    def test_kl(self):

        """
        Tests Dasu et al's KL detector on some fairly easy test cases it should ace...

        50 dimensions, change in 50% of features
        """

        k = 50
        n = 500

        np.random.seed(0)

        print()

        bootstrap = np.random.randn(n * 10, k)
        detector = KLDetector(k, 2, bounds=np.vstack(([-10, -10], [10, 10])))
        detector.bootstrap(bootstrap, n, 50)

        c_n, st_n = self._run_n_times(detector, 5, self._null_generator(n, k))
        c_a, st_a = self._run_n_times(detector, 5, self._alternative_generator(n, k))

        assert not any(c_n), "False positive on easy problem"
        assert all(c_a), "False negative on easy problem"

    def _run_n_times(self, detector, n, generator):

        changes, stats = [], []

        for i in range(0, n):
            w1, w2 = generator()
            change, st = detector.test(w1, w2)

            changes.append(change)
            stats.append(st)

            print(f"st={st}")
            print("Change" if change else "No change" + " was detected")

        return changes, stats


if __name__ == '__main__':
    unittest.main()
