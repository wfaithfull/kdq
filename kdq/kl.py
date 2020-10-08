from kdq import KDQTree
import numpy as np


class KLDetector:

    def __init__(self, k, split_size, bounds=None):
        if bounds is not None:
            self.tree = KDQTree(k, split_size, bounds=bounds)
        else:
            self.tree = KDQTree(k, split_size)

        self.threshold = None

    def bootstrap(self, sample, n, resamples):

        assert sample.shape[0] > 2 * n, "You need to have a bigger sample than 2*n, preferably much bigger..."

        estimates = []

        for i in range(0, resamples):
            subsample = sample[np.random.choice(sample.shape[0], 2*n), :]

            w1 = subsample[:n, :]
            w2 = subsample[n:, :]

            estimates.append(self.dist(w1, w2))

        self.threshold = np.percentile(estimates, 95)

    def dist(self, w1, w2):
        self.tree.insert_batch(w1)
        w1_dist = self.tree.to_hist()
        w2_dist = self.tree.in_structure_hist(w2)
        self.tree.clear()
        return self.KL(w1_dist, w2_dist)

    def test(self, w1, w2):
        assert self.threshold, "Detector must be bootstrapped"
        dist = self.dist(w1, w2)
        return dist > self.threshold, dist

    @staticmethod
    def KL(a, b):
        a = np.asarray(a, dtype=np.float)
        b = np.asarray(b, dtype=np.float)

        eps = np.spacing(1.0)

        a[a == 0] = eps
        b[b == 0] = eps

        return np.sum(a * np.log(a / b))
