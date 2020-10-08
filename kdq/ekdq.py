import numpy as np
from scipy.stats import chi2

from kdq import KDQTree
from kdq.tree import Type


def to_trinomial(results):
    """
    Convert the EKDQ event stream into a 3-parameter multinomial

    Since many events may occur in response to one insert/delete, we need to squash these to make a simple multinomial.

    For example, the real stream might look like:

    MERGE
    MERGE                    SPLIT
    MERGE, SPLIT, UNCHANGED, SPLIT, MERGE, UNCHANGED

    We convert to:

    MERGE, SPLIT, UNCHANGED, SPLIT, MERGE, UNCHANGED

    But in numerical form;

    2, 1, 0, 1, 2, 0
    """
    all_events = [result[1] for result in results]
    events = [next((x.type.value for x in events if x.type != Type.UNCHANGED), Type.UNCHANGED.value)
              for events in all_events]
    return np.array(events)


def to_count_multinomial(results):
    all_events = [result[1] for result in results]
    events = [len(events) for events in all_events]
    return np.array(events)


def to_depth_multinomial(results):
    all_events = [result[1] for result in results]
    events = [min([e.node.depth if e.node else 0 for e in events]) for events in all_events]
    return np.array(events)


class EKDQDetector:

    def to_multinomial(self, results):
        return to_trinomial(results)

    def __init__(self, k, split_size, max_size, bounds=None, significance=0.05):
        if bounds is not None:
            self.tree = KDQTree(k, split_size, bounds=bounds, max_size=max_size)
        else:
            self.tree = KDQTree(k, split_size, max_size=max_size)

        self.significance = significance

    def bootstrap(self, sample):
        self.tree.insert_batch(sample)

    def test(self, w1, w2):
        assert w1.shape[1] == self.tree.k, "Data dimensionality must match tree dimensionality"
        assert w1.shape[1] == w2.shape[1], "W1 and W2 must have the same dimensions"
        assert self.tree.root.weight > 0, "Need to bootstrap the tree first with a sample"

        results = self.tree.insert_batch(w1)
        vector_events_w1 = to_trinomial(results)

        results = self.tree.insert_batch(w2)
        vector_events_w2 = to_trinomial(results)

        all_events = np.hstack((vector_events_w1, vector_events_w2))
        M = np.unique(all_events).shape[0]

        y1, _ = np.histogram(vector_events_w1, M)
        n1 = vector_events_w1.shape[0]

        y2, _ = np.histogram(vector_events_w2, M)

        # multinomial parameters estimated from W1
        n, p = n1, np.array(y1 / n1)
        p[p == 0] = np.finfo(float).eps

        # https://en.wikipedia.org/wiki/G-test
        #
        # The G-test is a simplified log-likelihood ratio test for multinomials.
        # "Usually", you would do something like the following:
        #
        # # Null hypothesis
        # ll1 = ll(y1, p)
        #
        # # Alternate hypothesis
        # ll2 = ll(y2, p)
        #
        # # Arrive at the asymptotic scaled log-likelihood ratio
        # scaled_llr = -2 * (ll1 - ll2)
        #
        # However, this actually *is* the log-likelihood ratio - this just simplifies for multinomials because the MLE
        # for multinomials is just the counts we observe for each category.
        g = 2 * np.sum(np.multiply(y2, np.log(y2 / y1)))

        # Under H0, this should be asymptotically χ2, with degrees of freedom equal to size of the parameter space - 1
        pst = chi2.sf(g, df=M - 1)

        # If the probability that a χ2 distribution takes this value is less than significance, reject null hypothesis
        return pst < self.significance, g
