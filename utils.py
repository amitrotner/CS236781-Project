import numpy as np
import sklearn.metrics


# Metrics class was copied from DCEC article authors repository (link in README)
class metrics:
    nmi = sklearn.metrics.normalized_mutual_info_score
    ari = sklearn.metrics.adjusted_rand_score
    acc = sklearn.metrics.accuracy_score
