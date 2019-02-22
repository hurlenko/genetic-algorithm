import numpy as np
import itertools as it


def distance_h(a, b):
    return sum(x[i] != y[i] for x, y in zip(a.chromosomes(), b.chromosomes()) for i in range(len(x)))


def distance_e(a, b):
    return abs(a.fitness - b.fitness)


def get_pair(arr, cmp, tp=True):
    a = np.random.choice(arr)
    b = sorted(list(arr), key=lambda x: cmp(a, x))
    b = b[1] if tp is True else b[-1]
    return [a, b]


class parent:
    @staticmethod
    def panmixia(parent_vector, size=None):
        pair_size = 2
        size = size or len(parent_vector) // pair_size
        return np.random.choice(parent_vector, (size, pair_size))
    
    @staticmethod
    def inbreeding_e(parent_vector, size=None):
        """
        Inbreeding that uses Euclidean distance
        """
        pair_size = 2
        size = size or len(parent_vector) // pair_size
        return [get_pair(parent_vector, distance_e) for _ in range(size)]

    @staticmethod
    def outbreeding_e(parent_vector, size=None):
        """
        Inbreeding that uses Euclidean distance
        """
        pair_size = 2
        size = size or len(parent_vector) // pair_size
        return [get_pair(parent_vector, distance_e, False) for _ in range(size)]
    
    @staticmethod
    def inbreeding_h(parent_vector, size=None):
        """
        Inbreeding that uses Hamming distance
        """
        pair_size = 2
        size = size or len(parent_vector) // pair_size
        return [get_pair(parent_vector, distance_h) for _ in range(size)]

    @staticmethod
    def outbreeding_h(parent_vector, size=None):
        """
        Inbreeding that uses Hamming distance
        """
        pair_size = 2
        size = size or len(parent_vector) // pair_size
        return [get_pair(parent_vector, distance_h, False) for _ in range(size)]

    @staticmethod
    def tournament(parent_vector, size=None):
        pair_size = 2
        size = size or len(parent_vector) // pair_size
        f = lambda: min(np.random.choice(parent_vector, pair_size), key=lambda x: x.fitness)
        return [[f(), f()] for _ in range(size)]

    @staticmethod
    def roulette_wheel(parent_vector, size=None):
        pair_size = 2
        size = size or len(parent_vector) // pair_size
        parent_vector = sorted(parent_vector,
                               key=lambda x: x.fitness,
                               reverse=True)
        s = np.sum(float("{:f}".format(x.fitness)) for x in parent_vector)
        prob = None if s == 0 else [float("{:f}".format(x.fitness)) / s for x in parent_vector][::-1]
        return np.random.choice(parent_vector, size=(size, pair_size), p=prob)


class offspring:
    @staticmethod
    def truncution(size=None, **kwargs):
        bound = 0.5
        parents = kwargs["par"]
        offspr = kwargs["offspr"]
        generation = list(it.chain.from_iterable(parents)) + offspr
        generation = sorted(generation, key=lambda x: x.fitness)
        size = size or len(offspr)
        generation = generation[:int(len(generation) * bound)]
        return np.random.choice(generation, size)

    @staticmethod
    def elite(size=None, **kwargs):
        parents = kwargs["par"]
        offspr = kwargs["offspr"]
        generation = list(it.chain.from_iterable(parents)) + offspr
        generation = sorted(generation, key=lambda x: x.fitness)
        size = size or len(offspr)
        return generation[:size]

    @staticmethod
    def exclusion(size=None, **kwargs):
        parents = kwargs["par"]
        offspr = kwargs["offspr"]
        size = size or len(offspr)
        generation = list(it.chain.from_iterable(parents)) + offspr
        generation = sorted(generation, key=lambda x: x.fitness)
        seen = set()
        seen_add = seen.add
        unique = sorted([x for x in generation if not (x in seen or seen_add(x))], key=lambda x: x.fitness)
        if len(unique) < size:
            unique += generation[:size - len(unique)]
        return unique[:size]

    @staticmethod
    def bolzmann(size=None, **kwargs):
        parents = kwargs["par"]
        offspr = kwargs["offspr"]
        size = size or len(offspr)
        pair_size = 2
        t = 30
        generation = np.array(list(it.chain.from_iterable(parents)) + offspr)
        generation = np.random.choice(generation, size=(size, pair_size))
        return [x if np.random.random_sample() > 1 / (1 + np.exp((x.fitness - y.fitness) / t))
                else y for x, y in generation]
