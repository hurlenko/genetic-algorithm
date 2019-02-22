import numpy as np


def single_point(individuum, probability):
    chromosomes = [list(x) for x in individuum]
    new_chrs = []
    for c in chromosomes:
        if np.random.random_sample() < probability:
            index = np.random.randint(1, len(c))
            c[index] = "1" if c[index] == "0" else "0"
        new_chrs.append(int("".join(c), 2))
    return new_chrs
