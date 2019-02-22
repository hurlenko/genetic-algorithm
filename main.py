import numpy as np
import itertools as it
import selector
import crossover
import mutator


class GA(object):
    def __init__(self,
                 u_bound,
                 prob_cross,
                 prob_mut,
                 cross,
                 mutation,
                 selection_parents,
                 selection_offspring,
                 population=None):
        self.l_bound = 0
        self.h_bound = u_bound
        self.population = population
        self.prob_cross = prob_cross
        self.prob_mut = prob_mut
        self.f_cross = cross
        self.f_mutation = mutation
        self.f_selector_parents = selection_parents
        self.f_selector_offspring = selection_offspring
        self.parents = None
        self.offspring = None

    def generate_population(self, size, fitness_vector_size):
        population = np.random.randint(self.l_bound,
                                       self.h_bound,
                                       (size, fitness_vector_size))
        self.population = [Individual(p) for p in population]

    def select_parents(self):
        self.parents = self.f_selector_parents(self.population)

    def perform_crossover(self):
        self.offspring = list(it.chain.from_iterable([self.f_cross(*x, self.prob_cross) for x in self.parents]))

    def mutate(self):
        self.offspring = [Individual(self.f_mutation(x, self.prob_mut)) for x in self.offspring]

    def select_offspring(self):
        self.offspring = self.f_selector_offspring(par=self.parents, offspr=self.offspring)

    def set_population(self, new_popul=None):
        self.population = self.offspring if new_popul is None else new_popul


class Individual(object):
    def __init__(self, phenotypes):
        self.phenotypes = phenotypes  # phenotype
        self.fitness = fitness_func(self.phenotypes)  # value of the fitness function

    def chromosomes(self):
        chrom = []
        for phenotype in self.phenotypes:
            if phenotype < 0:
                num = '-{0}' + str(bin(phenotype))[3:]
            else:
                num = '+{0}' + str(bin(phenotype))[2:]
            chrom.append(num.format((bit_num - len(num[4:])) * '0'))
        return chrom

    def __str__(self):
        return '{0} = {1}'.format(real(self.phenotypes), self.fitness)

def real(x):
    return [y * eps + interval[0] for y in x]


def fitness_func(arg_vec):
    arg_vec = real(arg_vec)
    # Sphere model (DeJong1)
    # return np.sum([x ** 2 for x in arg_vec])
    # Rosenbrock's saddle (DeJong2)
    # return sum([(100 * (xj - xi ** 2) ** 2 + (xi - 1) ** 2) for xi, xj in zip(arg_vec[:-1], arg_vec[1:])])
    # Rastrigin's function
    # return 10 * len(arg_vec) + np.sum([x ** 2 - 10 * np.cos(2 * np.pi * x) for x in arg_vec])
    # Ackley's Function
    s1 = -0.2 * np.sqrt(np.sum([x ** 2 for x in arg_vec]) / len(arg_vec))
    s2 = np.sum([np.cos(2 * np.pi * x) for x in arg_vec]) / len(arg_vec)
    return 20 + np.e - 20 * np.exp(s1) - np.exp(s2)


# Initial values
#
# fitness_func      - fitness_func to be evaluated
# interval          - fitness function interval
# p_c               - crossover probability
# p_b               - mutation probability
# population_size   - the size of desirable population
# ff_vec_size       - number of arguments that is given to fitness_func
# max_epochs        - number of iteration to be evaluated

interval = (-5.12, 5.12)
eps = 1E-3
p_c = 1.0
p_m = 0.4
population_size = 20
ff_vec_size = 5
max_epochs = 1000

bit_num = int(np.log2((interval[1] - interval[0]) / eps) + 1)


def main():
    upper_bound = 2 ** bit_num
    global eps
    eps = (interval[1] - interval[0]) / upper_bound
    ga = GA(upper_bound,
            p_c,  # crossover prob
            p_m,  # mutation prob
            crossover.reduced_surrogate,
            mutator.single_point,
            selector.parent.inbreeding_h,
            selector.offspring.elite
)

    ga.generate_population(population_size, ff_vec_size)
    print('Initial population')
    for ind in sorted(ga.population, key=lambda x: x.fitness):
        print(ind, ind.chromosomes())
    for i in range(max_epochs):
        ga.select_parents()
        ga.perform_crossover()
        ga.mutate()
        ga.select_offspring()
        ga.set_population()
        print('{0}/{1} Current population:'.format(i + 1, max_epochs))
        print(sorted(ga.population, key=lambda x: x.fitness)[0])

if __name__ == '__main__':
    main()
