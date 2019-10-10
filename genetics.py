import numpy as np

def basic_crossover(config, survivors, crosses):
    k = 0
    for i in range(1, survivors.shape[0]):
        for j in range(1, survivors.shape[1]):
            crosses[k, 0:j] = survivors[0,0:j]
            crosses[k, j:]  = survivors[i,j:]
            crosses[k+1, 0:j] = survivors[i,0:j]
            crosses[k+1, j:]  = survivors[0,j:]
            k += 2
    return crosses[0:k,:]
    
def basic_mutation(config, crosses):
    p = np.random.rand(crosses.shape[0])
    mutation_probability = 0.15
    search_space = config["search_space"]
    search_space_span = search_space[1,:] - search_space[0,:]
    selection = p >= mutation_probability
    mutation_selection = crosses[selection, :]
    mutation_selection[:,:] = mutation_selection +  0.01 * search_space_span * (np.random.rand(mutation_selection.shape[0], crosses.shape[1]) - 0.5) 
    np.clip(mutation_selection, search_space[0,:], search_space[1,:], out=mutation_selection)
    crosses[selection] = mutation_selection
    return crosses

def basic_survivors(config, population, fitness, survivors):
    max_survivors = config["max_survivors"]
    n_survivors = min(max_survivors, fitness.shape[0])
    sort_index = np.argsort(fitness)[0:n_survivors]
    survivors[0:n_survivors,:] = population[sort_index,:]
    return survivors[0:n_survivors,:]

class GeneticSolver():
    def __init__(self, fitness_function, search_space, config=None):
        if config is None :
            config = {
                "max_survivors" : 5,
                "max_childs" : 2 * (search_space.shape[1] - 1) * 4,
                "crossover_function": basic_crossover,
                "mutation_function": basic_mutation,
                "survivors_function": basic_survivors,
            }
        config["search_space"] = search_space
        self.config = config
        self.fitness_function = fitness_function
        self.survivors_space = np.zeros((self.config["max_survivors"], config["search_space"].shape[1]))
        self.childs_space = np.zeros((self.config["max_survivors"] + self.config["max_childs"], config["search_space"].shape[1]))

    def solve(self, fitness_function_args, N = 10000, tol = 1e-10, max_equals_steps=200):
        search_space = self.config["search_space"]
        max_survivors = self.config["max_survivors"]
        search_space_span = search_space[1,:] - search_space[0,:]
        self.survivors_space = np.random.rand(max_survivors, search_space.shape[1]) * search_space_span + search_space[0,:]
        self.survivors_space = self.survivors_space[np.argsort(self.fitness_function(self.survivors_space, *fitness_function_args)) ,:]
        crossover_function = self.config["crossover_function"]
        mutation_function = self.config["mutation_function"]
        survivors_function = self.config["survivors_function"]
        survivors = self.survivors_space
        best_fit = self.fitness_function(survivors[0:1,:], *fitness_function_args)[0]
        not_best_count = 0
        for _ in range(N):
            nm_childs = crossover_function(self.config, survivors, self.childs_space)
            childs = mutation_function(self.config, nm_childs)
            self.childs_space[0: childs.shape[0],:] = childs
            self.childs_space[childs.shape[0]: childs.shape[0] + survivors.shape[0],:] = survivors
            fitness = self.fitness_function(self.childs_space[0 : childs.shape[0] + survivors.shape[0],:] , *fitness_function_args)
            survivors = survivors_function(self.config, self.childs_space, fitness, self.survivors_space)
            new_best_fit = self.fitness_function(survivors[0:1,:], *fitness_function_args)[0]
            if abs(best_fit - new_best_fit) < tol:
                not_best_count += 1
                if not_best_count > max_equals_steps:
                    break
            else:
                not_best_count = 0
            best_fit = min(best_fit, new_best_fit)
        return survivors[0,:]
