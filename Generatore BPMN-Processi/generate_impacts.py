import numpy as np
import random
from scipy.spatial.distance import cosine
import matplotlib.pyplot as plt

def generate_vectors(num_vec, dim, mode="random"):
    vectors = []
    for _ in range(num_vec):
        vector = np.random.random(dim)
        if mode != "random":
            indexes = [random.randint(0, dim-1) for _ in range(dim)]
            if mode == "bagging_divide":
                for i in indexes:
                    vector[i] /= 10
            elif mode == "bagging_remove":
                new_vector = np.zeros(dim)
                for i in indexes:
                    new_vector[i] = vector[i]
                vector = new_vector
            elif mode == "bagging_remove_divide":
                new_vector = np.zeros(dim)
                for i in indexes:
                    new_vector[i] = vector[i] / 10
                vector = new_vector
            elif mode == "bagging_remove_reverse":
                new_vector = vector.copy()
                for i in indexes:
                    new_vector[i] = 0
                vector = new_vector
            elif mode == "bagging_remove_reverse_divide":
                new_vector = vector.copy()
                for i in indexes:
                    new_vector[i] = 0
                non_zero_indexes = np.nonzero(new_vector)[0]
                if len(non_zero_indexes) > 0:
                    divide_indexes = [random.choice(non_zero_indexes) for _ in range(dim)]
                    for i in divide_indexes:
                        new_vector[i] /= 10
                vector = new_vector
        vectors.append(vector)
    return vectors

