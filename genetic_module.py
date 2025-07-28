import random
import numpy as np

def mutate(nums):
    nums = nums.copy()
    idx = random.randint(0, 6)
    nums[idx] = random.randint(1, 37)
    return sorted(set(nums))[:7]

def crossover(p1, p2):
    child = sorted(set(p1[:3] + p2[3:]))
    while len(child) < 7:
        r = random.randint(1, 37)
        if r not in child:
            child.append(r)
    return sorted(child)

def evolve(pop, score_fn, gen=5):
    for _ in range(gen):
        pop.sort(key=score_fn, reverse=True)
        elite = pop[:5]
        children = []
        for _ in range(25):
            a, b = random.sample(elite, 2)
            children.append(mutate(crossover(a, b)))
        pop = elite + children
    return pop[:5]