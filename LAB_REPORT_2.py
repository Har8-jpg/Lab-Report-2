# ==========================================
# Lab Report 2: BSD2513 - Search Algorithms
# Student ID: SD23005
# ==========================================

import math
import random
from dataclasses import dataclass
from typing import Callable, List, Tuple

import numpy as np
import pandas as pd
import streamlit as st

# -------------------- Problem Definition --------------------
@dataclass
class GAProblem:
    name: str
    chromosome_type: str  
    dim: int
    bounds: Tuple[float, float] | None
    fitness_fn: Callable[[np.ndarray], float]


def make_bitpattern(dim: int, target_ones: int) -> GAProblem:
    """
    Fitness peaks (value 80) when number of ones == target_ones (50)
    Decreases as it deviates from target.
    """
    def fitness(x: np.ndarray) -> float:
        ones = np.sum(x)
        return 80 - abs(target_ones - ones)

    return GAProblem(
        name=f"Bit Pattern Optimization ({dim} bits, target={target_ones})",
        chromosome_type="bit",
        dim=dim,
        bounds=None,
        fitness_fn=fitness,
    )

# -------------------- GA Operators --------------------
def init_population(problem: GAProblem, pop_size: int, rng: np.random.Generator) -> np.ndarray:
    return rng.integers(0, 2, size=(pop_size, problem.dim), dtype=np.int8)

def tournament_selection(fitness: np.ndarray, k: int, rng: np.random.Generator) -> int:
    idxs = rng.integers(0, fitness.size, size=k)
    return int(idxs[np.argmax(fitness[idxs])])

def one_point_crossover(a: np.ndarray, b: np.ndarray, rng: np.random.Generator) -> Tuple[np.ndarray, np.ndarray]:
    if a.size <= 1:
        return a.copy(), b.copy()
    point = int(rng.integers(1, a.size))
    return np.concatenate([a[:point], b[point:]]), np.concatenate([b[:point], a[point:]])

def bit_mutation(x: np.ndarray, mut_rate: float, rng: np.random.Generator) -> np.ndarray:
    mask = rng.random(x.shape) < mut_rate
    y = x.copy()
    y[mask] = 1 - y[mask]
    return y

def evaluate(pop: np.ndarray, problem: GAProblem) -> np.ndarray:
    return np.array([problem.fitness_fn(ind) for ind in pop], dtype=float)

# -------------------- GA Execution --------------------
def run_ga(problem: GAProblem,
           pop_size: int = 300,
           generations: int = 50,
           crossover_rate: float = 0.9,
           mutation_rate: float = 0.01,
           tournament_k: int = 3,
           elitism: int = 2,
           seed: int | None = 42,
           stream_live: bool = True):

    rng = np.random.default_rng(seed)
    pop = init_population(problem, pop_size, rng)
    fit = evaluate(pop, problem)

    chart_area = st.empty()
    best_area = st.empty()

    history_best, history_avg, history_worst = [], [], []

    for gen in range(generations):
        best_idx = int(np.argmax(fit))
        best_fit = float(fit[best_idx])
        avg_fit = float(np.mean(fit))
        worst_fit = float(np.min(fit))
        history_best.append(best_fit)
        history_avg.append(avg_fit)
        history_worst.append(worst_fit)

        if stream_live:
            df = pd.DataFrame({"Best": history_best, "Average": history_avg, "Worst": history_worst})
            chart_area.line_chart(df)
            best_area.markdown(f"Generation {gen+1}/{generations} — Best fitness: **{best_fit:.4f}**")

        # Elitism
        E = max(0, min(elitism, pop_size))
        elite_idx = np.argpartition(fit, -E)[-E:] if E > 0 else np.array([], dtype=int)
        elites = pop[elite_idx].copy() if E > 0 else np.empty((0, pop.shape[1]))

        next_pop = []
        while len(next_pop) < pop_size - E:
            i1, i2 = tournament_selection(fit, tournament_k, rng), tournament_selection(fit, tournament_k, rng)
            p1, p2 = pop[i1], pop[i2]

            if rng.random() < crossover_rate:
                c1, c2 = one_point_crossover(p1, p2, rng)
            else:
                c1, c2 = p1.copy(), p2.copy()

            c1 = bit_mutation(c1, mutation_rate, rng)
            c2 = bit_mutation(c2, mutation_rate, rng)
            next_pop.extend([c1, c2])

        pop = np.vstack([np.array(next_pop[: pop_size - E]), elites]) if E > 0 else np.array(next_pop[:pop_size])
        fit = evaluate(pop, problem)

    best_idx = int(np.argmax(fit))
    best = pop[best_idx].copy()
    best_fit = float(fit[best_idx])
    df = pd.DataFrame({"Best": history_best, "Average": history_avg, "Worst": history_worst})
    return {"best": best, "best_fitness": best_fit, "history": df, "final_population": pop, "final_fitness": fit}

# -------------------- Streamlit UI --------------------
st.set_page_config(page_title="Genetic Algorithm - Bit Pattern", page_icon="🧬", layout="wide")
st.title("Genetic Algorithm (GA) - Bit Pattern Optimization")
st.caption("Task: Generate an 80-bit pattern that maximizes fitness when number of ones = 50")

# Fixed parameters based on the question
dim = 80
target_ones = 50
problem = make_bitpattern(dim, target_ones)

with st.sidebar:
    st.header("GA Parameters")
    st.write(f"Chromosome length: {dim} bits")
    st.write(f"Target number of ones: {target_ones}")
    st.write("Fitness = 80 − |50 − actual_ones|")
    pop_size = 300
    generations = 50
    crossover_rate = 0.9
    mutation_rate = 0.01
    tournament_k = 3
    elitism = 2
    seed = 42
    live = True

if st.button("Run Genetic Algorithm", type="primary"):
    result = run_ga(problem, pop_size, generations, crossover_rate, mutation_rate, tournament_k, elitism, seed, live)

    st.subheader("Fitness Over Generations")
    st.line_chart(result["history"])

    st.subheader("Best Solution")
    st.write(f"Best fitness: {result['best_fitness']:.2f}")
    bitstring = ''.join(map(str, result["best"].astype(int).tolist()))
    st.code(bitstring, language="text")
    st.write(f"Number of ones: {int(np.sum(result['best']))} / {problem.dim}")