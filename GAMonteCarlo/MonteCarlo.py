import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from collections import defaultdict
import random
from multiprocessing import Pool, cpu_count
import argparse
import sys

# -------------------- PARAMETERS --------------------
grid_size = 2
area_w = 12.0
area_h = 12.0

# disease params
incubation_days = 8
trans_distance = 0.1 # distance between infected and susceptable
beta = 0.25        # chance that [S]usceptible will be [E]xposed
sigma = 1/5.2      # TODO: chance that [E]xposed will be [I]nfected after incubation_days
gamma = 1/21.0     # recovery rate
mu = 0.005         # TODO: chance of [I]nfected to [D]ead

# mobility
move_scale = 0.6
area_border_cross_prob = 0.01

# Genetic Algorithm & Monte Carlo parameters
# `population_size` is now handled by the command-line argument.
mutation_rate = 0.1        # Probability of a mutation occurring
num_processes = cpu_count() # Automatically use all available CPU cores

# performance bins
bin_size = 1.8
random_seed = 42

# -------------------- END PARAMETERS --------------------
np.random.seed(random_seed)
random.seed(random_seed)

n_areas = grid_size * grid_size

# states
S, E, I, R, D, V = 0, 1, 2, 3, 4, 5
state_names = ['S','E','I','R','D','V']
state_colors = ['tab:blue','tab:orange','tab:red','tab:green','k','tab:purple']

# area coords
area_coords = {}
for idx in range(n_areas):
    col = idx % grid_size
    row = idx // grid_size
    area_coords[idx] = (col*area_w, row*area_h)

class Agent:
    __slots__ = ('id','area','home_area','x','y','state','days','vaccinated')
    def __init__(self, aid, area):
        self.id = aid
        self.area = area      # Agent's current area
        self.home_area = area # Agent's original, permanent home area
        x0,y0 = area_coords[area]
        self.x = np.random.uniform(x0, x0 + area_w)
        self.y = np.random.uniform(y0, y0 + area_h)
        self.state = S
        self.days = 0
        self.vaccinated = False

    def step_move(self):
        if self.state == D: return
        dx = np.random.uniform(-move_scale, move_scale)
        dy = np.random.uniform(-move_scale, move_scale)
        new_x = self.x + dx
        new_y = self.y + dy
        if np.random.rand() < area_border_cross_prob:
            col = self.area % grid_size
            row = self.area // grid_size
            neigh_cols = [c for c in [col-1,col,col+1] if 0 <= c < grid_size]
            neigh_rows = [r for r in [row-1,row,row+1] if 0 <= r < grid_size]
            new_col = random.choice(neigh_cols)
            new_row = random.choice(neigh_rows)
            new_area = new_row * grid_size + new_col
            nx0, ny0 = area_coords[new_area]
            new_x = np.clip(self.x + np.random.uniform(-move_scale, move_scale), nx0, nx0+area_w)
            new_y = np.clip(self.y + np.random.uniform(-move_scale, move_scale), ny0, ny0+area_h)
            self.area = new_area
        else:
            x0,y0 = area_coords[self.area]
            new_x = np.clip(new_x, x0, x0 + area_w)
            new_y = np.clip(new_y, y0, y0 + area_h)
        self.x = new_x; self.y = new_y

def run_simulation(vaccine_distribution, area_population, steps, hesitancy_rate, vaccine_production_rate, vaccine_spoilage_rate, vaccine_total, vaccination_delay_days):
    agents = []
    aid = 0
    for area_idx, popn in enumerate(area_population):
        for _ in range(popn):
            agents.append(Agent(aid, area_idx))
            aid += 1

    initial_infected = max(1, int(0.01 * len(agents)))
    infected_idxs = np.random.choice(len(agents), initial_infected, replace=False)
    for idx in infected_idxs: agents[idx].state = I

    history_per_area = [ {s: [] for s in state_names} for _ in range(n_areas) ]
    history_total = {s: [] for s in state_names}

    # Per-area vaccine stock and history
    vaccine_stock = [0] * n_areas
    vaccine_stock_history = [[] for _ in range(n_areas)]

    agent_history = []
    day_counter = 0
    total_population = sum(area_population)

    def build_bins(agents):
        bins = defaultdict(list)
        for i, a in enumerate(agents):
            bx = int(a.x // bin_size)
            by = int(a.y // bin_size)
            bins[(bx,by)].append(i)
        return bins

    def nearby_candidates(a, bins):
        bx = int(a.x // bin_size)
        by = int(a.y // bin_size)
        res = []
        for dx in (-1,0,1):
            for dy in (-1,0,1):
                res.extend(bins.get((bx+dx, by+dy), []))
        return res

    def sim_step():
        nonlocal day_counter, vaccine_total
        day_counter += 1

        for a in agents: a.step_move()
        bins = build_bins(agents)

        for i, a in enumerate(agents):
            if a.state == I:
                cands = nearby_candidates(a, bins)
                for j in cands:
                    if j == i: continue
                    b = agents[j]
                    if b.state == S and b.area == a.area:
                        dx = a.x - b.x; dy = a.y - b.y
                        # this is pythagorean theoream a^2 + b^2 = c^2
                        if dx*dx + dy*dy <= trans_distance:
                            if np.random.rand() < beta:
                                b.state = E

        for a in agents:
            if a.state == E:
                if a.days >= incubation_days:
                    if np.random.rand() < sigma:
                        a.state = I; a.days = 0
                    else:
                        a.state = S; a.days = 0
            elif a.state == I:
                if np.random.rand() < mu:
                    a.state = D
                elif np.random.rand() < gamma:
                    a.state = R
            a.days += 1

        if day_counter > vaccination_delay_days:
            # Distribute production to each area's stock
            for aidx in range(n_areas):
                doses_to_distribute = int(vaccine_total * vaccine_distribution[aidx])

                # Check if enough vaccines are available
                if vaccine_total >= doses_to_distribute:
                    vaccine_stock[aidx] += doses_to_distribute
                    vaccine_total -= doses_to_distribute

                # Vaccinate from the local area stock
                if vaccine_stock[aidx] > 0:
                    susceptibles_in_area = [p for p in agents if p.state == S and p.area == aidx]
                    if susceptibles_in_area:
                        doses_to_use = min(len(susceptibles_in_area), vaccine_stock[aidx])
                        chosen = np.random.choice(susceptibles_in_area, doses_to_use, replace=False)
                        for p in chosen:
                            if np.random.rand() > hesitancy_rate:
                                p.state = V; p.vaccinated = True; vaccine_stock[aidx] -= 1

        # Handle spoilage per area
        for aidx in range(n_areas):
            spoiled = int(vaccine_stock[aidx] * vaccine_spoilage_rate)
            vaccine_stock[aidx] = max(0, vaccine_stock[aidx] - spoiled)

        # Record per-area vaccine stock history
        for aidx in range(n_areas):
            vaccine_stock_history[aidx].append(vaccine_stock[aidx])

        # Increase vaccine_total by vaccine_production_rate
        if day_counter > vaccination_delay_days and day_counter % vaccination_delay_days == 0:
            vaccine_total += vaccine_production_rate

        for aidx in range(n_areas):
            counts = [0]*6
            for p in agents:
                if p.area == aidx: counts[p.state] += 1
            for s_idx,name in enumerate(state_names):
                history_per_area[aidx][name].append(counts[s_idx])
        counts_tot = [0]*6
        for p in agents: counts_tot[p.state] += 1
        for s_idx,name in enumerate(state_names):
            history_total[name].append(counts_tot[s_idx])

        # Record full agent data for animation replay
        agent_data = [{'x': a.x, 'y': a.y, 'state': a.state} for a in agents]
        agent_history.append(agent_data)

    for _ in range(steps):
        sim_step()

    total_infected = history_total['I'][-1]
    total_deceased = history_total['D'][-1]

    # Return all data, including agent history, to be saved
    return {
        'history_per_area': history_per_area,
        'history_total': history_total,
        'vaccine_stock_history': vaccine_stock_history,
        'agent_history': agent_history
    }

def save_data(data, filename, distribution=None):
    """Saves the simulation results to a file."""
    # Convert list of dictionaries to a format that can be saved
    hist_per_area_dict = {}
    for i, area_hist in enumerate(data['history_per_area']):
        for sname, sdata in area_hist.items():
            hist_per_area_dict[f"area_{i}_{sname}"] = sdata

    np.savez_compressed(
        filename,
        history_per_area=hist_per_area_dict,
        history_total=data['history_total'],
        vaccine_stock_history=np.array(data['vaccine_stock_history']),
        agent_history=data['agent_history'],
        distribution=distribution
    )

def normalize_distribution(dist):
    """Ensures the sum of the distribution equals 1.0. Adds robustness for non-positive sums."""
    total = sum(dist)
    if total <= 0:
        return [1.0 / n_areas] * n_areas
    return [x / total for x in dist]

def initialize_population(population_size):
    population = []
    for _ in range(population_size):
        dist = np.random.rand(n_areas)
        population.append(normalize_distribution(dist))
    return population

def _evaluate_distribution_for_run(params):
    """Wrapper function to run a single Monte Carlo simulation for fitness evaluation."""
    distribution, area_population, steps, hesitancy_rate, vaccine_production_rate, vaccine_spoilage_rate, vaccine_total, vaccination_delay_days = params
    data = run_simulation(distribution, area_population, steps, hesitancy_rate, vaccine_production_rate, vaccine_spoilage_rate, vaccine_total, vaccination_delay_days)
    total_infected = data['history_total']['I'][-1]
    total_deceased = data['history_total']['D'][-1]
    
    # Return the fitness score and the full data object
    return (total_infected + total_deceased, data)

def fitness(distribution, area_population, steps, hesitancy_rate, vaccine_production_rate, vaccine_spoilage_rate, vaccine_total, vaccination_delay_days, monte_carlo_runs):
    """Calculates fitness by running Monte Carlo simulations in parallel."""
    params_list = [(distribution, area_population, steps, hesitancy_rate, vaccine_production_rate, vaccine_spoilage_rate, vaccine_total, vaccination_delay_days)] * monte_carlo_runs
    
    with Pool(num_processes) as p:
        # Get both scores and data
        results = p.map(_evaluate_distribution_for_run, params_list)
    
    scores = [res[0] for res in results]
    all_data = [res[1] for res in results]
    
    # Find the best run's data within this batch of Monte Carlo runs
    best_score_in_batch = min(scores)
    best_data_in_batch = all_data[scores.index(best_score_in_batch)]
    
    return sum(scores) / monte_carlo_runs, best_data_in_batch

def select_parents(population, fitness_scores):
    # Tournament selection
    parents = []
    for _ in range(2):
        contenders = random.sample(list(zip(population, fitness_scores)), min(5, options.population_size))
        winner = min(contenders, key=lambda x: x[1])[0]
        parents.append(winner)
    return parents[0], parents[1]

def crossover(parent1, parent2):
    crossover_point = random.randint(1, n_areas - 1)
    child = parent1[:crossover_point] + parent2[crossover_point:]
    return normalize_distribution(child)

def mutate(child, mutation_rate):
    if random.random() < mutation_rate:
        idx1, idx2 = random.sample(range(n_areas), 2)
        amount = random.uniform(0.01, 0.1) # Mutate by a small percentage
        child[idx1] += amount
        child[idx2] = max(0.0, child[idx2] - amount)
    return normalize_distribution(child)

def run_ga(area_population, steps, generations, population_size, hesitancy_rate, vaccine_production_rate, vaccine_spoilage_rate, vaccine_total, vaccination_delay_days, monte_carlo_runs):
    print("Starting genetic algorithm to find optimal vaccine distribution...")
    population = initialize_population(population_size)
    best_fitness_history = []

    for generation in range(generations):
        fitness_scores = []
        best_data_this_gen = None
        best_fitness_this_gen = float('inf')

        for dist in population:
            # Call the modified fitness function
            avg_fitness, best_data_in_run = fitness(dist, area_population, steps, hesitancy_rate, vaccine_production_rate, vaccine_spoilage_rate, vaccine_total, vaccination_delay_days, monte_carlo_runs)
            fitness_scores.append(avg_fitness)

            # Track the single best run's data for this generation
            if avg_fitness < best_fitness_this_gen:
                best_fitness_this_gen = avg_fitness
                best_data_this_gen = best_data_in_run

        best_fitness = min(fitness_scores)
        best_dist = population[fitness_scores.index(best_fitness)]
        best_fitness_history.append(best_fitness)

        save_data(best_data_this_gen, f"gen_{generation+1}_fitness_{best_fitness_this_gen:.2f}.npz", distribution=best_dist)

        print(f"Generation {generation+1}/{generations}: Best Fitness = {best_fitness:.2f}, Best Dist = {[f'{x:.2f}' for x in best_dist]}")

        new_population = [best_dist] # Elitism

        while len(new_population) < population_size:
            parent1, parent2 = select_parents(population, fitness_scores)
            child = crossover(parent1, parent2)
            mutated_child = mutate(child, mutation_rate)
            new_population.append(mutated_child)

        population = new_population

    print("\nGenetic Algorithm complete. Finding optimal and worst distribution...")

    # Run and save final best and worst runs for replay
    final_results = [fitness(dist, area_population, steps, hesitancy_rate, vaccine_production_rate, vaccine_spoilage_rate, vaccine_total, vaccination_delay_days, monte_carlo_runs) for dist in population]
    final_fitness_scores_only = [score for score, data in final_results]

    optimal_distribution_index = final_fitness_scores_only.index(min(final_fitness_scores_only))
    worst_distribution_index = final_fitness_scores_only.index(max(final_fitness_scores_only))

    optimal_distribution = population[optimal_distribution_index]
    worst_distribution = population[worst_distribution_index]
    best_run_data = final_results[optimal_distribution_index][1]
    worst_run_data = final_results[worst_distribution_index][1]

    print(f"\nOptimal Vaccine Distribution found: {[f'{x:.2f}' for x in optimal_distribution]}")
    print(f"\nWorst Vaccine Distribution found: {[f'{x:.2f}' for x in worst_distribution]}")

    # Save final best and worst runs for replay
    print("\nSaving final best and worst strategies...")
    save_data(best_run_data, "best_strategy_results.npz", distribution=optimal_distribution)
    save_data(worst_run_data, "worst_strategy_results.npz", distribution=worst_distribution)

    print("Simulation data for best and worst strategies saved.")

    return best_fitness_history

def parse_args():
    parser = argparse.ArgumentParser(description="Run the Monte Carlo/Genetic Algorithm simulation.")
    parser.add_argument("--area_population", type=str, default="100,100,100,100", help="Comma-separated list of population for each area.")
    parser.add_argument("--steps", type=int, default=365, help="Number of simulation steps (days).")
    parser.add_argument("--generations", type=int, default=30, help="Number of genetic algorithm generations.")
    parser.add_argument("--monte_carlo_runs", type=int, default=10, help="Number of simulations to run per strategy for a robust average.")
    parser.add_argument("--population_size", type=int, default=50, help="Number of individuals in the genetic algorithm population.")
    parser.add_argument("--hesitancy_rate", type=float, default=0.2, help="Rate of vaccine hesitancy.")
    parser.add_argument("--vaccine_production_rate", type=int, default=4, help="Daily vaccine production rate.")
    parser.add_argument("--vaccine_spoilage_rate", type=float, default=0.01, help="Daily vaccine spoilage rate.")
    parser.add_argument("--vaccine_total", type=int, default=200, help="Initial total vaccine stock.")
    parser.add_argument("--vaccination_delay_days", type=int, default=0, help="Days before vaccination begins.")
    return parser.parse_args()


if __name__ == "__main__":
    global options
    options = parse_args()

    # Update parameters with command-line arguments
    area_population = [int(p) for p in options.area_population.split(',')]
    steps = options.steps
    generations = options.generations
    monte_carlo_runs = options.monte_carlo_runs
    population_size = options.population_size
    hesitancy_rate = options.hesitancy_rate
    vaccine_production_rate = options.vaccine_production_rate
    vaccine_spoilage_rate = options.vaccine_spoilage_rate
    vaccine_total = options.vaccine_total
    vaccination_delay_days = options.vaccination_delay_days

    total_population = sum(area_population)

    best_fitness_history = run_ga(area_population, steps, generations, population_size, hesitancy_rate, vaccine_production_rate, vaccine_spoilage_rate, vaccine_total, vaccination_delay_days, monte_carlo_runs)

    try:
        np.savez_compressed('genetic_algorithm_performance_data.npz', best_fitness_history=np.array(best_fitness_history))
        print("Best fitness history saved to genetic_algorithm_performance_data.npz")
    except Exception as e:
        print(f"Error saving fitness history: {e}")

    # This plot will only show if run with a GUI environment.
    plt.figure(figsize=(8,4)); plt.plot(best_fitness_history, marker='o'); plt.title("Genetic Algorithm Performance"); plt.xlabel("Generation"); plt.ylabel("Best Fitness Score (Infections + Deaths)"); plt.grid(alpha=0.2); plt.show()
