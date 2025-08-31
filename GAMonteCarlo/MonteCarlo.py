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
beta = 0.25        # chance that susceptible will be exposed
sigma = 1/5.2      # chance that exposed will be infected after incubation_days
gamma = 1/21.0     # recovery rate
mu = 0.005         # chance of infected to dead

# mobility
move_scale = 0.6
area_border_cross_prob = 0.01

# Genetic Algorithm & Monte Carlo parameters
population_size = 50       # Number of different vaccine distribution strategies to test
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
    __slots__ = ('id','area','x','y','state','days','vaccinated')
    def __init__(self, aid, area):
        self.id = aid
        self.area = area
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

def run_simulation(vaccine_distribution, area_population, steps, hesitancy_rate, vaccine_production_rate, vaccine_spoilage_rate, vaccine_total, vaccination_delay_days, animate_and_plot=False):
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

    # New: Per-area vaccine stock and history
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
            # New: Distribute production to each area's stock
            for aidx in range(n_areas):
                doses_to_distribute = int(vaccine_total * vaccine_distribution[aidx])

                # Check if enough vaccines are available
                if vaccine_total >= doses_to_distribute:
                    vaccine_stock[aidx] += doses_to_distribute
                    vaccine_total -= doses_to_distribute

                # New: Vaccinate from the local area stock
                if vaccine_stock[aidx] > 0:
                    susceptibles_in_area = [p for p in agents if p.state == S and p.area == aidx]
                    if susceptibles_in_area:
                        doses_to_use = min(len(susceptibles_in_area), vaccine_stock[aidx])
                        chosen = np.random.choice(susceptibles_in_area, doses_to_use, replace=False)
                        for p in chosen:
                            if np.random.rand() > hesitancy_rate:
                                p.state = V; p.vaccinated = True; vaccine_stock[aidx] -= 1

        # New: Handle spoilage per area
        for aidx in range(n_areas):
            spoiled = int(vaccine_stock[aidx] * vaccine_spoilage_rate)
            vaccine_stock[aidx] = max(0, vaccine_stock[aidx] - spoiled)

        # New: Record per-area vaccine stock history
        for aidx in range(n_areas):
            vaccine_stock_history[aidx].append(vaccine_stock[aidx])

        # Increase vaccine_total by vaccine_production_rate
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

    if animate_and_plot:
        # Code to handle animation and plots
        fig, ax = plt.subplots(figsize=(7,7))
        ax.set_xlim(0, grid_size * area_w); ax.set_ylim(0, grid_size * area_h); ax.set_aspect('equal')
        for i in range(grid_size+1):
            ax.axvline(i*area_w, color='gray', lw=0.9); ax.axhline(i*area_h, color='gray', lw=0.9)
        for aidx in range(n_areas):
            x0,y0 = area_coords[aidx]; ax.text(x0 + area_w*0.5, y0 + area_h*0.9, f"Area {aidx}\nPop {area_population[aidx]}", ha='center', va='top', fontsize=8, color='dimgray')
        sc = ax.scatter([p.x for p in agents],[p.y for p in agents], c=[state_colors[p.state] for p in agents], s=16)
        legend_lines = [plt.Line2D([0],[0], marker='o', color='w', markerfacecolor=state_colors[i], markersize=8) for i in range(6)]
        ax.legend(legend_lines, state_names, loc='upper right', title='States')

        # New: Display per-area vaccine stock
        stock_text_display = [ax.text(x0 + area_w*0.5, y0 + area_h*0.8, '', ha='center', va='top', fontsize=8) for aidx, (x0,y0) in area_coords.items()]
        day_text = ax.text(0.02, 0.98, '', transform=ax.transAxes, va='top', fontsize=9)

        def anim_update(frame):
            sim_step()
            x_coords = [a.x for a in agents]
            y_coords = [a.y for a in agents]
            colors = [state_colors[a.state] for a in agents]
            sc.set_offsets(np.c_[x_coords, y_coords])
            sc.set_color(colors)

            # Update day and stock display
            day_text.set_text(f"Day {day_counter}")
            for aidx in range(n_areas):
                stock_text_display[aidx].set_text(f"Stock: {vaccine_stock[aidx]}")

            return (sc, day_text) + tuple(stock_text_display)

        ani = animation.FuncAnimation(fig, anim_update, frames=steps, interval=50, blit=True, repeat=False)
        plt.show()

        fig2, axs = plt.subplots(3,2, figsize=(12,10)); axs = axs.flatten()
        for aidx in range(n_areas):
            axp = axs[aidx]
            for s_idx,sname in enumerate(state_names):
                axp.plot(history_per_area[aidx][sname], color=state_colors[s_idx], label=sname)
            axp.set_title(f"Area {aidx} (pop {area_population[aidx]})"); axp.set_xlabel("Day"); axp.set_ylabel("Count")
            axp.set_ylim(0, max(area_population[aidx], 10)); axp.legend(fontsize=8)
        ax_tot = axs[4]
        for s_idx,sname in enumerate(state_names):
            ax_tot.plot(history_total[sname], color=state_colors[s_idx], label=sname, linewidth=1.5)
        ax_tot.set_title("Total across all areas"); ax_tot.set_xlabel("Day"); ax_tot.set_ylabel("Count")
        ax_tot.set_ylim(0, total_population + 10); ax_tot.legend(fontsize=9)
        axs[5].axis('off')
        plt.tight_layout(); plt.show()

        # New: Plot per-area vaccine stock
        plt.figure(figsize=(8,4)); plt.title("Vaccine stock over time by Area"); plt.xlabel("Day"); plt.ylabel("Stock"); plt.grid(alpha=0.2)
        for aidx in range(n_areas):
            plt.plot(vaccine_stock_history[aidx], label=f'Area {aidx}');
        plt.legend(); plt.show()

    else:
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

def save_data(data, filename):
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
        agent_history=data['agent_history']
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
    return total_infected + total_deceased

def fitness(distribution, area_population, steps, hesitancy_rate, vaccine_production_rate, vaccine_spoilage_rate, vaccine_total, vaccination_delay_days, monte_carlo_runs):
    """Calculates fitness by running Monte Carlo simulations in parallel."""
    params_list = [(distribution, area_population, steps, hesitancy_rate, vaccine_production_rate, vaccine_spoilage_rate, vaccine_total, vaccination_delay_days)] * monte_carlo_runs
    with Pool(num_processes) as p:
        scores = p.map(_evaluate_distribution_for_run, params_list)
    return sum(scores) / monte_carlo_runs

def select_parents(population, fitness_scores):
    # Tournament selection
    parents = []
    for _ in range(2):
        contenders = random.sample(list(zip(population, fitness_scores)), 5)
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

def run_ga(area_population, steps, generations, hesitancy_rate, vaccine_production_rate, vaccine_spoilage_rate, vaccine_total, vaccination_delay_days, monte_carlo_runs):
    print("Starting genetic algorithm to find optimal vaccine distribution...")
    population = initialize_population(population_size)
    best_fitness_history = []

    for generation in range(generations):
        fitness_scores = [fitness(dist, area_population, steps, hesitancy_rate, vaccine_production_rate, vaccine_spoilage_rate, vaccine_total, vaccination_delay_days, monte_carlo_runs) for dist in population]

        best_fitness = min(fitness_scores)
        best_dist = population[fitness_scores.index(best_fitness)]
        best_fitness_history.append(best_fitness)

        print(f"Generation {generation+1}/{generations}: Best Fitness = {best_fitness:.2f}, Best Dist = {[f'{x:.2f}' for x in best_dist]}")

        new_population = [best_dist] # Elitism

        while len(new_population) < population_size:
            parent1, parent2 = select_parents(population, fitness_scores)
            child = crossover(parent1, parent2)
            mutated_child = mutate(child, mutation_rate)
            new_population.append(mutated_child)

        population = new_population

    print("\nGenetic Algorithm complete. Finding optimal and worst distribution...")
    final_fitness_scores = [fitness(dist, area_population, steps, hesitancy_rate, vaccine_production_rate, vaccine_spoilage_rate, vaccine_total, vaccination_delay_days, monte_carlo_runs) for dist in population]
    optimal_distribution = population[final_fitness_scores.index(min(final_fitness_scores))]
    worst_distribution = population[final_fitness_scores.index(max(final_fitness_scores))]

    print(f"\nOptimal Vaccine Distribution found: {[f'{x:.2f}' for x in optimal_distribution]}")
    print(f"\nWorst Vaccine Distribution found: {[f'{x:.2f}' for x in worst_distribution]}")

    # Run and save final best and worst runs for replay
    print("\nRunning and saving final best and worst strategies...")
    best_run_data = run_simulation(optimal_distribution, area_population, steps, hesitancy_rate, vaccine_production_rate, vaccine_spoilage_rate, vaccine_total, vaccination_delay_days, animate_and_plot=False)
    save_data(best_run_data, "best_strategy_results.npz")

    worst_run_data = run_simulation(worst_distribution, area_population, steps, hesitancy_rate, vaccine_production_rate, vaccine_spoilage_rate, vaccine_total, vaccination_delay_days, animate_and_plot=False)
    save_data(worst_run_data, "worst_strategy_results.npz")

    print("Simulation data for best and worst strategies saved.")

    return best_fitness_history

def parse_args():
    parser = argparse.ArgumentParser(description="Run the Monte Carlo/Genetic Algorithm simulation.")
    parser.add_argument("--area_population", type=str, default="100,100,100,100", help="Comma-separated list of population for each area.")
    parser.add_argument("--steps", type=int, default=365, help="Number of simulation steps (days).")
    parser.add_argument("--generations", type=int, default=30, help="Number of genetic algorithm generations.")
    parser.add_argument("--monte_carlo_runs", type=int, default=10, help="Number of simulations to run per strategy for a robust average.")
    parser.add_argument("--hesitancy_rate", type=float, default=0.2, help="Rate of vaccine hesitancy.")
    parser.add_argument("--vaccine_production_rate", type=int, default=2, help="Daily vaccine production rate.")
    parser.add_argument("--vaccine_spoilage_rate", type=float, default=0.01, help="Daily vaccine spoilage rate.")
    parser.add_argument("--vaccine_total", type=int, default=200, help="Initial total vaccine stock.")
    parser.add_argument("--vaccination_delay_days", type=int, default=7, help="Days before vaccination begins.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    # Update parameters with command-line arguments
    area_population = [int(p) for p in args.area_population.split(',')]
    steps = args.steps
    generations = args.generations
    monte_carlo_runs = args.monte_carlo_runs
    hesitancy_rate = args.hesitancy_rate
    vaccine_production_rate = args.vaccine_production_rate
    vaccine_spoilage_rate = args.vaccine_spoilage_rate
    vaccine_total = args.vaccine_total
    vaccination_delay_days = args.vaccination_delay_days

    total_population = sum(area_population)

    best_fitness_history = run_ga(area_population, steps, generations, hesitancy_rate, vaccine_production_rate, vaccine_spoilage_rate, vaccine_total, vaccination_delay_days, monte_carlo_runs)

    try:
        np.savez_compressed('genetic_algorithm_performance_data.npz', best_fitness_history=np.array(best_fitness_history))
        print("Best fitness history saved to genetic_algorithm_performance_data.npz")
    except Exception as e:
        print(f"Error saving fitness history: {e}")

    # This plot will only show if run with a GUI environment.
    plt.figure(figsize=(8,4)); plt.plot(best_fitness_history, marker='o'); plt.title("Genetic Algorithm Performance"); plt.xlabel("Generation"); plt.ylabel("Best Fitness Score (Infections + Deaths)"); plt.grid(alpha=0.2); plt.show()
