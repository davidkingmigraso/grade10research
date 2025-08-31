import numpy as np
import matplotlib.pyplot as plt

# Load the saved data
data = np.load('genetic_algorithm_performance_data.npz')
best_fitness_history = data['best_fitness_history']

# You can now use the loaded data to replot or analyze
plt.figure(figsize=(8,4))
plt.plot(best_fitness_history, marker='o')
plt.title("Genetic Algorithm Performance (Loaded)")
plt.xlabel("Generation")
plt.ylabel("Best Fitness Score (Infections + Deaths)")
plt.grid(alpha=0.2)
plt.show()