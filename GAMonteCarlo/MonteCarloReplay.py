import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import sys
import os

# -------------------- UTILITY PARAMETERS --------------------
grid_size = 2
n_areas = 4 # Number of areas in the simulation
state_names = ['S','E','I','R','D','V']
state_colors = ['tab:blue','tab:orange','tab:red','tab:green','k','tab:purple']
area_population = [100, 100, 100, 100]
total_population = sum(area_population)
area_w = 12.0
area_h = 12.0
interval_ms = int(os.getenv('REPLAY_INTERVAL_MS', '100'))

# area coords
area_coords = {}
for idx in range(n_areas):
    col = idx % grid_size
    row = idx // grid_size
    area_coords[idx] = (col*area_w, row*area_h)

# -------------------- END PARAMETERS --------------------

def plot_results(data, label=""):
    """Plots the results from the simulation data."""
    # Unpack the data
    history_per_area = data['history_per_area'].item()
    history_total = data['history_total'].item()
    vaccine_stock_history = data['vaccine_stock_history']
    agent_history = data['agent_history']
    steps = len(vaccine_stock_history[0]) if len(vaccine_stock_history) > 0 else 0
    
    # Load the distribution and format it for the title
    distribution = data.get('distribution', None) # Use .get() for backward compatibility
    dist_str = ""
    if distribution is not None and len(distribution) > 0:
        dist_str = ", ".join([f'{d*100:.1f}%' for d in distribution])
        dist_str = f"\nDistribution: {dist_str}"

    # --- Animation Replay ---
    fig, ax = plt.subplots(figsize=(7,7))
    ax.set_xlim(0, grid_size * area_w); ax.set_ylim(0, grid_size * area_h); ax.set_aspect('equal')
    for i in range(grid_size+1):
        ax.axvline(i*area_w, color='gray', lw=0.9); ax.axhline(i*area_h, color='gray', lw=0.9)
    for aidx in range(n_areas):
        x0,y0 = area_coords[aidx]; ax.text(x0 + area_w*0.5, y0 + area_h*0.9, f"Area {aidx}\nPop {area_population[aidx]}", ha='center', va='top', fontsize=8, color='dimgray')
    
    # Initialize a list of scatter points, one for each agent
    num_agents = len(agent_history[0])
    initial_x = [d['x'] for d in agent_history[0]]
    initial_y = [d['y'] for d in agent_history[0]]
    initial_colors = [state_colors[d['state']] for d in agent_history[0]]
    sc = ax.scatter(initial_x, initial_y, c=initial_colors, s=16)

    # Legend for the states
    legend_lines = [plt.Line2D([0],[0], marker='o', color='w', markerfacecolor=state_colors[i], markersize=8) for i in range(6)]
    ax.legend(legend_lines, state_names, loc='upper right', title='States')
    ax.set_title(f"{dist_str}")
    
    # Day text
    day_text = ax.text(0.02, 0.98, '', transform=ax.transAxes, va='top', fontsize=9)
    
    # Per-area vaccine stock text displays
    stock_text_display = [ax.text(x0 + area_w * 0.5, y0 + area_h * 0.8, '', ha='center', va='top', fontsize=8) for aidx, (x0, y0) in area_coords.items()]

    def anim_update(frame):
        # Update positions and colors from the pre-computed history
        if frame < steps:
            x_coords = [d['x'] for d in agent_history[frame]]
            y_coords = [d['y'] for d in agent_history[frame]]
            colors = [state_colors[d['state']] for d in agent_history[frame]]
            sc.set_offsets(np.c_[x_coords, y_coords])
            sc.set_color(colors)
            
            # Update day text
            day_text.set_text(f"Day {frame+1}")

            # Update per-area vaccine stock
            for aidx in range(n_areas):
                current_stock = vaccine_stock_history[aidx][frame]
                stock_text_display[aidx].set_text(f"Stock: {current_stock}")

        return (sc, day_text) + tuple(stock_text_display)
    
    ani = animation.FuncAnimation(fig, anim_update, frames=steps, interval=interval_ms, blit=True, repeat=False)
    plt.show()

    # --- Static Plots ---
    fig2, axs = plt.subplots(3,2, figsize=(12,10)); axs = axs.flatten()
    for aidx in range(n_areas):
        axp = axs[aidx]
        for s_idx,sname in enumerate(state_names):
            # Accessing data saved in the dictionary format
            axp.plot(history_per_area[f"area_{aidx}_{sname}"], color=state_colors[s_idx], label=sname)
        axp.set_title(f"Area {aidx} ({label})"); axp.set_xlabel("Day"); axp.set_ylabel("Count")
        axp.set_ylim(0, max(area_population[aidx], 10)); axp.legend(fontsize=8)
    ax_tot = axs[4]
    for s_idx,sname in enumerate(state_names):
        ax_tot.plot(history_total[sname], color=state_colors[s_idx], label=sname, linewidth=1.5)
    ax_tot.set_title(f"Total across all areas ({label})"); ax_tot.set_xlabel("Day"); ax_tot.set_ylabel("Count")
    ax_tot.set_ylim(0, total_population + 10); ax_tot.legend(fontsize=9)
    axs[5].axis('off')
    plt.tight_layout(); plt.show()

    plt.figure(figsize=(8,3)); plt.plot(vaccine_stock_history.T, label=[f'Area {i}' for i in range(n_areas)]); plt.xlabel('Day'); plt.ylabel('Stock'); plt.title(f"Vaccine stock over time ({label})"); plt.grid(alpha=0.2); plt.legend(); plt.show()

if __name__ == "__main__":
    if len(sys.argv) > 1:
        filename = sys.argv[1]
        try:
            print(f"Replaying plots and animation for {filename}...")
            data = np.load(filename, allow_pickle=True)
            
            try:
                parts = filename.split('_')
                gen_num = parts[1]
                fitness_score = parts[3].replace('.npz', '')
                label = f"Generation {gen_num}, Fitness {fitness_score}"
            except (IndexError, ValueError):
                label = filename.replace('.npz', '')
                
            plot_results(data, label=label)
            
        except FileNotFoundError:
            print(f"Error: The file '{filename}' was not found.")
            sys.exit(1)
