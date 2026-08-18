
import numpy as np
import matplotlib.pyplot as plt
from model import EvaporationPondModel
from implicit_euler import ImplicitEuler
from dirk import DIRK2


def create_all_plots():


    print("Creating visualizations...")

    # Run simulations
    model = EvaporationPondModel()
    y0 = model.get_initial_conditions()
    t_span = (0.0, 240.0)
    h = 0.5

    # Run all methods
    results = {}

    solver1 = ImplicitEuler(model, solver_type='fixed_point')
    t1, y1, stats1 = solver1.solve(t_span, y0, h)
    results['IE-FP'] = {'t': t1, 'y': y1, 'stats': stats1, 'color': 'blue', 'style': '-'}

    solver2 = ImplicitEuler(model, solver_type='newton')
    t2, y2, stats2 = solver2.solve(t_span, y0, h)
    results['IE-Newton'] = {'t': t2, 'y': y2, 'stats': stats2, 'color': 'red', 'style': '--'}

    solver3 = DIRK2(model, solver_type='fixed_point')
    t3, y3, stats3 = solver3.solve(t_span, y0, h)
    results['DIRK2-FP'] = {'t': t3, 'y': y3, 'stats': stats3, 'color': 'green', 'style': '-.'}

    solver4 = DIRK2(model, solver_type='newton')
    t4, y4, stats4 = solver4.solve(t_span, y0, h)
    results['DIRK2-Newton'] = {'t': t4, 'y': y4, 'stats': stats4, 'color': 'purple', 'style': ':'}

    # Create figure with subplots
    fig = plt.figure(figsize=(16, 10))

    # Plot 1: Temperature vs Time
    ax1 = plt.subplot(2, 3, 1)
    for name, data in results.items():
        t = data['t'] / 24  # Convert to days
        T = data['y'][:, 0]
        ax1.plot(t, T, label=name, color=data['color'], linestyle=data['style'], linewidth=2)
    ax1.set_xlabel('Time (days)', fontsize=12)
    ax1.set_ylabel('Temperature (°C)', fontsize=12)
    ax1.set_title('Temperature Evolution', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)

    # Plot 2: Salinity vs Time
    ax2 = plt.subplot(2, 3, 2)
    for name, data in results.items():
        t = data['t'] / 24
        S = data['y'][:, 1]
        ax2.plot(t, S, label=name, color=data['color'], linestyle=data['style'], linewidth=2)
    ax2.set_xlabel('Time (days)', fontsize=12)
    ax2.set_ylabel('Salinity (g/L)', fontsize=12)
    ax2.set_title('Salinity Evolution', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)

    # Plot 3: Volume vs Time
    ax3 = plt.subplot(2, 3, 3)
    for name, data in results.items():
        t = data['t'] / 24
        V = data['y'][:, 2]
        ax3.plot(t, V, label=name, color=data['color'], linestyle=data['style'], linewidth=2)
    ax3.set_xlabel('Time (days)', fontsize=12)
    ax3.set_ylabel('Volume (m³)', fontsize=12)
    ax3.set_title('Volume Evolution', fontsize=14, fontweight='bold')
    ax3.legend(fontsize=10)
    ax3.grid(True, alpha=0.3)

    # Plot 4: Phase Plot (Temperature vs Salinity)
    ax4 = plt.subplot(2, 3, 4)
    for name, data in results.items():
        T = data['y'][:, 0]
        S = data['y'][:, 1]
        ax4.plot(T, S, label=name, color=data['color'], linestyle=data['style'], linewidth=2)
        ax4.plot(T[0], S[0], 'o', color=data['color'], markersize=8, label=f'{name} start')
        ax4.plot(T[-1], S[-1], 's', color=data['color'], markersize=8, label=f'{name} end')
    ax4.set_xlabel('Temperature (°C)', fontsize=12)
    ax4.set_ylabel('Salinity (g/L)', fontsize=12)
    ax4.set_title('Phase Plot: T vs S', fontsize=14, fontweight='bold')
    ax4.grid(True, alpha=0.3)

    # Plot 5: Computational Cost Comparison
    ax5 = plt.subplot(2, 3, 5)
    methods = list(results.keys())
    cpu_times = [results[m]['stats']['solve_time'] for m in methods]
    colors = [results[m]['color'] for m in methods]
    bars = ax5.bar(methods, cpu_times, color=colors, alpha=0.7, edgecolor='black')
    ax5.set_ylabel('CPU Time (seconds)', fontsize=12)
    ax5.set_title('Computational Cost Comparison', fontsize=14, fontweight='bold')
    ax5.tick_params(axis='x', rotation=45)
    ax5.grid(True, alpha=0.3, axis='y')
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        ax5.text(bar.get_x() + bar.get_width() / 2., height,
                 f'{height:.3f}s', ha='center', va='bottom', fontsize=10)

    # Plot 6: Average Iterations Comparison
    ax6 = plt.subplot(2, 3, 6)
    avg_iters = [results[m]['stats']['avg_iterations'] for m in methods]
    bars = ax6.bar(methods, avg_iters, color=colors, alpha=0.7, edgecolor='black')
    ax6.set_ylabel('Average Iterations per Step', fontsize=12)
    ax6.set_title('Convergence Speed Comparison', fontsize=14, fontweight='bold')
    ax6.tick_params(axis='x', rotation=45)
    ax6.grid(True, alpha=0.3, axis='y')
    # Add value labels
    for bar in bars:
        height = bar.get_height()
        ax6.text(bar.get_x() + bar.get_width() / 2., height,
                 f'{height:.2f}', ha='center', va='bottom', fontsize=10)

    plt.tight_layout()
    plt.savefig('all_results.png', dpi=300, bbox_inches='tight')
    print("✓ Saved: all_results.png")

    # Create separate detailed plot for temperature with day/night cycle
    fig2, ax = plt.subplots(figsize=(12, 6))

    # Plot temperature
    for name, data in results.items():
        t = data['t'] / 24
        T = data['y'][:, 0]
        ax.plot(t, T, label=name, color=data['color'], linestyle=data['style'], linewidth=2.5)

    # Add shaded regions for night time
    for day in range(11):
        ax.axvspan(day, day + 6 / 24, alpha=0.1, color='gray', label='Night' if day == 0 else '')
        ax.axvspan(day + 18 / 24, day + 1, alpha=0.1, color='gray')

    ax.set_xlabel('Time (days)', fontsize=14)
    ax.set_ylabel('Temperature (°C)', fontsize=14)
    ax.set_title('Temperature Evolution with Day/Night Cycle', fontsize=16, fontweight='bold')
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('temperature_detailed.png', dpi=300, bbox_inches='tight')
    print("✓ Saved: temperature_detailed.png")

    # Create error plot
    fig3, axes = plt.subplots(1, 3, figsize=(15, 5))

    ref = results['DIRK2-Newton']['y']
    t_ref = results['DIRK2-Newton']['t'] / 24

    for idx, (var_name, var_idx) in enumerate([('Temperature', 0), ('Salinity', 1), ('Volume', 2)]):
        ax = axes[idx]
        for name, data in results.items():
            if name == 'DIRK2-Newton':
                continue
            y = data['y']
            error = np.abs(y[:, var_idx] - ref[:, var_idx])
            ax.semilogy(t_ref, error, label=name, color=data['color'],
                        linestyle=data['style'], linewidth=2)

        ax.set_xlabel('Time (days)', fontsize=12)
        ax.set_ylabel(f'Absolute Error', fontsize=12)
        ax.set_title(f'{var_name} Error vs Reference', fontsize=13, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3, which='both')

    plt.tight_layout()
    plt.savefig('error_analysis.png', dpi=300, bbox_inches='tight')
    print("✓ Saved: error_analysis.png")

    print("\n✓ All visualizations created successfully!")
    print("  - all_results.png (main 6-plot figure)")
    print("  - temperature_detailed.png (temperature with day/night)")
    print("  - error_analysis.png (error comparison)")


if __name__ == "__main__":
    create_all_plots()