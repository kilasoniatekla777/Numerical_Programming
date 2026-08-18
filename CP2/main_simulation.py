
import numpy as np
import matplotlib.pyplot as plt
from model import EvaporationPondModel
from implicit_euler import ImplicitEuler
from dirk import DIRK2
import json


def run_all_simulations():


    print("=" * 60)
    print("EVAPORATION POND SIMULATION")
    print("=" * 60)

    # Create model
    model = EvaporationPondModel()

    # Initial conditions
    y0 = model.get_initial_conditions()
    print(f"\nInitial Conditions:")
    print(f"  Temperature: {y0[0]:.2f} °C")
    print(f"  Salinity: {y0[1]:.2f} g/L")
    print(f"  Volume: {y0[2]:.2f} m³")

    # Time span (10 days in hours)
    t_span = (0.0, 240.0)  # 10 days
    h = 0.5  # Time step: 0.5 hour

    print(f"\nSimulation Parameters:")
    print(f"  Time span: {t_span[0]} to {t_span[1]} hours ({t_span[1] / 24:.1f} days)")
    print(f"  Time step: {h} hours")
    print()

    # Dictionary to store all results
    results = {}

    # 1. Implicit Euler with Fixed-Point
    print("Running: Implicit Euler with Fixed-Point Iteration...")
    solver1 = ImplicitEuler(model, solver_type='fixed_point')
    t1, y1, stats1 = solver1.solve(t_span, y0, h)
    results['IE_FP'] = {'t': t1, 'y': y1, 'stats': stats1}
    print(f"  ✓ Completed in {stats1['solve_time']:.3f}s")
    print(f"    Avg iterations/step: {stats1['avg_iterations']:.2f}")

    # 2. Implicit Euler with Newton
    print("\nRunning: Implicit Euler with Newton Method...")
    solver2 = ImplicitEuler(model, solver_type='newton')
    t2, y2, stats2 = solver2.solve(t_span, y0, h)
    results['IE_Newton'] = {'t': t2, 'y': y2, 'stats': stats2}
    print(f"  ✓ Completed in {stats2['solve_time']:.3f}s")
    print(f"    Avg iterations/step: {stats2['avg_iterations']:.2f}")

    # 3. DIRK2 with Fixed-Point
    print("\nRunning: DIRK2 with Fixed-Point Iteration...")
    solver3 = DIRK2(model, solver_type='fixed_point')
    t3, y3, stats3 = solver3.solve(t_span, y0, h)
    results['DIRK2_FP'] = {'t': t3, 'y': y3, 'stats': stats3}
    print(f"  ✓ Completed in {stats3['solve_time']:.3f}s")
    print(f"    Avg iterations/step: {stats3['avg_iterations']:.2f}")

    # 4. DIRK2 with Newton
    print("\nRunning: DIRK2 with Newton Method...")
    solver4 = DIRK2(model, solver_type='newton')
    t4, y4, stats4 = solver4.solve(t_span, y0, h)
    results['DIRK2_Newton'] = {'t': t4, 'y': y4, 'stats': stats4}
    print(f"  ✓ Completed in {stats4['solve_time']:.3f}s")
    print(f"    Avg iterations/step: {stats4['avg_iterations']:.2f}")

    print("\n" + "=" * 60)
    print("All simulations completed!")
    print("=" * 60)

    return results, model


def print_comparison_table(results):
    """Print comparison table"""
    print("\n" + "=" * 80)
    print("COMPARISON TABLE")
    print("=" * 80)
    print(f"{'Method':<20} {'Time Step':<12} {'Avg Iter/Step':<15} {'CPU Time (s)':<15} {'Conv. Failures':<15}")
    print("-" * 80)

    for name, data in results.items():
        stats = data['stats']
        method_name = name.replace('_', ' ')
        print(f"{method_name:<20} {'0.5 h':<12} {stats['avg_iterations']:<15.2f} "
              f"{stats['solve_time']:<15.3f} {stats.get('convergence_failures', 0):<15}")

    print("=" * 80)


def compute_errors(results):
    """Compute errors between methods using DIRK2-Newton as reference"""
    print("\n" + "=" * 80)
    print("ERROR ANALYSIS (vs DIRK2-Newton as reference)")
    print("=" * 80)

    ref = results['DIRK2_Newton']['y']

    print(f"{'Method':<20} {'T Error (°C)':<15} {'S Error (g/L)':<15} {'V Error (m³)':<15}")
    print("-" * 80)

    for name, data in results.items():
        if name == 'DIRK2_Newton':
            continue

        y = data['y']

        # Compute max absolute error
        T_error = np.max(np.abs(y[:, 0] - ref[:, 0]))
        S_error = np.max(np.abs(y[:, 1] - ref[:, 1]))
        V_error = np.max(np.abs(y[:, 2] - ref[:, 2]))

        method_name = name.replace('_', ' ')
        print(f"{method_name:<20} {T_error:<15.4e} {S_error:<15.4e} {V_error:<15.4e}")

    print("=" * 80)


def save_results_to_file(results):
    """Save numerical results to text file"""
    with open('results_summary.txt', 'w') as f:
        f.write("EVAPORATION POND SIMULATION RESULTS\n")
        f.write("=" * 60 + "\n\n")

        for name, data in results.items():
            f.write(f"\n{name}:\n")
            f.write(f"  Final Temperature: {data['y'][-1, 0]:.4f} °C\n")
            f.write(f"  Final Salinity: {data['y'][-1, 1]:.4f} g/L\n")
            f.write(f"  Final Volume: {data['y'][-1, 2]:.4f} m³\n")
            f.write(f"  CPU Time: {data['stats']['solve_time']:.4f} s\n")
            f.write(f"  Avg Iterations: {data['stats']['avg_iterations']:.2f}\n")

    print("\n✓ Results saved to 'results_summary.txt'")


if __name__ == "__main__":
    # Run all simulations
    results, model = run_all_simulations()

    # Print comparison table
    print_comparison_table(results)

    # Compute errors
    compute_errors(results)

    # Save results
    save_results_to_file(results)

    print("\n✓ Ready for visualization! Run 'python visualize.py' next.")