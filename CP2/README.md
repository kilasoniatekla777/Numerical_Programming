# CP2 — Evaporation Pond ODE Simulation

Simulates the temperature, salinity, and volume of an evaporating pond over time by solving a coupled ODE system with multiple implicit numerical methods, and compares their accuracy and cost.

## What it does

- **Model (`model.py`):** an `EvaporationPondModel` with the physics — solar heating (day/night cycle), evaporative cooling, convective heat loss, salinity-dependent specific heat, and volume loss from evaporation. Produces the ODE system `dy/dt = f(t, y)` for `y = [Temperature, Salinity, Volume]`.
- **Solvers:**
  - `implicit_euler.py` — Implicit (Backward) Euler, solved via either fixed-point iteration or Newton's method.
  - `dirk.py` — a 2-stage Diagonally Implicit Runge-Kutta method (SDIRK2), also solved via fixed-point or Newton.
- **`main_simulation.py`** runs all four combinations (IE+fixed-point, IE+Newton, DIRK2+fixed-point, DIRK2+Newton) over a 10-day span and reports final state, solve time, and average iterations per step for each.
- **`visualize.py`** re-runs the simulations and plots temperature/salinity/volume trajectories and error comparisons across methods.

## Run it

```bash
pip install numpy matplotlib
python main_simulation.py   # run all 4 solver combinations, print summary
python visualize.py         # generate comparison plots
```

## Output

- `results_summary.txt` — final state, CPU time, and average iteration count per solver
- `all_results.png`, `temperature_detailed.png`, `error_analysis.png` — comparison plots

## Results at a glance

All four methods converge to essentially the same final state (~19.76°C, ~35.31 g/L, ~99.13 m³ after 10 days), confirming the model is well-behaved under implicit integration. DIRK2 needs more iterations per step (4 vs 2) since it evaluates 2 internal stages, and Newton's method is consistently more expensive per step than fixed-point iteration despite converging in the same number of iterations for this problem.
