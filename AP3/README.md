# AP3 — Numerical Differentiation: Tangents & Normals

Compares finite-difference schemes (forward, central, five-point) against exact derivatives for a 1D and a 2D function, then visualizes the resulting tangent line and tangent plane.

## What it does

- **1D:** approximates `d/dx sin(x)` at `x0 = 1.0` using forward, central, and five-point differences across a range of step sizes `h`, and compares against the exact derivative `cos(x)`.
- **2D:** approximates the partial derivatives of a Gaussian bump `F(x, y) = exp(-(x-1)^2 - (y+0.5)^2)` the same way.
- Computes absolute error vs. step size `h` for every method and saves error tables to CSV.
- Plots the tangent line to `sin(x)` at `x0`, and the tangent plane to `F(x, y)` at `(x0, y0)`.

## Run it

```bash
pip install numpy pandas matplotlib
python ap3.py
```

## Output

Everything is written to a generated `normals_tangents_results/` folder (not tracked in git):
- `errors_1d.csv`, `errors_2d.csv` — error tables
- `error_vs_h_1d.png`, `error_vs_h_2d.png` — log-log error plots
- `tangent_line.png`, `tangent_plane.png` — geometric visualizations

`ap3.pdf` in this folder is the write-up/report for the assignment.
