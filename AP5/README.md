# AP5 — Parametric Splines for Letter Shapes

Fits linear, cubic, and B-spline curves to hand-picked control points for the letters **C**, **S**, and **U**, then compares smoothness and fitting accuracy.

## What it does

- Defines control points for letters C, S, and U.
- Fits three spline types through each letter's points: piecewise linear, natural cubic spline, and B-spline.
- Computes a **smoothness metric** (mean curvature) and a **fitting error** (average distance from control points to the fitted curve) for each spline type.
- Visualizes each letter with all three spline types side by side.
- Compares all three letters together using cubic splines.
- Analyzes how the number of control points affects fit quality for the letter S.
- Prints a summary table of smoothness/error across all letters and spline types.

## Run it

```bash
pip install numpy scipy matplotlib
python main.py
```

## Output

PNG figures saved to this folder:
- `letter_C_splines.png`, `letter_S_splines.png`, `letter_U_splines.png` — per-letter spline comparison
- `all_letters_comparison.png` — all three letters using cubic splines
- `node_density_analysis_S.png` — effect of control point count on the letter S

Plus a console summary of smoothness and fitting error per letter/spline type.

## Key findings

- Cubic splines give the smoothest curves (lowest curvature).
- All spline types pass near the control points (low fitting error).
- Letter S needs the most control points due to its double curvature.
- Linear splines show higher curvature because of their sharp corners.
