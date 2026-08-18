# AP1 — Vector & Matrix Norms

Computes and compares distance metrics between random vectors and matrices, then visualizes the geometry of unit balls under different norms.

## What it does

- Generates two random 4-element vectors and reshapes them into 2x2 matrices.
- Computes vector distances: L1 (Manhattan) and L2 (Euclidean).
- Computes matrix distances: induced 1-norm and Frobenius norm.
- Plots 2D cross-sections of the L1 and L2 unit balls side by side.

## Run it

```bash
pip install numpy matplotlib
python ap1.py
```

## Output

A matplotlib figure with two subplots: the L1 unit ball ("diamond") and the L2 unit ball ("circle").
